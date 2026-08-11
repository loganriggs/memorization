"""T20: super-unlearning round 2 + relearn-attack variants + weight
geometry. Training protocol matches t13/t17 (200 steps, AdamW 1e-5,
bs 8+8, clip 1.0, seed 0). New methods:

  flat_l1  — curriculum pin (100 steps) then pin + LAM1 * sum|grad CE_f|.
             Adam's step is ~lr*sign(g) per coordinate, so one-step CE
             decrease under Adam is ~lr*sum|g_i|: L1 grad norm is the
             Adam-invariant first-order flatness (flatten2's L2 was the
             SGD-invariant one, and failed).
  npo_sam  — NPO + retain-CE trained with SAM (rho=0.05): the
             literature's relearn-resilient baseline (SAM on the
             unlearning objective).
  pin_sam  — our pin gamma2 + hinge/KL trained with SAM.
  reoccupy — curriculum: unlearn 100 steps (pin + hinge/KL), then keep
             the pin while training the SAME questions to fixed random
             8-token answers — hard reoccupation of the forget keys
             with off-manifold content (v1 limitation: exact-key remap,
             not neighborhood saturation).

Stages:
  train <method>                        -> results/t20_<method>/
  relearn <model_dir> <tag> <lr> <src>  -> src in {forget, adjacent,
             para}: forget = forget01 itself; adjacent = 200 unseen
             retain facts (benign-data "jog" attack); para =
             paraphrased questions + true answers (familiarity
             control; contaminates para-eval for that throwaway run
             only). Forget R-L on ORIGINAL questions every 5 steps.
  wdist    -> ||theta - theta_base||_F for every unlearned checkpoint,
             joined with relearn steps (does resistance reduce to
             weight distance?). Appends to results/t20_methods.jsonl.
"""

import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM

import t11_tofu as t11
import t13_sweep as t13
import t15_tofu_metrics as t15
import t17_methods as t17
from t11_tofu import BASE_DIR, DEVICE, fact_margins, get_tok, make_batch

OUT = "results/t20_methods.jsonl"
STEPS, LR, LAMB, GAMMA = 200, 1e-5, 1.0, 2.0
NPO_BETA, SAM_RHO, LAM1 = 0.1, 0.05, 3e-6
# LAM1: sum|grad CE_f| ~ 1.06e6 at an unlearned 410M checkpoint, so
# 3e-6 puts the penalty at ~3 initially — same order as pin+retain.
# (flatten2's L2 penalty was accidentally ~1200x its other terms; its
# zero-resistance verdict is thereby a fortiori, and its retain damage
# is likely this over-weighting.)


def log(rec):
    with open(OUT, "a") as f:
        f.write(json.dumps(rec) + "\n")
    print(json.dumps(rec), flush=True)


def rand_answers(tok, n, seed=0, toks=8):
    g = torch.Generator().manual_seed(seed)
    return [tok.decode(torch.randint(1000, 40000, (toks,),
                                     generator=g).tolist()).strip()
            for _ in range(n)]


def stage_train():
    import datasets
    method = sys.argv[2]
    tok = get_tok()
    forget = list(datasets.load_dataset("locuslab/TOFU", "forget01",
                                        split="train"))
    retain = list(datasets.load_dataset("locuslab/TOFU", "retain99",
                                        split="train"))[:400]
    if method == "reoccupy":
        ra = rand_answers(tok, len(forget))
        reoc_rows = [{"question": r["question"], "answer": a}
                     for r, a in zip(forget, ra)]

    attn = ({"attn_implementation": "eager"}
            if method == "flat_l1" else {})
    model = AutoModelForCausalLM.from_pretrained(
        BASE_DIR, torch_dtype=torch.float32, **attn).to(DEVICE)
    base = AutoModelForCausalLM.from_pretrained(
        BASE_DIR, torch_dtype=torch.float32).to(DEVICE)
    base.eval()
    m0_r, _, _ = fact_margins(base, tok, retain)
    cap = float(np.median(m0_r))
    target_r = torch.tensor(np.minimum(m0_r, cap), dtype=torch.float32,
                            device=DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    g = torch.Generator().manual_seed(0)

    for step in range(STEPS):
        fi = torch.randperm(len(forget), generator=g)[:8].tolist()
        ri = torch.randperm(len(retain), generator=g)[:8].tolist()
        fids, flab, fm = make_batch(tok, [forget[j] for j in fi])
        rids, rlab, rm = make_batch(tok, [retain[j] for j in ri])

        def pin_loss():
            fam, _ = t13.tok_margins(model, fids, flab, fm)
            return torch.stack([F.relu(m + GAMMA).mean()
                                for m in fam]).mean()

        def our_retain():
            ram, lg_r = t13.tok_margins(model, rids, rlab, rm)
            mr = torch.stack([m.min() for m in ram])
            with torch.no_grad():
                lg_rb = base(input_ids=rids, attention_mask=rm).logits
            kl = F.kl_div(F.log_softmax(lg_r, -1),
                          F.log_softmax(lg_rb, -1), log_target=True,
                          reduction="none").sum(-1)
            kl = (kl * rm).sum() / rm.sum()
            tr = target_r[torch.tensor(ri, device=DEVICE)]
            return F.relu(tr - mr).mean() + LAMB * kl

        def compute_loss():
            if method == "flat_l1":
                floss = pin_loss()
                if step >= STEPS // 2:
                    hids, hlab, hm = make_batch(
                        tok, [forget[j] for j in fi[:2]])
                    ce_f = t11.batch_ce(model, hids, hlab, hm)
                    gs = torch.autograd.grad(
                        ce_f, [p for p in model.parameters()
                               if p.requires_grad], create_graph=True)
                    floss = floss + LAM1 * sum(gg.abs().sum()
                                               for gg in gs)
                return floss + our_retain()
            if method == "npo_sam":
                lp, _ = t17.seq_logps(model, fids, flab, fm)
                with torch.no_grad():
                    lpr, _ = t17.seq_logps(base, fids, flab, fm)
                floss = (-2 / NPO_BETA) * F.logsigmoid(
                    -NPO_BETA * (lp - lpr)).mean()
                return floss + t11.batch_ce(model, rids, rlab, rm)
            if method == "pin_sam":
                return pin_loss() + our_retain()
            if method == "reoccupy":
                floss = pin_loss()
                if step >= STEPS // 2:
                    dids, dlab, dm = make_batch(
                        tok, [reoc_rows[j] for j in fi])
                    floss = floss + t11.batch_ce(model, dids, dlab, dm)
                return floss + our_retain()
            raise ValueError(method)

        opt.zero_grad(set_to_none=True)
        loss = compute_loss()
        loss.backward()
        if method in ("npo_sam", "pin_sam"):  # SAM second pass
            with torch.no_grad():
                gn = torch.sqrt(sum((p.grad ** 2).sum()
                                    for p in model.parameters()
                                    if p.grad is not None))
                eps = []
                for p in model.parameters():
                    e = (SAM_RHO * p.grad / (gn + 1e-12)
                         if p.grad is not None else None)
                    eps.append(e)
                    if e is not None:
                        p.add_(e)
            opt.zero_grad(set_to_none=True)
            compute_loss().backward()
            with torch.no_grad():
                for p, e in zip(model.parameters(), eps):
                    if e is not None:
                        p.sub_(e)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 50 == 0:
            _, acc, _ = fact_margins(model, tok, forget[:20])
            print(f"{method} step {step} loss {float(loss):.3f} "
                  f"forget_acc {acc.mean():.2f}", flush=True)
    model.save_pretrained(f"results/t20_{method}")
    log({"stage": "train_done", "method": method})


def stage_relearn():
    import datasets
    model_dir, tag, lr, src = (sys.argv[2], sys.argv[3],
                               float(sys.argv[4]), sys.argv[5])
    tag = f"{tag}@{lr:g}/{src}"
    tok = get_tok()
    forget = list(datasets.load_dataset("locuslab/TOFU", "forget01",
                                        split="train"))
    if src == "forget":
        train_rows = forget
    elif src == "adjacent":
        train_rows = list(datasets.load_dataset(
            "locuslab/TOFU", "retain99", split="train"))[400:600]
    elif src == "para":
        pert = list(datasets.load_dataset("locuslab/TOFU",
                                          "forget01_perturbed",
                                          split="train"))
        train_rows = [{"question": r["paraphrased_question"],
                       "answer": r["answer"]} for r in pert]
    else:
        raise ValueError(src)
    model = AutoModelForCausalLM.from_pretrained(
        model_dir, torch_dtype=torch.float32).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    g = torch.Generator().manual_seed(0)
    half_base, curve, hit = 0.429, [], None

    def frl():
        model.eval()
        gens = t15.greedy_batch(model, tok, forget)
        return float(np.mean([t15.rouge_l_recall(gen, r["answer"])
                              for gen, r in zip(gens, forget)]))

    curve.append([0, round(frl(), 3)])
    for step in range(1, 31):
        model.train()
        fi = torch.randperm(len(train_rows), generator=g)[:8].tolist()
        ids, labels, mask = make_batch(tok, [train_rows[j] for j in fi])
        opt.zero_grad(set_to_none=True)
        t11.batch_ce(model, ids, labels, mask).backward()
        opt.step()
        if step % 5 == 0:
            rl = frl()
            curve.append([step, round(rl, 3)])
            print(f"relearn {tag} step {step} R-L {rl:.3f}", flush=True)
            if hit is None and rl >= half_base:
                hit = step
                if src == "forget":
                    break
    log({"stage": "t20_relearn", "tag": tag, "src": src,
         "steps_to_half_base": hit, "curve": curve})


def stage_wdist():
    base_sd = AutoModelForCausalLM.from_pretrained(
        BASE_DIR, torch_dtype=torch.float32).state_dict()
    rl = {}
    for f in ("results/t17_methods.jsonl",):
        for l in open(f):
            r = json.loads(l)
            if r["stage"] == "t17_relearn" and r["tag"].endswith("1e-05"):
                rl[r["tag"].split("@")[0]] = r["steps_to_half_base"]
    out = []
    for tag, d in [("npo", "results/t11_tofu_npo"),
                   ("all_g2", "results/t13_all_g2.0"),
                   ("all_g8", "results/t13_all_g8.0"),
                   ("pin_ce", "results/t17_pin_ce"),
                   ("simnpo", "results/t17_simnpo"),
                   ("decoy2", "results/t17_decoy2"),
                   ("flatten2", "results/t17_flatten2"),
                   ("rmu", "results/t16_rmu")]:
        sd = AutoModelForCausalLM.from_pretrained(
            d, torch_dtype=torch.float32).state_dict()
        dist = float(torch.sqrt(sum(
            ((sd[k].float() - base_sd[k].float()) ** 2).sum()
            for k in sd if sd[k].dtype.is_floating_point)))
        out.append({"tag": tag, "wdist": round(dist, 2),
                    "relearn_1e5": rl.get(tag)})
        print(out[-1], flush=True)
    log({"stage": "wdist", "rows": out})


if __name__ == "__main__":
    {"train": stage_train, "relearn": stage_relearn,
     "wdist": stage_wdist}[sys.argv[1]]()
