"""T17: fairness factorial + new baselines + relearn curves on
TOFU/Pythia-410M. All training matches t13 exactly: 200 steps, AdamW
lr 1e-5, bs 8 forget + 8 retain, grad clip 1.0, seed 0.

The 2x2 on {forget objective} x {retain handling} (bundles):
  npo + retain-CE        = t11 npo            (exists)
  pin(all,g2) + hinge+KL = t13 all_g2         (exists)
  npo + hinge+KL         = t17 npo_klhinge    (new: NPO with OUR anchor)
  pin(all,g2) + retain-CE= t17 pin_ce         (new: our pin, THEIR anchor)
Baselines beyond NPO:
  simnpo — SimNPO (Fan et al. 2024): reference-free, length-normalized
           sigmoid forget term (beta=2.5, delta=0) + retain-CE, per its
           standard TOFU setup. Hyperparams to be validated against
           open-unlearning configs before tuned comparisons.
  decoy  — super-unlearning pilot S1: CE toward TOFU's perturbed_answer
           (a plausible wrong answer) + our hinge+KL retain bundle.
RMU lives in t16_rmu.py.

Stages:
  train <method>            -> results/t17_<method>/
  relearn <model_dir> <tag> -> finetune on forget01, ROUGE-L curve every
                               5 steps (cache-free decoder), steps to
                               half-base (0.429); appends t17_relearn
                               records. Run on results/t15_retain_ref
                               for the never-knew control.
Appends to results/t17_methods.jsonl.
"""

import json
import sys

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM

import t11_tofu as t11
import t13_sweep as t13
import t15_tofu_metrics as t15
from t11_tofu import BASE_DIR, DEVICE, fact_margins, get_tok, make_batch

OUT = "results/t17_methods.jsonl"
STEPS, LR, LAMB, GAMMA, NPO_BETA, SIMNPO_BETA = 200, 1e-5, 1.0, 2.0, 0.1, 2.5


def log(rec):
    with open(OUT, "a") as f:
        f.write(json.dumps(rec) + "\n")
    print(json.dumps(rec), flush=True)


def seq_logps(model, ids, labels, mask):
    """Per-sequence summed answer logprob + token count."""
    lg = model(input_ids=ids, attention_mask=mask).logits
    lp = -F.cross_entropy(lg[:, :-1].reshape(-1, lg.shape[-1]),
                          labels[:, 1:].reshape(-1), ignore_index=-100,
                          reduction="none").reshape(ids.shape[0], -1)
    nmask = (labels[:, 1:] != -100).float()
    return (lp * nmask).sum(1), nmask.sum(1)


def stage_train():
    import datasets
    method = sys.argv[2]
    tok = get_tok()
    forget = list(datasets.load_dataset("locuslab/TOFU", "forget01",
                                        split="train"))
    retain = list(datasets.load_dataset("locuslab/TOFU", "retain99",
                                        split="train"))[:400]
    if method == "decoy":
        pert = list(datasets.load_dataset("locuslab/TOFU",
                                          "forget01_perturbed",
                                          split="train"))
        decoy_rows = [{"question": r["question"],
                       "answer": r["perturbed_answer"][0]} for r in pert]

    model = AutoModelForCausalLM.from_pretrained(
        BASE_DIR, torch_dtype=torch.float32).to(DEVICE)
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
        opt.zero_grad(set_to_none=True)

        # forget objective
        if method == "npo_klhinge":
            lp, _ = seq_logps(model, fids, flab, fm)
            with torch.no_grad():
                lpr, _ = seq_logps(base, fids, flab, fm)
            floss = (-2 / NPO_BETA) * F.logsigmoid(
                -NPO_BETA * (lp - lpr)).mean()
        elif method in ("pin_ce", "pin_klhinge"):
            fam, _ = t13.tok_margins(model, fids, flab, fm)
            floss = torch.stack([F.relu(m + GAMMA).mean()
                                 for m in fam]).mean()
        elif method == "simnpo":
            lp, n = seq_logps(model, fids, flab, fm)
            floss = (-2 / SIMNPO_BETA) * F.logsigmoid(
                -SIMNPO_BETA * lp / n).mean()
        elif method == "decoy":
            dids, dlab, dm = make_batch(tok, [decoy_rows[j] for j in fi])
            floss = t11.batch_ce(model, dids, dlab, dm)
        else:
            raise ValueError(method)

        # retain handling
        if method in ("pin_ce", "simnpo"):
            rloss = t11.batch_ce(model, rids, rlab, rm)
        else:  # our bundle: restoration hinge + KL anchor (t13 exactly)
            ram, lg_r = t13.tok_margins(model, rids, rlab, rm)
            mr = torch.stack([m.min() for m in ram])
            with torch.no_grad():
                lg_rb = base(input_ids=rids, attention_mask=rm).logits
            kl = F.kl_div(F.log_softmax(lg_r, -1),
                          F.log_softmax(lg_rb, -1), log_target=True,
                          reduction="none").sum(-1)
            kl = (kl * rm).sum() / rm.sum()
            tr = target_r[torch.tensor(ri, device=DEVICE)]
            rloss = F.relu(tr - mr).mean() + LAMB * kl

        (floss + rloss).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 50 == 0:
            _, acc, _ = fact_margins(model, tok, forget[:20])
            print(f"{method} step {step} floss {float(floss):.3f} "
                  f"rloss {float(rloss):.3f} "
                  f"forget_acc {acc.mean():.2f}", flush=True)
    model.save_pretrained(f"results/t17_{method}")
    log({"stage": "train_done", "method": method, "steps": STEPS})


def stage_relearn():
    import datasets
    model_dir, tag = sys.argv[2], sys.argv[3]
    tok = get_tok()
    forget = list(datasets.load_dataset("locuslab/TOFU", "forget01",
                                        split="train"))
    model = AutoModelForCausalLM.from_pretrained(
        model_dir, torch_dtype=torch.float32).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    g = torch.Generator().manual_seed(0)
    half_base = 0.429  # base forget R-L 0.858 / 2
    curve, hit = [], None
    model.eval()
    gens = t15.greedy_batch(model, tok, forget)
    r0 = float(np.mean([t15.rouge_l_recall(gen, r["answer"])
                        for gen, r in zip(gens, forget)]))
    curve.append([0, round(r0, 3)])
    for step in range(1, 61):
        model.train()
        fi = torch.randperm(len(forget), generator=g)[:8].tolist()
        ids, labels, mask = make_batch(tok, [forget[j] for j in fi])
        opt.zero_grad(set_to_none=True)
        t11.batch_ce(model, ids, labels, mask).backward()
        opt.step()
        if step % 5 == 0:
            model.eval()
            gens = t15.greedy_batch(model, tok, forget)
            rl = float(np.mean([t15.rouge_l_recall(gen, r["answer"])
                                for gen, r in zip(gens, forget)]))
            curve.append([step, round(rl, 3)])
            print(f"relearn {tag} step {step} R-L {rl:.3f}", flush=True)
            if hit is None and rl >= half_base:
                hit = step
                break
    log({"stage": "t17_relearn", "tag": tag, "model_dir": model_dir,
         "steps_to_half_base": hit, "curve": curve})


if __name__ == "__main__":
    {"train": stage_train, "relearn": stage_relearn}[sys.argv[1]]()
