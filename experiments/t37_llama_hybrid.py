"""T37: (a) the v1<->v3 retain-objective Pareto dial, (b) the NPO hybrid.

  mix    retain loss = (1-lam)*margin_hinge + lam*logprob_pin, + KL always.
         lam=0 is v1 (already measured), lam=1 is v3-lppin (measured);
         lam in {0.25, 0.5, 0.75} traces the curve between them.
         Forget side unchanged: min-token margin pin, gamma=4.
  npolp  forget side REPLACED by NPO's reference-anchored loss
         (-2/beta * logsigmoid(-beta*(lp - lp_ref)), beta=0.1, sequence-sum);
         retain side = ours (logprob_pin + KL). The best-FQ forget mechanism
         with the best-utility retain mechanism.

Usage:
  python t37_llama_hybrid.py train mix <lam> <seed>
  python t37_llama_hybrid.py train npolp <lr> <seed>
Tags: t37_forget05_mix<lam>_s<seed> / t37_forget05_npolp_lr<lr>_s<seed>
"""
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

os.environ.setdefault("T15_TEMPLATE", "llama3")
import t15_tofu_metrics as t15  # noqa: E402
import t20_llama_ours as t20  # noqa: E402
from t33_llama_v3 import token_logprobs  # noqa: E402

DEVICE = "cuda"
OUT = "results/t37_llama.jsonl"
GAMMA = 4.0
STEPS = int(os.environ.get("T37_STEPS", "450"))
RETAIN_CAP = 400
BETA = 0.1


def stage_train():
    import datasets
    kind = sys.argv[2]
    assert kind in ("mix", "npolp")
    split = os.environ.get("T37_SPLIT", "forget05")
    pair = t20.PAIR[split]
    if kind == "mix":
        lam, seed = float(sys.argv[3]), int(sys.argv[4])
        lr = 1e-5
        tag = f"t37_{split}_mix{lam:g}_s{seed}"
    else:
        lr, seed = float(sys.argv[3]), int(sys.argv[4])
        tag = f"t37_{split}_npolp_lr{lr:g}_s{seed}"
    outdir = f"results/{tag}"

    tok = t20.get_tok()
    forget = list(datasets.load_dataset("locuslab/TOFU", split, split="train"))
    retain = list(datasets.load_dataset("locuslab/TOFU", pair,
                                        split="train"))[:RETAIN_CAP]

    torch.manual_seed(seed)
    model = t20.load(t20.MODEL_ID)
    model.gradient_checkpointing_enable()
    ref = t20.load(t20.MODEL_ID)
    ref.eval()
    for p in ref.parameters():
        p.requires_grad_(False)

    m0_r, _ = t20.fact_margins(ref, tok, retain)
    cap = float(np.median(m0_r))
    base_m0 = torch.tensor(np.minimum(m0_r, cap), dtype=torch.float32,
                           device=DEVICE)

    from transformers.optimization import Adafactor
    opt = Adafactor(model.parameters(), lr=lr, scale_parameter=False,
                    relative_step=False, warmup_init=False)
    g = torch.Generator().manual_seed(seed)

    for step in range(STEPS):
        fi = torch.randperm(len(forget), generator=g)[:4].tolist()
        ri = torch.randperm(len(retain), generator=g)[:4].tolist()
        fids, flab, fm = t20.make_batch(tok, [forget[j] for j in fi])
        rids, rlab, rm = t20.make_batch(tok, [retain[j] for j in ri])
        opt.zero_grad(set_to_none=True)

        # ---- forget side ----
        if kind == "mix":
            fam, fmin, _ = t20.seq_tok_margins(model, fids, flab, fm)
            forget_loss = F.relu(fmin + GAMMA).mean()
        else:  # npolp: NPO sequence-level reference-anchored loss
            lg_f = model(input_ids=fids, attention_mask=fm).logits.float()
            lp_f, mf = token_logprobs(lg_f, flab)
            lp_seq = (lp_f * mf).sum(-1)
            with torch.no_grad():
                lg_fr = ref(input_ids=fids, attention_mask=fm).logits.float()
                lp_fr, _ = token_logprobs(lg_fr, flab)
                lp_seq_r = (lp_fr * mf).sum(-1)
            forget_loss = (-2.0 / BETA) * F.logsigmoid(
                -BETA * (lp_seq - lp_seq_r)).mean()

        # ---- retain side: logprob pin (+ margin hinge if mix) + KL ----
        _, mr, lg_r = t20.seq_tok_margins(model, rids, rlab, rm)
        with torch.no_grad():
            lg_rb = ref(input_ids=rids, attention_mask=rm).logits.float()
        kl = F.kl_div(F.log_softmax(lg_r, -1), F.log_softmax(lg_rb, -1),
                      log_target=True, reduction="none").sum(-1)
        kl = (kl * rm).sum() / rm.sum()
        lp, mmask = token_logprobs(lg_r, rlab)
        with torch.no_grad():
            lp_ref, _ = token_logprobs(lg_rb, rlab)
        lp_hinge = (F.relu(lp_ref - lp) * mmask).sum() / mmask.sum()
        if kind == "mix":
            tr = base_m0[torch.tensor(ri, device=DEVICE)]
            margin_hinge = F.relu(tr - mr).mean()
            retain_loss = (1 - lam) * margin_hinge + lam * lp_hinge + kl
        else:
            retain_loss = lp_hinge + kl

        loss = forget_loss + retain_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if step % 100 == 0:
            print(f"{tag} step {step}/{STEPS} loss {float(loss):.4f}",
                  flush=True)

    model.save_pretrained(outdir)
    tok.save_pretrained(outdir)
    with open(OUT, "a") as f:
        f.write(json.dumps({"stage": "train_done", "tag": tag, "kind": kind,
                            "seed": seed, "lr": lr, "steps": STEPS}) + "\n")
    print(json.dumps({"stage": "train_done", "tag": tag}), flush=True)


if __name__ == "__main__":
    {"train": stage_train}[sys.argv[1]]()
