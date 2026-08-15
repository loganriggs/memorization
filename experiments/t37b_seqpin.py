"""T37b: sequential composition (Logan's refinement) — take the saved tuned
NPO checkpoint (lr 2e-5, the frontier winner) and apply our margin pin ON TOP,
with the v3 retain side (logprob pin toward the FULL model + KL) holding
utility. Two depths:

  min g4, 200 steps   conservative (per-sequence ceiling limits depth)
  all g4, 100 steps   max content-removal depth

Init = t23_forget05_npo_lr2e-05_s<seed>; reference/anchor = frozen FULL model.

Usage: python t37b_seqpin.py train <scope> <steps> <seed>
Tag:   t37s_forget05_<scope>_g4_s<seed>
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
RETAIN_CAP = 400


def stage_train():
    import datasets
    scope, steps, seed = sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
    assert scope in ("min", "all")
    # T37B_INIT/T37B_TAG env overrides let t41 reuse this trainer with a
    # different starting checkpoint (pin-on-AltPO) without touching t37s dirs
    split = os.environ.get("T37B_SPLIT", "forget05")
    init = os.environ.get("T37B_INIT",
                          f"results/t23_{split}_npo_lr2e-05_s{seed}")
    tag = os.environ.get("T37B_TAG", f"t37s_{split}_{scope}_g4_s{seed}")
    outdir = f"results/{tag}"

    tok = t20.get_tok()
    forget = list(datasets.load_dataset("locuslab/TOFU", split, split="train"))
    retain = list(datasets.load_dataset("locuslab/TOFU", t20.PAIR[split],
                                        split="train"))[:RETAIN_CAP]

    torch.manual_seed(seed)
    model = t20.load(init)                     # start FROM tuned NPO
    model.gradient_checkpointing_enable()
    ref = t20.load(t20.MODEL_ID)               # anchor to the FULL model
    ref.eval()
    for p in ref.parameters():
        p.requires_grad_(False)

    from transformers.optimization import Adafactor
    opt = Adafactor(model.parameters(), lr=1e-5, scale_parameter=False,
                    relative_step=False, warmup_init=False)
    g = torch.Generator().manual_seed(seed)

    for step in range(steps):
        fi = torch.randperm(len(forget), generator=g)[:4].tolist()
        ri = torch.randperm(len(retain), generator=g)[:4].tolist()
        fids, flab, fm = t20.make_batch(tok, [forget[j] for j in fi])
        rids, rlab, rm = t20.make_batch(tok, [retain[j] for j in ri])
        opt.zero_grad(set_to_none=True)

        fam, fmin, _ = t20.seq_tok_margins(model, fids, flab, fm)
        if scope == "all":
            pin = torch.stack([F.relu(m + GAMMA).mean() for m in fam]).mean()
        else:
            pin = F.relu(fmin + GAMMA).mean()

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

        loss = pin + lp_hinge + kl
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 50 == 0:
            print(f"{tag} step {step}/{steps} loss {float(loss):.4f}",
                  flush=True)

    model.save_pretrained(outdir)
    tok.save_pretrained(outdir)
    with open(OUT, "a") as f:
        f.write(json.dumps({"stage": "train_done", "tag": tag, "init": init,
                            "scope": scope, "steps": steps,
                            "seed": seed}) + "\n")
    print(json.dumps({"stage": "train_done", "tag": tag}), flush=True)


if __name__ == "__main__":
    {"train": stage_train}[sys.argv[1]]()
