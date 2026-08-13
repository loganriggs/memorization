"""T33: ours v3 — fixing the retain objective (post-campaign, Logan-approved).

t32 showed the retain collateral is GLOBAL and objective-limited: the margin
hinge restores rank margins and the KL preserves distribution shape, but
neither pushes absolute gold-answer probability mass back (retain/prob 0.33
vs NPO's 0.62, whose retain term is plain CE). Two candidate fixes, both on
the selected config (min-token, γ=4, 450 steps, retain cap 400 — one change
at a time vs v1):

  ce     loss = pin + margin_hinge + KL + CE(retain answers)
         (NPO's retain term bolted onto ours unchanged)
  lppin  loss = pin + logprob_hinge + KL
         (margin hinge REPLACED by per-token absolute log-prob restoration:
          relu(lp_ref - lp_model) on retain answer tokens — pushes prob mass
          back up to reference level, but unlike CE has no incentive to
          overshoot past the reference model)

Usage: python t33_llama_v3.py train <variant> <seed> [<forget_split>]
Tags:  t33_<split>_min_g4_s<seed>_<variant>
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

DEVICE = "cuda"
OUT = "results/t33_llama.jsonl"
GAMMA = 4.0
STEPS = int(os.environ.get("T33_STEPS", "450"))
RETAIN_CAP = 400


def token_logprobs(logits, labels):
    """Per-token log-prob of the label tokens; returns (lp, mask)."""
    lsm = F.log_softmax(logits[:, :-1], -1)
    lab = labels[:, 1:]
    m = lab != -100
    lp = lsm.gather(-1, lab.clamp(min=0).unsqueeze(-1)).squeeze(-1)
    return lp, m


def stage_train():
    import datasets
    variant, seed = sys.argv[2], int(sys.argv[3])
    split = sys.argv[4] if len(sys.argv) > 4 else "forget05"
    assert variant in ("ce", "lppin")
    tag = f"t33_{split}_min_g{GAMMA:g}_s{seed}_{variant}"
    outdir = f"results/{tag}"

    tok = t20.get_tok()
    forget = list(datasets.load_dataset("locuslab/TOFU", split, split="train"))
    retain = list(datasets.load_dataset("locuslab/TOFU", t20.PAIR[split],
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
    opt = Adafactor(model.parameters(), lr=1e-5, scale_parameter=False,
                    relative_step=False, warmup_init=False)
    g = torch.Generator().manual_seed(seed)

    for step in range(STEPS):
        fi = torch.randperm(len(forget), generator=g)[:4].tolist()
        ri = torch.randperm(len(retain), generator=g)[:4].tolist()
        fids, flab, fm = t20.make_batch(tok, [forget[j] for j in fi])
        rids, rlab, rm = t20.make_batch(tok, [retain[j] for j in ri])
        opt.zero_grad(set_to_none=True)

        fam, fmin, _ = t20.seq_tok_margins(model, fids, flab, fm)
        pin = F.relu(fmin + GAMMA).mean()   # min-token scope, γ=4

        _, mr, lg_r = t20.seq_tok_margins(model, rids, rlab, rm)
        with torch.no_grad():
            lg_rb = ref(input_ids=rids, attention_mask=rm).logits.float()
        kl = F.kl_div(F.log_softmax(lg_r, -1), F.log_softmax(lg_rb, -1),
                      log_target=True, reduction="none").sum(-1)
        kl = (kl * rm).sum() / rm.sum()

        if variant == "ce":
            tr = base_m0[torch.tensor(ri, device=DEVICE)]
            ce_r = F.cross_entropy(
                lg_r[:, :-1].reshape(-1, lg_r.shape[-1]),
                rlab[:, 1:].reshape(-1), ignore_index=-100)
            loss = pin + F.relu(tr - mr).mean() + kl + ce_r
        else:  # lppin: absolute log-prob restoration replaces the margin hinge
            lp, mmask = token_logprobs(lg_r, rlab)
            with torch.no_grad():
                lp_ref, _ = token_logprobs(lg_rb, rlab)
            lp_hinge = (F.relu(lp_ref - lp) * mmask).sum() / mmask.sum()
            loss = pin + lp_hinge + kl

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if step % 50 == 0:
            fmg, facc = t20.fact_margins(model, tok, forget[:20])
            print(f"{tag} step {step}/{STEPS} loss {float(loss):.4f} "
                  f"forget_acc {facc.mean():.2f} min_margin {fmg.mean():.2f}",
                  flush=True)

    model.save_pretrained(outdir)
    tok.save_pretrained(outdir)
    with open(OUT, "a") as f:
        f.write(json.dumps({"stage": "train_done", "tag": tag,
                            "variant": variant, "seed": seed, "split": split,
                            "steps": STEPS, "out": outdir}) + "\n")
    print(json.dumps({"stage": "train_done", "tag": tag}), flush=True)


if __name__ == "__main__":
    {"train": stage_train}[sys.argv[1]]()
