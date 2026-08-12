"""T25: relearn-resistance curves on Llama TOFU (t18 protocol, ported).

Finetune a checkpoint on the forget set with plain CE at a given lr; measure
forget-set prob and generation leakage at log-spaced step counts. The
retain-reference run is the never-knew CONTROL — a method only demonstrates
relearn resistance by recovering *slower than the control*, not merely slowly
(LOCAL t18: relearning is lr-fragile, so both lrs are required).

Usage:
  python t25_relearn.py run <ckpt_dir_or_hf_id> <tag> <lr> [<forget_split>]

Appends per-eval-point records to results/t25_relearn.jsonl and a copy to
../reports/remote/t25_relearn.jsonl (tracked). Optimizer: AdamW (the point of
t18/t19 was that AdamW's preconditioner defeats first-order flatness tricks,
so AdamW is the honest adversary). Labels: answer text tokens only (no eot),
matching the unlearning-side protocol. Eval: forget set only (prob + leakage
under the frozen decode protocol), cheap enough for many points.
"""
import copy
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

os.environ.setdefault("T15_TEMPLATE", "llama3")
os.environ.setdefault("T15_ROUGE", "rouge_score")
import t15_tofu_metrics as t15  # noqa: E402
import t20_llama_ours as t20  # noqa: E402  (make_batch, batch_ce, load)

DEVICE = "cuda"
EVAL_AT = [0, 5, 10, 20, 40, 80, 160]
OUT = "results/t25_relearn.jsonl"
MIRROR = "../reports/remote/t25_relearn.jsonl"


def log(rec):
    for p in (OUT, MIRROR):
        with open(p, "a") as f:
            f.write(json.dumps(rec) + "\n")
    print(json.dumps(rec), flush=True)


def eval_forget(model, tok, rows_pert):
    """Forget-set prob + gen leakage under the frozen decode protocol."""
    model.eval()
    probs = []
    with torch.no_grad():
        for r in rows_pert:
            probs.append(float(np.exp(t15.norm_logprob(
                model, tok, r["question"], r["answer"]))))
    gens = t15.greedy_batch(model, tok, rows_pert)
    rouges = [t15.rouge_l_recall(g, r["answer"])
              for g, r in zip(gens, rows_pert)]
    model.train()
    return float(np.mean(probs)), float(np.mean(rouges))


def main():
    import datasets
    _, ckpt, tag, lr = sys.argv[1:5]
    split = sys.argv[5] if len(sys.argv) > 5 else "forget05"
    lr = float(lr)

    tok = t20.get_tok()
    forget = list(datasets.load_dataset("locuslab/TOFU", split, split="train"))
    pert = list(datasets.load_dataset("locuslab/TOFU", f"{split}_perturbed",
                                      split="train"))

    model = t20.load(ckpt)
    model.gradient_checkpointing_enable()
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    g = torch.Generator().manual_seed(0)

    step = 0
    for target in EVAL_AT:
        while step < target:
            fi = torch.randperm(len(forget), generator=g)[:4].tolist()
            ids, lab, mask = t20.make_batch(tok, [forget[j] for j in fi])
            opt.zero_grad(set_to_none=True)
            loss = t20.batch_ce(model, ids, lab, mask)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            step += 1
        prob, rouge = eval_forget(model, tok, pert)
        log({"tag": tag, "ckpt": ckpt, "lr": lr, "split": split,
             "relearn_step": step, "forget_prob": round(prob, 4),
             "forget_rouge": round(rouge, 4),
             "template": "llama3", "rouge_impl": t15.ROUGE_IMPL,
             "max_new": t15.MAX_NEW})


if __name__ == "__main__":
    main()
