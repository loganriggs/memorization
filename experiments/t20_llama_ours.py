"""T20: our method (margin pin + retain hinge + KL anchor) on Llama-3.2-1B TOFU.

Port of t14_phi.py's `ours` trainer to the chat-template Llama checkpoints,
for the pre-registered matrix. Losses are t13/t14 verbatim; what changes is
batch construction (open-unlearning chat template via t15's split_ids) and the
model/reference (open-unlearning/tofu_Llama-3.2-1B-Instruct_full).

Usage:
  python t20_llama_ours.py train <scope> <gamma> <seed> <forget_split>
    scope        all | min          (all-token / min-token margin pin)
    gamma        margin depth, e.g. 2.0
    seed         0 | 1 | 2
    forget_split forget01 | forget05 | forget10

  -> checkpoint at results/t20_<split>_<scope>_g<gamma>_s<seed>/
  -> record appended to results/t20_llama.jsonl

Retain split pairs with the forget split (forget05 -> retain95), subsampled to
400 rows as in t14. Steps default to ~15 sample-epochs over the forget set at
batch 4 (t13 protocol); override with T20_STEPS. The pre-registration freezes
the step count per split before the sweep is scored.
"""
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

os.environ.setdefault("T15_TEMPLATE", "llama3")
import t15_tofu_metrics as t15  # noqa: E402  (template helpers; sets llama3)

DEVICE = "cuda"
MODEL_ID = "open-unlearning/tofu_Llama-3.2-1B-Instruct_full"
OUT = "results/t20_llama.jsonl"
PAIR = {"forget01": "retain99", "forget05": "retain95", "forget10": "retain90"}


def log(rec):
    with open(OUT, "a") as f:
        f.write(json.dumps(rec) + "\n")
    print(json.dumps(rec), flush=True)


def get_tok():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def load(path):
    from transformers import AutoModelForCausalLM
    m = AutoModelForCausalLM.from_pretrained(
        path, dtype=torch.bfloat16, attn_implementation="sdpa").to(DEVICE)
    return m


def make_batch(tok, rows, max_len=320):
    """Chat-template batch: ids/labels/mask, loss on the answer span only
    (closing <|eot_id|> included, matching open-unlearning's labels)."""
    enc = []
    for r in rows:
        pids, aids = t15.split_ids(tok, r["question"], r["answer"])
        ids = (pids + aids)[:max_len]
        lab = ([-100] * len(pids) + aids)[:max_len]
        enc.append((ids, lab))
    L = max(len(e[0]) for e in enc)
    ids = torch.full((len(enc), L), tok.pad_token_id)
    labels = torch.full((len(enc), L), -100)
    mask = torch.zeros((len(enc), L), dtype=torch.long)
    for i, (x, y) in enumerate(enc):
        ids[i, :len(x)] = torch.tensor(x)
        labels[i, :len(y)] = torch.tensor(y)
        mask[i, :len(x)] = 1
    return ids.to(DEVICE), labels.to(DEVICE), mask.to(DEVICE)


def batch_ce(model, ids, labels, mask):
    lg = model(input_ids=ids, attention_mask=mask).logits.float()
    return F.cross_entropy(lg[:, :-1].reshape(-1, lg.shape[-1]),
                           labels[:, 1:].reshape(-1), ignore_index=-100)


def seq_tok_margins(model, ids, labels, mask):
    """t14 verbatim: per-token logit margin own - best-other on answer tokens."""
    lg = model(input_ids=ids, attention_mask=mask).logits.float()
    all_m, min_m = [], []
    for b in range(ids.shape[0]):
        pos = (labels[b] != -100).nonzero().squeeze(1)
        lgb = lg[b, pos - 1]
        tg = labels[b, pos]
        own = lgb.gather(1, tg.unsqueeze(1)).squeeze(1)
        oth = lgb.scatter(1, tg.unsqueeze(1), float("-inf")).max(1).values
        m = own - oth
        all_m.append(m)
        min_m.append(m.min())
    return all_m, torch.stack(min_m), lg


def fact_margins(model, tok, rows, bs=8):
    margs, accs = [], []
    with torch.no_grad():
        for i in range(0, len(rows), bs):
            ids, labels, mask = make_batch(tok, rows[i:i + bs])
            am, mm, _ = seq_tok_margins(model, ids, labels, mask)
            for b, m in enumerate(am):
                margs.append(float(m.min()))
                accs.append(bool((m > 0).all()))
    return np.array(margs), np.array(accs)


def stage_train():
    import datasets
    scope, gamma, seed, split = (sys.argv[2], float(sys.argv[3]),
                                 int(sys.argv[4]), sys.argv[5])
    assert scope in ("all", "min") and split in PAIR
    tag = f"t20_{split}_{scope}_g{gamma:g}_s{seed}"
    outdir = f"results/{tag}"

    tok = get_tok()
    forget = list(datasets.load_dataset("locuslab/TOFU", split, split="train"))
    retain = list(datasets.load_dataset("locuslab/TOFU", PAIR[split],
                                        split="train"))[:400]
    steps = int(os.environ.get(
        "T20_STEPS", str(max(150, round(15 * len(forget) / 4 / 50) * 50))))

    torch.manual_seed(seed)
    model = load(MODEL_ID)
    model.gradient_checkpointing_enable()
    ref = load(MODEL_ID)
    ref.eval()
    for p in ref.parameters():
        p.requires_grad_(False)

    # retain restoration targets: reference min-margins, capped at the median
    m0_r, _ = fact_margins(ref, tok, retain)
    cap = float(np.median(m0_r))
    base_m0 = torch.tensor(np.minimum(m0_r, cap), dtype=torch.float32,
                           device=DEVICE)

    from transformers.optimization import Adafactor
    opt = Adafactor(model.parameters(), lr=1e-5, scale_parameter=False,
                    relative_step=False, warmup_init=False)
    g = torch.Generator().manual_seed(seed)

    for step in range(steps):
        fi = torch.randperm(len(forget), generator=g)[:4].tolist()
        ri = torch.randperm(len(retain), generator=g)[:4].tolist()
        fids, flab, fm = make_batch(tok, [forget[j] for j in fi])
        rids, rlab, rm = make_batch(tok, [retain[j] for j in ri])
        opt.zero_grad(set_to_none=True)

        fam, fmin, _ = seq_tok_margins(model, fids, flab, fm)
        if scope == "all":
            pin = torch.stack([F.relu(m + gamma).mean() for m in fam]).mean()
        else:
            pin = F.relu(fmin + gamma).mean()

        _, mr, lg_r = seq_tok_margins(model, rids, rlab, rm)
        with torch.no_grad():
            lg_rb = ref(input_ids=rids, attention_mask=rm).logits.float()
        kl = F.kl_div(F.log_softmax(lg_r, -1), F.log_softmax(lg_rb, -1),
                      log_target=True, reduction="none").sum(-1)
        kl = (kl * rm).sum() / rm.sum()
        tr = base_m0[torch.tensor(ri, device=DEVICE)]
        loss = pin + F.relu(tr - mr).mean() + kl

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if step % 50 == 0:
            fmg, facc = fact_margins(model, tok, forget[:20])
            print(f"{tag} step {step}/{steps} loss {float(loss):.4f} "
                  f"forget_acc {facc.mean():.2f} min_margin {fmg.mean():.2f}",
                  flush=True)
        if step % 200 == 199:
            model.save_pretrained(outdir)
            tok.save_pretrained(outdir)

    model.save_pretrained(outdir)
    tok.save_pretrained(outdir)
    log({"stage": "train_done", "tag": tag, "scope": scope, "gamma": gamma,
         "seed": seed, "split": split, "steps": steps, "out": outdir})


if __name__ == "__main__":
    {"train": stage_train}[sys.argv[1]]()
