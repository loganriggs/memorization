"""P2 follow-up: our cache-free greedy decoder vs model.generate(), everything else equal.

With prompt, decode length and truncation all matched to open-unlearning, our
t15 decoder still scores ~0.075 ROUGE below generate() on the unlearned
checkpoint. Both are supposed to be plain greedy decoding, so they should emit
identical token sequences. They do not, and this finds where they diverge.

Reports: first divergent step per example, how many examples diverge at all,
and the ROUGE each path achieves. Prints a diverging example verbatim.
"""
import os

import numpy as np
import torch
from datasets import load_dataset
from rouge_score import rouge_scorer
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = os.environ.get("P2_MODEL", "/workspace/memorization/results/p2_phi_ga")
TOK = os.environ.get("P2_TOK", "microsoft/phi-1_5")
N = int(os.environ.get("P2_N", "16"))
MAXNEW = int(os.environ.get("P2_MAXNEW", "64"))
SUFFIX = os.environ.get("P2_SUFFIX", "")
DEVICE = "cuda"


def ours_cachefree(model, tok, rows, max_new, bs=8):
    """t15_tofu_metrics.greedy_batch, no truncation, returns token id lists."""
    out = []
    for i in range(0, len(rows), bs):
        chunk = rows[i:i + bs]
        enc = [tok(f"Question: {r['question']}\nAnswer:" + SUFFIX,
                   add_special_tokens=False).input_ids for r in chunk]
        lens = torch.tensor([len(e) for e in enc], device=DEVICE)
        B, L = len(chunk), int(lens.max()) + max_new
        ids = torch.full((B, L), tok.eos_token_id, dtype=torch.long, device=DEVICE)
        mask = torch.zeros((B, L), dtype=torch.long, device=DEVICE)
        for b, e in enumerate(enc):
            ids[b, :len(e)] = torch.tensor(e, device=DEVICE)
            mask[b, :len(e)] = 1
        cur, ar = lens.clone(), torch.arange(B, device=DEVICE)
        with torch.no_grad():
            for _ in range(max_new):
                lg = model(input_ids=ids, attention_mask=mask, use_cache=False).logits
                ids[ar, cur] = lg[ar, cur - 1].argmax(-1)
                mask[ar, cur] = 1
                cur = cur + 1
        for b in range(B):
            out.append(ids[b, int(lens[b]):int(cur[b])].tolist())
    return out


def theirs_generate(model, tok, rows, max_new, bs=8):
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    out = []
    for i in range(0, len(rows), bs):
        chunk = rows[i:i + bs]
        pr = [f"Question: {r['question']}\nAnswer:" + SUFFIX for r in chunk]
        enc = tok(pr, return_tensors="pt", padding=True,
                  add_special_tokens=False).to(DEVICE)
        with torch.no_grad():
            o = model.generate(**enc, max_new_tokens=max_new, do_sample=False,
                               use_cache=True, pad_token_id=tok.eos_token_id)
        for b in range(len(chunk)):
            out.append(o[b, enc["input_ids"].shape[1]:].tolist())
    return out


def trim_eos(ids, eos):
    return ids[:ids.index(eos)] if eos in ids else ids


def main():
    tok = AutoTokenizer.from_pretrained(TOK)
    tok.pad_token = tok.eos_token
    rows = list(load_dataset("locuslab/TOFU", "forget01_perturbed", split="train"))[:N]
    refs = [r["answer"] for r in rows]
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).to(DEVICE)
    model.eval()

    a = ours_cachefree(model, tok, rows, MAXNEW)
    b = theirs_generate(model, tok, rows, MAXNEW)

    sc = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    firsts, n_div = [], 0
    for x, y in zip(a, b):
        d = next((k for k in range(min(len(x), len(y))) if x[k] != y[k]), None)
        if d is not None:
            n_div += 1
            firsts.append(d)

    ta = [tok.decode(trim_eos(x, tok.eos_token_id)).strip() for x in a]
    tb = [tok.decode(trim_eos(y, tok.eos_token_id)).strip() for y in b]
    ra = np.mean([sc.score(r, g)["rougeL"].recall for r, g in zip(refs, ta)])
    rb = np.mean([sc.score(r, g)["rougeL"].recall for r, g in zip(refs, tb)])

    print(f"model={MODEL}\nn={N} max_new={MAXNEW} suffix={SUFFIX!r}")
    print(f"examples with any divergence: {n_div}/{len(a)}")
    if firsts:
        print(f"first divergent step: min={min(firsts)} median={int(np.median(firsts))} max={max(firsts)}")
    print(f"ROUGE-L recall  cache-free {ra:.4f}   generate {rb:.4f}")
    for i, (x, y) in enumerate(zip(a, b)):
        d = next((k for k in range(min(len(x), len(y))) if x[k] != y[k]), None)
        if d is not None:
            print(f"\n--- example {i}, diverges at step {d} ---")
            print(f"prefix      : {tok.decode(x[:d])!r}")
            print(f"cache-free  : {tok.decode(x[d:d+12])!r}")
            print(f"generate    : {tok.decode(y[d:d+12])!r}")
            break


if __name__ == "__main__":
    main()
