"""P2 sub-diff: attribute the forget-ROUGE gap (ours 0.9194 vs theirs 0.8424).

Two things differ at once, so measuring them together explains nothing:

  GENERATION  ours: cache-free greedy, max_new=64, right-pad, truncate at
                    "\\nQuestion"
              theirs: model.generate, max_new_tokens=200, use_cache=True,
                    left-pad (configs/generation/default.yaml)
  SCORING     ours: hand-rolled word-level LCS recall on .lower().split()
              theirs: rouge_score.RougeScorer(rougeL, use_stemmer=True).recall,
                    which strips non-alphanumerics and Porter-stems

This runs the 2x2: both generators scored by both scorers on the same rows.
The row differences isolate scoring; the column differences isolate generation.
"""
import os
import sys

import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

MODEL = os.environ.get("P2_MODEL", "locuslab/tofu_ft_phi-1.5")
TOK = os.environ.get("P2_TOK", "microsoft/phi-1_5")
SPLIT = os.environ.get("P2_SPLIT", "forget01_perturbed")
N = int(os.environ.get("P2_N", "40"))
DEVICE = "cuda"


def ours_rouge(gen, ref):
    """t15_tofu_metrics.rouge_l_recall, verbatim."""
    g, r = gen.lower().split(), ref.lower().split()
    if not g or not r:
        return 0.0
    dp = np.zeros((len(g) + 1, len(r) + 1), dtype=int)
    for i in range(len(g)):
        for j in range(len(r)):
            dp[i + 1, j + 1] = (dp[i, j] + 1 if g[i] == r[j]
                                else max(dp[i, j + 1], dp[i + 1, j]))
    return dp[len(g), len(r)] / len(r)


def theirs_rouge(gen, ref):
    from rouge_score import rouge_scorer
    sc = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    return sc.score(ref, gen)["rougeL"].recall


def gen_ours(model, tok, rows, max_new=64, bs=8):
    """t15_tofu_metrics.greedy_batch, verbatim behaviour."""
    texts = []
    for i in range(0, len(rows), bs):
        chunk = rows[i:i + bs]
        enc = [tok(f"Question: {r['question']}\nAnswer:",
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
            g = ids[b, int(lens[b]):int(cur[b])].tolist()
            if tok.eos_token_id in g:
                g = g[:g.index(tok.eos_token_id)]
            texts.append(tok.decode(g).split("\nQuestion")[0].strip())
    return texts


def gen_theirs(model, tok, rows, bs=8):
    """open-unlearning: left-pad + generate, max_new_tokens=200, use_cache."""
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    texts = []
    for i in range(0, len(rows), bs):
        chunk = rows[i:i + bs]
        prompts = [f"Question: {r['question']}\nAnswer: " for r in chunk]
        enc = tok(prompts, return_tensors="pt", padding=True,
                  add_special_tokens=True).to(DEVICE)
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=200, do_sample=False,
                                 use_cache=True, pad_token_id=tok.eos_token_id)
        for b in range(len(chunk)):
            new = out[b, enc["input_ids"].shape[1]:]
            texts.append(tok.decode(new, skip_special_tokens=True).strip())
    return texts


def main():
    tok = AutoTokenizer.from_pretrained(TOK)
    rows = list(load_dataset("locuslab/TOFU", SPLIT, split="train"))[:N]
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).to(DEVICE)
    model.eval()
    refs = [r["answer"] for r in rows]

    g_ours, g_theirs = gen_ours(model, tok, rows), gen_theirs(model, tok, rows)

    print(f"model={MODEL} split={SPLIT} n={len(rows)}\n")
    print(f"{'':22} {'ours scorer':>12} {'theirs scorer':>14}")
    for label, gens in (("ours generation", g_ours), ("theirs generation", g_theirs)):
        a = np.mean([ours_rouge(g, r) for g, r in zip(gens, refs)])
        b = np.mean([theirs_rouge(g, r) for g, r in zip(gens, refs)])
        print(f"{label:22} {a:12.4f} {b:14.4f}")

    lo = np.mean([len(g.split()) for g in g_ours])
    lt = np.mean([len(g.split()) for g in g_theirs])
    lr = np.mean([len(r.split()) for r in refs])
    print(f"\nmean words -- ours gen {lo:.1f}, theirs gen {lt:.1f}, reference {lr:.1f}")
    print("\n--- first 2 examples ---")
    for i in range(min(2, len(rows))):
        print(f"\n[{i}] REF    : {refs[i][:150]}")
        print(f"[{i}] OURS   : {g_ours[i][:150]}")
        print(f"[{i}] THEIRS : {g_theirs[i][:150]}")


if __name__ == "__main__":
    main()
