"""T32: what retain-set damage concretely looks like (Logan Q, 2026-08-13).

Compares the selected checkpoint (min-token γ4 seed 0, re-fetched from HF)
against the full model on retain95 rows, split into:
  - anchored:   rows [0:400]   (the hinge/KL saw these during training)
  - unanchored: rows [400:800] (never in any training batch)

Reports the TOFU 'prob' metric (P(gold answer)^(1/len)) per bucket for both
models, and prints side-by-side generations for a few rows. Inference only.
"""
import json
import os

import numpy as np
import torch

os.environ.setdefault("T15_TEMPLATE", "llama3")
import t15_tofu_metrics as t15  # noqa: E402

DEVICE = "cuda"
FULL = "open-unlearning/tofu_Llama-3.2-1B-Instruct_full"
OURS = ("/workspace/.hf_home/hub/models--Elriggs--memorization-unlearning/"
        "snapshots/9e97b1f8d26d7ba254e0f3131b1c67990d894ac3/"
        "llama3.2-1b/forget05/ours_min_g4/seed0")
N_PROB = 100     # rows per bucket for the prob metric
N_GEN = 5        # rows per bucket to print generations for


def load(path):
    from transformers import AutoModelForCausalLM
    return AutoModelForCausalLM.from_pretrained(
        path, dtype=torch.float32, attn_implementation="sdpa").to(DEVICE).eval()


@torch.no_grad()
def answer_prob(model, tok, row):
    """TOFU prob metric: P(gold answer | question)^(1/n_tokens)."""
    pids, aids = t15.split_ids(tok, row["question"], row["answer"])
    ids = torch.tensor([pids + aids], device=DEVICE)
    lg = model(input_ids=ids).logits.float().log_softmax(-1)
    lp = lg[0, len(pids) - 1:-1].gather(1, ids[0, len(pids):].unsqueeze(1))
    return float(lp.mean().exp())


@torch.no_grad()
def generate(model, tok, row, max_new=64):
    pids, _ = t15.split_ids(tok, row["question"], row["answer"])
    out = model.generate(torch.tensor([pids], device=DEVICE),
                         max_new_tokens=max_new, do_sample=False,
                         pad_token_id=tok.pad_token_id)
    txt = tok.decode(out[0, len(pids):], skip_special_tokens=True)
    return txt.strip()


def main():
    import datasets
    tok = t15.get_tok()
    retain = list(datasets.load_dataset("locuslab/TOFU", "retain95",
                                        split="train"))
    rng = np.random.default_rng(0)
    buckets = {
        "anchored[0:400]": [retain[i] for i in rng.choice(400, N_PROB, False)],
        "unanchored[400:800]": [retain[i] for i in
                                rng.choice(np.arange(400, 800), N_PROB, False)],
    }
    results = {}
    for name, path in (("full", FULL), ("ours_selected", OURS)):
        model = load(path)
        for bname, rows in buckets.items():
            ps = [answer_prob(model, tok, r) for r in rows]
            results[(name, bname)] = float(np.mean(ps))
            print(f"{name:14s} {bname:22s} mean prob = {np.mean(ps):.4f}",
                  flush=True)
        # generations on fixed rows (same for both models)
        for bname, idxs in (("anchored", [3, 17, 42, 111, 256]),
                            ("unanchored", [403, 471, 555, 610, 777])):
            print(f"\n--- {name} generations, {bname} rows ---", flush=True)
            for i in idxs[:N_GEN]:
                r = retain[i]
                g = generate(model, tok, r)
                print(f"[{i}] Q: {r['question']}\n    gold: {r['answer'][:110]}"
                      f"\n    {name}: {g[:110]}\n", flush=True)
        del model
        torch.cuda.empty_cache()
    json.dump({f"{m}|{b}": v for (m, b), v in results.items()},
              open("../reports/remote/t32_retain_probs.json", "w"), indent=1)
    print("wrote t32_retain_probs.json")


if __name__ == "__main__":
    main()
