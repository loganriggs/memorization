"""T15: TOFU's official metrics — truth ratio, forget quality (KS test),
model utility — for our Pythia-410M checkpoints.

Definitions (TOFU, arXiv 2401.06121):
  normalized prob  P(a|q)^(1/|a|) = exp(mean answer-token logprob)
  truth ratio      R = mean_norm-prob(perturbed answers) / norm-prob(ref
                   answer), ref = paraphrased_answer where it exists
                   (forget/retain), else the original answer (real
                   authors / world facts).
  forget quality   two-sample KS p-value between the unlearned model's
                   forget-set R distribution and the retain-only
                   reference model's. p > 0.05 = indistinguishable.
  model utility    harmonic mean of 9 numbers: {norm prob, ROUGE-L
                   recall of greedy gen, max(0, 1-R)} on each of
                   {retain_perturbed, real_authors, world_facts}.

Stages (python t15_tofu_metrics.py <stage> ...):
  train_retain          — train the retain-only reference model
                          (Pythia-410m on retain99, t11 protocol) ->
                          results/t15_retain_ref
  eval <model_dir> <tag> — all metrics for one checkpoint; per-example
                          truth ratios saved to
                          results/t15_truthratios/<tag>.json, summary
                          appended to results/t15_metrics.jsonl
  ks <tag> <ref_tag>    — forget quality: KS test of stored R
                          distributions.
"""

import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM

import t11_tofu as t11

DEVICE = "cuda"
OUT = "results/t15_metrics.jsonl"
TR_DIR = "results/t15_truthratios"
REF_DIR = "results/t15_retain_ref"


def log(rec):
    with open(OUT, "a") as f:
        f.write(json.dumps(rec) + "\n")
    print(json.dumps(rec), flush=True)


def norm_logprob(model, tok, question, answer):
    """Mean answer-token logprob (log of TOFU's normalized prob)."""
    ids, labels, mask = t11.make_batch(tok, [{"question": question,
                                              "answer": answer}])
    with torch.no_grad():
        lg = model(input_ids=ids, attention_mask=mask).logits
    lp = -F.cross_entropy(lg[0, :-1], labels[0, 1:], ignore_index=-100,
                          reduction="none")
    n = (labels[0, 1:] != -100).sum()
    return float(lp.sum() / n)


def rouge_l_recall(gen, ref):
    g, r = gen.lower().split(), ref.lower().split()
    if not g or not r:
        return 0.0
    dp = np.zeros((len(g) + 1, len(r) + 1), dtype=int)
    for i in range(len(g)):
        for j in range(len(r)):
            dp[i + 1, j + 1] = (dp[i, j] + 1 if g[i] == r[j]
                                else max(dp[i, j + 1], dp[i + 1, j]))
    return dp[len(g), len(r)] / len(r)


def greedy_rouge(model, tok, row, max_new=64):
    prompt = f"Question: {row['question']}\nAnswer:"
    ids = tok(prompt, return_tensors="pt").input_ids.to(DEVICE)
    with torch.no_grad():
        out = model.generate(ids, max_new_tokens=max_new, do_sample=False,
                             pad_token_id=tok.eos_token_id)
    gen = tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True)
    return rouge_l_recall(gen, row["answer"])


def truth_ratio(model, tok, row):
    ref_ans = row.get("paraphrased_answer") or row["answer"]
    ref_lp = norm_logprob(model, tok, row["question"], ref_ans)
    pert = [np.exp(norm_logprob(model, tok, row["question"], a))
            for a in row["perturbed_answer"]]
    return float(np.mean(pert) / np.exp(ref_lp))


def eval_set(model, tok, rows, with_rouge=True):
    probs, rouges, ratios = [], [], []
    for row in rows:
        probs.append(np.exp(norm_logprob(model, tok, row["question"],
                                         row["answer"])))
        ratios.append(truth_ratio(model, tok, row))
        if with_rouge:
            rouges.append(greedy_rouge(model, tok, row))
    return probs, rouges, ratios


def stage_eval():
    import datasets
    model_dir, tag = sys.argv[2], sys.argv[3]
    os.makedirs(TR_DIR, exist_ok=True)
    tok = t11.get_tok()
    model = AutoModelForCausalLM.from_pretrained(
        model_dir, torch_dtype=torch.float32).to(DEVICE)
    model.eval()
    sets = {name: list(datasets.load_dataset("locuslab/TOFU", cfg,
                                             split="train"))
            for name, cfg in [("forget", "forget01_perturbed"),
                              ("retain", "retain_perturbed"),
                              ("real_authors", "real_authors_perturbed"),
                              ("world_facts", "world_facts_perturbed")]}
    res, ratios_store = {}, {}
    for name, rows in sets.items():
        probs, rouges, ratios = eval_set(model, tok, rows)
        ratios_store[name] = ratios
        res[name] = {"prob": float(np.mean(probs)),
                     "rouge": float(np.mean(rouges)),
                     "truth_ratio_med": float(np.median(ratios))}
        print(f"{tag} {name}: {res[name]}", flush=True)
    # model utility: harmonic mean of 9 (utility sets only, R -> 1-R)
    nine = []
    for name in ("retain", "real_authors", "world_facts"):
        probs = res[name]["prob"]
        nine += [probs, res[name]["rouge"],
                 float(np.mean([max(0.0, 1.0 - r)
                                for r in ratios_store[name]]))]
    utility = (len(nine) / sum(1.0 / max(x, 1e-9) for x in nine))
    with open(f"{TR_DIR}/{tag}.json", "w") as f:
        json.dump(ratios_store, f)
    log({"stage": "eval", "tag": tag, "model_dir": model_dir,
         "model_utility": round(utility, 4),
         **{f"{k}_{m}": round(v, 4) for k, d in res.items()
            for m, v in d.items()}})


def stage_ks():
    from scipy.stats import ks_2samp
    tag, ref_tag = sys.argv[2], sys.argv[3]
    a = json.load(open(f"{TR_DIR}/{tag}.json"))["forget"]
    b = json.load(open(f"{TR_DIR}/{ref_tag}.json"))["forget"]
    stat, p = ks_2samp(a, b)
    log({"stage": "forget_quality", "tag": tag, "ref": ref_tag,
         "ks_stat": round(float(stat), 4), "p_value": float(p),
         "indistinguishable_at_0.05": bool(p > 0.05)})


def stage_train_retain():
    """Retain-only reference: t11's training protocol on retain99."""
    import datasets
    tok = t11.get_tok()
    ds = list(datasets.load_dataset("locuslab/TOFU", "retain99",
                                    split="train"))
    model = AutoModelForCausalLM.from_pretrained(
        t11.MODEL_ID, torch_dtype=torch.float32).to(DEVICE)
    model.gradient_checkpointing_enable()
    opt = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
    g = torch.Generator().manual_seed(0)
    bs, epochs = 8, 5
    for ep in range(epochs):
        order = torch.randperm(len(ds), generator=g).tolist()
        for i in range(0, len(ds), bs):
            rows = [ds[j] for j in order[i:i + bs]]
            ids, labels, mask = t11.make_batch(tok, rows)
            opt.zero_grad(set_to_none=True)
            loss = t11.batch_ce(model, ids, labels, mask)
            loss.backward()
            opt.step()
            if (i // bs) % 100 == 0:
                print(f"retain_ref ep{ep} step{i//bs} "
                      f"loss {float(loss):.3f}", flush=True)
    model.save_pretrained(REF_DIR)
    log({"stage": "train_retain_done", "epochs": epochs})


if __name__ == "__main__":
    {"eval": stage_eval, "ks": stage_ks,
     "train_retain": stage_train_retain}[sys.argv[1]]()
