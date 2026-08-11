"""T15: TOFU's official metrics — truth ratio, forget quality (KS test),
model utility — for our Pythia-410M checkpoints.

Definitions verified line-by-line against the official code
(github.com/locuslab/tofu, aggregate_eval_stat.py + evaluate_util.py):
  normalized prob  exp(mean answer-token logprob); on real_authors/
                   world_facts, multiple-choice normalized
                   P(true)/(P(true)+sum P(perturbed)).
  truth ratio      R = exp(ref_lp - mean perturbed_lp), ref =
                   paraphrased_answer (forget/retain) else original
                   answer; > 1 = prefers the true answer.
  forget quality   two-sample KS p-value between the unlearned model's
                   forget-set R distribution and the retain-only
                   reference model's. p > 0.05 = indistinguishable.
  model utility    harmonic mean of 9 numbers: {prob, ROUGE-L recall of
                   greedy gen, mean max(0, 1-1/R)} on each of
                   {retain_perturbed, real_authors, world_facts}.
                   Forget set reports mean min(R, 1/R) instead.

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
from transformers import AutoModelForCausalLM, AutoTokenizer

import t11_tofu as t11

DEVICE = "cuda"
OUT = "results/t15_metrics.jsonl"
TR_DIR = "results/t15_truthratios"

# Overrides for the rented-GPU phase: the metric math below is what P2 is
# testing, so it is untouched -- only the model/split it points at move.
# Defaults reproduce the original Pythia/forget01 invocations exactly.
TOK_ID = os.environ.get("T15_TOK_ID")            # default: t11.MODEL_ID (Pythia-410m)
FORGET_SPLIT = os.environ.get("T15_FORGET_SPLIT", "forget01_perturbed")


def get_tok():
    if TOK_ID:
        tok = AutoTokenizer.from_pretrained(TOK_ID)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
    else:
        tok = t11.get_tok()
    # locuslab/tofu_ft_phi-1.5 ships NO tokenizer files. AutoTokenizer falls
    # back to an empty GPT2Tokenizer (vocab_size 0) that returns [] for every
    # input *without raising*, which would silently zero out every metric.
    # Never let that reach the eval loop.
    probe = tok("Question: probe\nAnswer: probe", add_special_tokens=False).input_ids
    if len(probe) < 4:
        raise RuntimeError(
            f"tokenizer {TOK_ID or 'default'} encoded a probe string to "
            f"{len(probe)} ids (vocab_size={tok.vocab_size}) -- it has no usable "
            "vocab. Point T15_TOK_ID at the base model (e.g. microsoft/phi-1_5).")
    return tok


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


def greedy_batch(model, tok, rows, max_new=64, bs=8):
    """Batched greedy decoding WITHOUT model.generate / KV cache: the
    cached one-token decode path segfaults intermittently on this
    torch-2.13/cu130/sm_120 setup (crash in rotate_half; see research
    log's tiny-tensor hot-loop class). Full-sequence forwards only."""
    texts = []
    for i in range(0, len(rows), bs):
        chunk = rows[i:i + bs]
        enc = [tok(f"Question: {r['question']}\nAnswer:",
                   add_special_tokens=False).input_ids for r in chunk]
        lens = torch.tensor([len(e) for e in enc], device=DEVICE)
        B, L = len(chunk), int(lens.max()) + max_new
        ids = torch.full((B, L), tok.eos_token_id, dtype=torch.long,
                         device=DEVICE)
        mask = torch.zeros((B, L), dtype=torch.long, device=DEVICE)
        for b, e in enumerate(enc):
            ids[b, :len(e)] = torch.tensor(e, device=DEVICE)
            mask[b, :len(e)] = 1
        cur = lens.clone()
        ar = torch.arange(B, device=DEVICE)
        with torch.no_grad():
            for _ in range(max_new):
                lg = model(input_ids=ids, attention_mask=mask,
                           use_cache=False).logits
                nxt = lg[ar, cur - 1].argmax(-1)
                ids[ar, cur] = nxt
                mask[ar, cur] = 1
                cur = cur + 1
        for b in range(B):
            gen = ids[b, int(lens[b]):int(cur[b])].tolist()
            if tok.eos_token_id in gen:
                gen = gen[:gen.index(tok.eos_token_id)]
            texts.append(tok.decode(gen).split("\nQuestion")[0].strip())
    return texts


def eval_set(model, tok, rows, with_rouge=True):
    """Official formulas (tofu/aggregate_eval_stat.py):
    truth ratio R = exp(ref_lp - mean(pert_lp)), ref = paraphrased
    answer where present else original (>1 = prefers true answer);
    prob = normalized P(answer) on QA sets, multiple-choice-normalized
    P(true)/(P(true)+sum P(pert)) on real_authors/world_facts."""
    probs, rouges, ratios = [], [], []
    for row in rows:
        q = row["question"]
        ans_lp = norm_logprob(model, tok, q, row["answer"])
        pert_lps = [norm_logprob(model, tok, q, a)
                    for a in row["perturbed_answer"]]
        para = row.get("paraphrased_answer")
        ref_lp = norm_logprob(model, tok, q, para) if para else ans_lp
        ratios.append(float(np.exp(ref_lp - np.mean(pert_lps))))
        if para:
            probs.append(float(np.exp(ans_lp)))
        else:
            p_true = np.exp(ans_lp)
            probs.append(float(p_true / (p_true +
                                         sum(np.exp(l)
                                             for l in pert_lps))))
    if with_rouge:
        gens = greedy_batch(model, tok, rows)
        rouges = [rouge_l_recall(g, row["answer"])
                  for g, row in zip(gens, rows)]
    return probs, rouges, ratios


def stage_eval():
    import datasets
    model_dir, tag = sys.argv[2], sys.argv[3]
    os.makedirs(TR_DIR, exist_ok=True)
    tok = get_tok()
    model = AutoModelForCausalLM.from_pretrained(
        model_dir, torch_dtype=torch.float32).to(DEVICE)
    model.eval()
    sets = {name: list(datasets.load_dataset("locuslab/TOFU", cfg,
                                             split="train"))
            for name, cfg in [("forget", FORGET_SPLIT),
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
    # model utility: harmonic mean of 9; utility sets use
    # mean(max(0, 1 - 1/R)); forget's reported stat is mean(min(R, 1/R))
    nine = []
    for name in ("retain", "real_authors", "world_facts"):
        nine += [res[name]["prob"], res[name]["rouge"],
                 float(np.mean([max(0.0, 1.0 - 1.0 / r)
                                for r in ratios_store[name]]))]
    utility = (len(nine) / sum(1.0 / max(x, 1e-9) for x in nine))
    res["forget"]["truth_ratio_stat"] = float(np.mean(
        [min(r, 1.0 / r) for r in ratios_store["forget"]]))
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
    tok = get_tok()
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
