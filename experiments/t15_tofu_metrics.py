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

# Generation prompt convention (Logan's call, 2026-08-11: report both, ours
# headline, open-unlearning's as an appendix column).
#   ""  -> "Question: {q}\nAnswer:"   our convention, correct for BPE models
#   " " -> "Question: {q}\nAnswer: "  open-unlearning's asst_start_tag
# Their trailing space tokenizes to a standalone ' ' token, which is off
# distribution for a model trained on "Answer:" + " The" and costs ~0.09
# ROUGE-L recall on Phi-1.5. See reports/remote/LOG.md.
PROMPT_SUFFIX = os.environ.get("T15_PROMPT_SUFFIX", "")

# Decode protocol. ROUGE-L recall is protocol-sensitive at the ~15% relative
# level with the model held fixed: recall can only rise with more generated
# tokens, and truncating the continuation removes text that could still match.
# Defaults are our original protocol (64 tokens, truncate at the next
# "Question"); open-unlearning uses 200 tokens and no truncation.
MAX_NEW = int(os.environ.get("T15_MAX_NEW", "64"))
TRUNCATE_AT_QUESTION = os.environ.get("T15_TRUNCATE", "1") == "1"

# ROUGE implementation. "lcs" is our word-level LCS recall (what every Pythia/Phi
# pilot used); "rouge_score" is the standard package with a Porter stemmer, used
# by official TOFU and open-unlearning. Measured 0.043 apart on the unlearned
# Phi checkpoint -- so a floor measured under one is not valid under the other.
ROUGE_IMPL = os.environ.get("T15_ROUGE", "lcs")

# Prompt template. "qa" is TOFU's raw Question/Answer format (Pythia, Phi-1.5).
# "llama3" is open-unlearning's chat template for the Llama TOFU checkpoints,
# copied from configs/model/Llama-3.2-1B-Instruct.yaml. Note its asst_start_tag
# ends in "\n\n", not a bare space, so the Phi trailing-space defect cannot
# occur on this path.
TEMPLATE = os.environ.get("T15_TEMPLATE", "qa")

# open-unlearning's Llama template args (configs/model/Llama-3.2-1B-Instruct.yaml).
# Rendering goes through tokenizer.apply_chat_template -- NOT hand-rolled tags --
# because the tokenizer's Jinja template inserts a date header
# ("Cutting Knowledge Date: ...\nToday Date: 10 Apr 2025") that hand-rolled
# strings miss; that omission alone moved forget prob by ~5%.
LLAMA3_SYSTEM = "You are a helpful assistant."
LLAMA3_DATE = "10 Apr 2025"


def _chat_ids(out):
    """apply_chat_template(tokenize=True) returns a list in transformers 4.x
    and a BatchEncoding in 5.x; normalize to a flat id list."""
    if hasattr(out, "keys"):
        out = out["input_ids"]
    if out and isinstance(out[0], list):
        out = out[0]
    return list(out)


def build_prompt(question, tok=None):
    """Prompt text, up to and including the assistant-turn opener."""
    if TEMPLATE == "llama3":
        chat = [{"role": "system", "content": LLAMA3_SYSTEM},
                {"role": "user", "content": question}]
        return tok.apply_chat_template(chat, tokenize=False,
                                       add_generation_prompt=True,
                                       date_string=LLAMA3_DATE)
    return f"Question: {question}\nAnswer:" + PROMPT_SUFFIX


def split_ids(tok, question, answer):
    """(prompt_ids, answer_ids) under the active template.

    The QA path tokenizes the two halves separately (' ' + answer keeps the
    leading space bound to the first word, which is correct for BPE). The chat
    path mirrors open-unlearning's preprocess_chat_instance exactly: full
    conversation ids vs generation-prompt ids, answer span = the difference --
    which includes the closing <|eot_id|>, as theirs does. Matching their span
    is required for the FQ self-test against their published logs.
    """
    if TEMPLATE == "llama3":
        chat = [{"role": "system", "content": LLAMA3_SYSTEM},
                {"role": "user", "content": question}]
        full = _chat_ids(tok.apply_chat_template(
            chat + [{"role": "assistant", "content": answer}], tokenize=True,
            add_generation_prompt=False, date_string=LLAMA3_DATE))
        pids = _chat_ids(tok.apply_chat_template(
            chat, tokenize=True, add_generation_prompt=True,
            date_string=LLAMA3_DATE))
        return pids, full[len(pids):]
    prompt = build_prompt(question)
    pids = tok(prompt, add_special_tokens=False).input_ids
    aids = tok(" " + answer, add_special_tokens=False).input_ids
    return pids, aids


def stop_ids(tok):
    """Token ids that end a generation.

    Llama-3 Instruct ends an assistant turn with <|eot_id|>, which is NOT always
    what `tokenizer.eos_token_id` reports. Trimming on eos alone would leave the
    post-turn continuation in the scored text -- precisely the behaviour we
    criticised in open-unlearning's evaluator. Collect every applicable stop.
    """
    ids = {tok.eos_token_id}
    for tokstr in ("<|eot_id|>", "<|end_of_text|>"):
        i = tok.convert_tokens_to_ids(tokstr)
        if i is not None and i != tok.unk_token_id:
            ids.add(i)
    return {i for i in ids if i is not None}


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
    if TEMPLATE == "qa" and not PROMPT_SUFFIX:
        # Original path, byte-for-byte, so existing Pythia numbers reproduce.
        ids, labels, mask = t11.make_batch(tok, [{"question": question,
                                                  "answer": answer}])
    else:
        pids, aids = split_ids(tok, question, answer)
        seq = (pids + aids)[:192]
        lab = ([-100] * len(pids) + aids)[:192]
        ids = torch.tensor([seq], device=DEVICE)
        labels = torch.tensor([lab], device=DEVICE)
        mask = torch.ones_like(ids)
    with torch.no_grad():
        lg = model(input_ids=ids, attention_mask=mask).logits
    lp = -F.cross_entropy(lg[0, :-1], labels[0, 1:], ignore_index=-100,
                          reduction="none")
    n = (labels[0, 1:] != -100).sum()
    return float(lp.sum() / n)


def rouge_l_recall(gen, ref):
    """ROUGE-L recall under the configured implementation."""
    if ROUGE_IMPL == "rouge_score":
        from rouge_score import rouge_scorer
        global _SCORER
        try:
            _SCORER
        except NameError:
            _SCORER = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
        return _SCORER.score(ref, gen)["rougeL"].recall
    return _lcs_recall(gen, ref)


def _lcs_recall(gen, ref):
    g, r = gen.lower().split(), ref.lower().split()
    if not g or not r:
        return 0.0
    dp = np.zeros((len(g) + 1, len(r) + 1), dtype=int)
    for i in range(len(g)):
        for j in range(len(r)):
            dp[i + 1, j + 1] = (dp[i, j] + 1 if g[i] == r[j]
                                else max(dp[i, j + 1], dp[i + 1, j]))
    return dp[len(g), len(r)] / len(r)


def greedy_batch(model, tok, rows, max_new=None, bs=8):
    """Batched greedy decoding WITHOUT model.generate / KV cache: the
    cached one-token decode path segfaults intermittently on this
    torch-2.13/cu130/sm_120 setup (crash in rotate_half; see research
    log's tiny-tensor hot-loop class). Full-sequence forwards only.

    (Rented-5090 note: that segfault was later traced to the local box's bad
    CPU core, and generate() is fine here -- this path is kept for exact
    comparability with the existing local numbers, not for safety.)"""
    max_new = MAX_NEW if max_new is None else max_new
    texts = []
    for i in range(0, len(rows), bs):
        chunk = rows[i:i + bs]
        enc = [tok(build_prompt(r["question"], tok),
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
        stops = stop_ids(tok)
        for b in range(B):
            gen = ids[b, int(lens[b]):int(cur[b])].tolist()
            cut = min((gen.index(s) for s in stops if s in gen), default=len(gen))
            gen = gen[:cut]
            txt = tok.decode(gen, skip_special_tokens=True)
            if TRUNCATE_AT_QUESTION and TEMPLATE == "qa":
                txt = txt.split("\nQuestion")[0]
            texts.append(txt.strip())
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
         # which prompt convention produced these generations -- ROUGE and
         # anything derived from it are not comparable across conventions
         # Protocol stamp -- ROUGE and model utility are comparable ONLY across
         # records sharing all four of these.
         "prompt_convention": "ou_trailing_space" if PROMPT_SUFFIX == " " else "ours",
         "template": TEMPLATE, "rouge_impl": ROUGE_IMPL,
         "max_new": MAX_NEW, "truncate_at_question": TRUNCATE_AT_QUESTION,
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
