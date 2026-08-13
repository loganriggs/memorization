"""T35: RWKU pilot — realistic unlearning of real pretrained knowledge.

RWKU (jinzhuoran/RWKU): 200 real famous-person targets; the model's knowledge
comes from PRETRAINING (no implant fine-tune). Per target:
  train_original_passage  wiki passages (the unlearn corpus)
  forget_level1           cloze probes        (knowledge memorization)
  forget_level2           QA probes           (knowledge manipulation)
  forget_level3           adversarial probes  (paraphrase/jailbreak attacks)
  neighbor_level1/2       adjacent-knowledge probes (collateral metric)

Pilot protocol (declared, not tuned):
  model    Llama-3.2-1B-Instruct (base instruct, NOT the TOFU fine-tune)
  targets  top-10 by base-model knowledge (level-2 QA ROUGE, measured first —
           forgetting is only meaningful for facts the model actually has)
  methods  ga    plain gradient ascent on passages, lr 1e-5
           npo   NPO on passages (ref-anchored, beta 0.1), lr 2e-5 (the TOFU-
                 tuned value)
           ours  all-token margin pin gamma=2 on passage tokens + KL anchor
                 on non-pilot targets' passages (declared extra data), lr 1e-5
  budget   2 epochs over the target's passages, batch 4, max_len 512, Adafactor
  eval     greedy, max_new 32, ROUGE-L recall vs gold (frozen decode protocol);
           forget score = mean over levels 1-3 (lower better), neighbor score
           = mean over levels 1-2 (higher better)

Stages:
  basecheck             rank all 200 targets by base 1B knowledge -> t35_targets.json
  train <method> <k>    unlearn pilot target k (0-9) -> checkpoint
  eval <ckpt> <k> <tag> probe suite -> results/t35_rwku.jsonl
  baseeval <k>          probe suite on the BASE model (the reference row)
"""
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

DEVICE = "cuda"
MODEL_ID = "meta-llama/Llama-3.2-1B-Instruct"
MODEL_FALLBACK = "unsloth/Llama-3.2-1B-Instruct"  # ungated mirror, same weights
OUT = "results/t35_rwku.jsonl"
TARGETS_F = "results/t35_targets.json"
MAXNEW = 32


def log(rec):
    with open(OUT, "a") as f:
        f.write(json.dumps(rec) + "\n")
    print(json.dumps(rec), flush=True)


def get_model_and_tok():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    for mid in (MODEL_ID, MODEL_FALLBACK):
        try:
            tok = AutoTokenizer.from_pretrained(mid)
            m = AutoModelForCausalLM.from_pretrained(
                mid, dtype=torch.bfloat16,
                attn_implementation="sdpa").to(DEVICE)
            print(f"loaded {mid}", flush=True)
            return m, tok, mid
        except Exception as e:
            print(f"cannot load {mid}: {e}", flush=True)
    raise SystemExit("no base model reachable")


def load_ckpt(path):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(path)
    m = AutoModelForCausalLM.from_pretrained(
        path, dtype=torch.bfloat16, attn_implementation="sdpa").to(DEVICE)
    return m, tok


def rwku(cfg):
    import datasets
    for sp in ("train", "test"):
        try:
            return list(datasets.load_dataset("jinzhuoran/RWKU", cfg, split=sp))
        except Exception:
            continue
    raise SystemExit(f"cannot load RWKU config {cfg}")


def probe_prompt(tok, q, ptype):
    if ptype == "cloze":
        user = ("Please complete the blank in the following sentence. "
                "Answer with the missing words only.\n" + q)
    else:
        user = "Please answer the following question briefly.\n" + q
    msgs = [{"role": "user", "content": user}]
    ids = tok.apply_chat_template(msgs, add_generation_prompt=True,
                                  tokenize=True)
    return list(ids["input_ids"] if hasattr(ids, "keys") else ids)


def rouge_l_recall(pred, ref):
    from rouge_score import rouge_scorer
    sc = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    return sc.score(ref, pred)["rougeL"].recall


@torch.no_grad()
def run_probes(model, tok, probes, bs=16):
    model.eval()
    scores = []
    for i in range(0, len(probes), bs):
        chunk = probes[i:i + bs]
        enc = [probe_prompt(tok, p["query"], p["type"]) for p in chunk]
        L = max(len(e) for e in enc)
        pad = tok.pad_token_id or tok.eos_token_id
        ids = torch.full((len(enc), L), pad)
        mask = torch.zeros((len(enc), L), dtype=torch.long)
        for j, e in enumerate(enc):          # left-pad for generation
            ids[j, L - len(e):] = torch.tensor(e)
            mask[j, L - len(e):] = 1
        out = model.generate(ids.to(DEVICE), attention_mask=mask.to(DEVICE),
                             max_new_tokens=MAXNEW, do_sample=False,
                             pad_token_id=pad)
        for j, p in enumerate(chunk):
            txt = tok.decode(out[j, L:], skip_special_tokens=True).strip()
            scores.append(rouge_l_recall(txt, p["answer"]))
    return float(np.mean(scores)), len(scores)


def passages_for(subject, chunks=None):
    rows = [r for r in rwku("train_original_passage") if r["subject"] == subject]
    return rows if chunks is None else rows[:chunks]


def passage_batch(tok, rows, max_len=512):
    enc = [tok(r["text"], add_special_tokens=True,
               truncation=True, max_length=max_len).input_ids for r in rows]
    L = max(len(e) for e in enc)
    pad = tok.pad_token_id or tok.eos_token_id
    ids = torch.full((len(enc), L), pad)
    labels = torch.full((len(enc), L), -100)
    mask = torch.zeros((len(enc), L), dtype=torch.long)
    for i, e in enumerate(enc):
        ids[i, :len(e)] = torch.tensor(e)
        labels[i, :len(e)] = torch.tensor(e)
        mask[i, :len(e)] = 1
    return ids.to(DEVICE), labels.to(DEVICE), mask.to(DEVICE)


def seq_logprob(model, ids, labels, mask):
    lg = model(input_ids=ids, attention_mask=mask).logits.float()
    lsm = F.log_softmax(lg[:, :-1], -1)
    lab = labels[:, 1:]
    m = (lab != -100).float()
    lp = lsm.gather(-1, lab.clamp(min=0).unsqueeze(-1)).squeeze(-1)
    return (lp * m).sum(-1), lg          # per-sequence sum, raw logits


def stage_basecheck():
    """Rank all 200 targets by base-model level-2 QA knowledge."""
    model, tok, mid = get_model_and_tok()
    l2 = rwku("forget_level2")
    bysub = {}
    for r in l2:
        bysub.setdefault(r["subject"], []).append(r)
    ranks = []
    for i, (sub, probes) in enumerate(sorted(bysub.items())):
        sc, n = run_probes(model, tok, probes[:10])
        ranks.append({"subject": sub, "base_l2_rouge": round(sc, 4), "n": n})
        if i % 20 == 0:
            print(f"basecheck {i}/200 {sub} {sc:.3f}", flush=True)
    ranks.sort(key=lambda r: -r["base_l2_rouge"])
    json.dump({"model": mid, "ranking": ranks, "pilot": ranks[:10]},
              open(TARGETS_F, "w"), indent=1)
    print("pilot targets:", [r["subject"] for r in ranks[:10]], flush=True)


def pilot_subjects():
    return [r["subject"] for r in json.load(open(TARGETS_F))["pilot"]]


def stage_train():
    method, k = sys.argv[2], int(sys.argv[3])
    assert method in ("ga", "npo", "ours")
    subject = pilot_subjects()[k]
    tag = f"t35_{method}_t{k}"
    outdir = f"results/{tag}"
    if os.path.exists(f"{outdir}/config.json"):
        print(f"{tag} exists, skip", flush=True)
        return
    model, tok, mid = get_model_and_tok()
    model.gradient_checkpointing_enable()
    rows = passages_for(subject)
    print(f"{tag} subject={subject} passages={len(rows)}", flush=True)

    ref = None
    anchor = None
    if method in ("npo", "ours"):
        from transformers import AutoModelForCausalLM
        ref = AutoModelForCausalLM.from_pretrained(
            mid, dtype=torch.bfloat16, attn_implementation="sdpa").to(DEVICE)
        ref.eval()
        for p in ref.parameters():
            p.requires_grad_(False)
    if method == "ours":
        # KL anchor corpus: passages of non-pilot targets (ranked 100-200)
        ranking = json.load(open(TARGETS_F))["ranking"]
        others = [r["subject"] for r in ranking[100:150]]
        allp = [r for r in rwku("train_original_passage")
                if r["subject"] in set(others)]
        rng = np.random.default_rng(0)
        anchor = [allp[i] for i in rng.choice(len(allp),
                                              min(200, len(allp)), False)]

    lr = 2e-5 if method == "npo" else 1e-5
    from transformers.optimization import Adafactor
    opt = Adafactor(model.parameters(), lr=lr, scale_parameter=False,
                    relative_step=False, warmup_init=False)
    g = torch.Generator().manual_seed(0)
    epochs, bs = 2, 4
    steps = max(1, (len(rows) * epochs) // bs)
    for step in range(steps):
        fi = torch.randperm(len(rows), generator=g)[:bs].tolist()
        ids, labels, mask = passage_batch(tok, [rows[j] for j in fi])
        opt.zero_grad(set_to_none=True)
        if method == "ga":
            lg = model(input_ids=ids, attention_mask=mask).logits.float()
            ce = F.cross_entropy(lg[:, :-1].reshape(-1, lg.shape[-1]),
                                 labels[:, 1:].reshape(-1), ignore_index=-100)
            loss = -ce
        elif method == "npo":
            lp, _ = seq_logprob(model, ids, labels, mask)
            with torch.no_grad():
                lp_ref, _ = seq_logprob(ref, ids, labels, mask)
            beta = 0.1
            loss = (-2.0 / beta) * F.logsigmoid(-beta * (lp - lp_ref)).mean()
        else:  # ours: all-token margin pin + KL anchor on other-entity text
            lg = model(input_ids=ids, attention_mask=mask).logits.float()
            lab = labels[:, 1:]
            m = lab != -100
            lgs = lg[:, :-1]
            own = lgs.gather(-1, lab.clamp(min=0).unsqueeze(-1)).squeeze(-1)
            oth = lgs.scatter(
                -1, lab.clamp(min=0).unsqueeze(-1), float("-inf")).max(-1).values
            pin = (F.relu((own - oth) + 2.0) * m).sum() / m.sum()
            ai = torch.randperm(len(anchor), generator=g)[:bs].tolist()
            aids, alab, am = passage_batch(tok, [anchor[j] for j in ai])
            lg_a = model(input_ids=aids, attention_mask=am).logits.float()
            with torch.no_grad():
                lg_ar = ref(input_ids=aids, attention_mask=am).logits.float()
            kl = F.kl_div(F.log_softmax(lg_a, -1), F.log_softmax(lg_ar, -1),
                          log_target=True, reduction="none").sum(-1)
            kl = (kl * am).sum() / am.sum()
            loss = pin + kl
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 10 == 0:
            print(f"{tag} step {step}/{steps} loss {float(loss):.4f}",
                  flush=True)
    model.save_pretrained(outdir)
    tok.save_pretrained(outdir)
    log({"stage": "train_done", "tag": tag, "method": method, "k": k,
         "subject": subject, "steps": steps, "lr": lr})


def eval_suite(model, tok, subject):
    out = {}
    for cfg, key in (("forget_level1", "forget_l1"),
                     ("forget_level2", "forget_l2"),
                     ("forget_level3", "forget_l3"),
                     ("neighbor_level1", "neighbor_l1"),
                     ("neighbor_level2", "neighbor_l2")):
        probes = [r for r in rwku(cfg) if r["subject"] == subject][:40]
        sc, n = run_probes(model, tok, probes)
        out[key] = round(sc, 4)
        out[f"{key}_n"] = n
    out["forget_mean"] = round(np.mean([out["forget_l1"], out["forget_l2"],
                                        out["forget_l3"]]), 4)
    out["neighbor_mean"] = round(np.mean([out["neighbor_l1"],
                                          out["neighbor_l2"]]), 4)
    return out


def stage_eval():
    ckpt, k, tag = sys.argv[2], int(sys.argv[3]), sys.argv[4]
    subject = pilot_subjects()[k]
    model, tok = load_ckpt(ckpt)
    rec = {"stage": "eval", "tag": tag, "k": k, "subject": subject,
           "ckpt": ckpt, "max_new": MAXNEW, "rouge_impl": "rouge_score"}
    rec.update(eval_suite(model, tok, subject))
    log(rec)


def stage_baseeval():
    k = int(sys.argv[2])
    subject = pilot_subjects()[k]
    model, tok, mid = get_model_and_tok()
    rec = {"stage": "eval", "tag": f"t35_base_t{k}", "k": k,
           "subject": subject, "ckpt": mid, "max_new": MAXNEW,
           "rouge_impl": "rouge_score"}
    rec.update(eval_suite(model, tok, subject))
    log(rec)


if __name__ == "__main__":
    {"basecheck": stage_basecheck, "train": stage_train,
     "eval": stage_eval, "baseeval": stage_baseeval}[sys.argv[1]]()
