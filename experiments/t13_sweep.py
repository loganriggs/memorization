"""T13: gamma / pin-scope frontier sweep on Pythia-410M + TOFU.

Configs: scope in {min, all} x gamma in {0.5, 2, 8}. Loss per t12:
  pin(scope, gamma) + relu(min(m0_r, cap) - m_r) + KL(retain || base)
200 steps AdamW lr 1e-5, bs 8. Diagnosis battery per config: forget
margins/acc, lens profile, relearn speed, retain acc, plus ROUGE-L
recall on greedy generations (forget + retain) — T14 showed margins
alone mislead. Saves results/t13_{scope}_g{gamma}/; appends to
results/t13_sweep.jsonl."""

import json

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM

import t11_tofu as t11
from t11_tofu import (BASE_DIR, DEVICE, fact_margins, get_tok, lens_rank,
                      make_batch)

OUT = "results/t13_sweep.jsonl"
STEPS, LR, LAMB = 200, 1e-5, 1.0


def log(rec):
    with open(OUT, "a") as f:
        f.write(json.dumps(rec) + "\n")
    print(json.dumps(rec), flush=True)


def tok_margins(model, ids, labels, mask):
    """Per-fact answer-token margins (list of tensors) + retain logits."""
    lg = model(input_ids=ids, attention_mask=mask).logits
    out = []
    for b in range(ids.shape[0]):
        pos = (labels[b] != -100).nonzero().squeeze(1)
        lgb = lg[b, pos - 1]
        tg = labels[b, pos]
        own = lgb.gather(1, tg.unsqueeze(1)).squeeze(1)
        oth = lgb.scatter(1, tg.unsqueeze(1), float("-inf")).max(1).values
        out.append(own - oth)
    return out, lg


def lcs_len(a, b):
    m, n = len(a), len(b)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m):
        for j in range(n):
            dp[i + 1][j + 1] = (dp[i][j] + 1 if a[i] == b[j]
                                else max(dp[i][j + 1], dp[i + 1][j]))
    return dp[m][n]


def gen_rouge(model, tok, rows, bs=8):
    scores = []
    for i in range(0, len(rows), bs):
        batch = rows[i:i + bs]
        prompts = [f"Question: {r['question']}\nAnswer:" for r in batch]
        enc = tok(prompts, return_tensors="pt", padding=True,
                  padding_side="left")
        ids = enc.input_ids.to(DEVICE)
        mask = enc.attention_mask.to(DEVICE)
        with torch.no_grad():
            out = model.generate(ids, attention_mask=mask,
                                 max_new_tokens=60, do_sample=False,
                                 pad_token_id=tok.eos_token_id)
        for b, r in enumerate(batch):
            gen = tok.decode(out[b, ids.shape[1]:],
                             skip_special_tokens=True)
            g, h = r["answer"].lower().split(), gen.lower().split()
            scores.append(lcs_len(g, h) / len(g) if g else 0.0)
    return float(np.mean(scores))


def run_config(scope, gamma, tok, forget, retain, base, target_r, bpos,
               known):
    import os
    name = f"{scope}_g{gamma}"
    ckpt = f"results/t13_{name}"
    if os.path.isdir(ckpt):
        print(f"{name}: checkpoint exists, skipping train", flush=True)
        model = AutoModelForCausalLM.from_pretrained(
            ckpt, torch_dtype=torch.float32).to(DEVICE)
        opt = None
        g = torch.Generator().manual_seed(0)
        steps = 0
    else:
        model = AutoModelForCausalLM.from_pretrained(
            BASE_DIR, torch_dtype=torch.float32).to(DEVICE)
        opt = torch.optim.AdamW(model.parameters(), lr=LR)
        g = torch.Generator().manual_seed(0)
        steps = STEPS
    for step in range(steps):
        fi = torch.randperm(len(forget), generator=g)[:8].tolist()
        ri = torch.randperm(len(retain), generator=g)[:8].tolist()
        fids, flab, fm = make_batch(tok, [forget[j] for j in fi])
        rids, rlab, rm = make_batch(tok, [retain[j] for j in ri])
        opt.zero_grad(set_to_none=True)
        fam, _ = tok_margins(model, fids, flab, fm)
        if scope == "min":
            pin = torch.stack([F.relu(m.min() + gamma)
                               for m in fam]).mean()
        else:
            pin = torch.stack([F.relu(m + gamma).mean()
                               for m in fam]).mean()
        ram, lg_r = tok_margins(model, rids, rlab, rm)
        mr = torch.stack([m.min() for m in ram])
        with torch.no_grad():
            lg_rb = base(input_ids=rids, attention_mask=rm).logits
        kl = F.kl_div(F.log_softmax(lg_r, -1), F.log_softmax(lg_rb, -1),
                      log_target=True, reduction="none").sum(-1)
        kl = (kl * rm).sum() / rm.sum()
        tr = target_r[torch.tensor(ri, device=DEVICE)]
        loss = pin + F.relu(tr - mr).mean() + LAMB * kl
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 50 == 0:
            _, acc, _ = fact_margins(model, tok, forget[:20])
            print(f"{name} step {step} loss {float(loss):.3f} "
                  f"forget_acc {acc.mean():.2f}", flush=True)
    if steps:
        model.save_pretrained(ckpt)
    if opt is not None:
        del opt
    model.zero_grad(set_to_none=True)
    torch.cuda.empty_cache()

    # diagnosis
    m, acc, _ = fact_margins(model, tok, forget)
    lens_prof = [lens_rank(model, tok, forget[int(fi)], bpos[int(fi)][0],
                           bpos[int(fi)][1]) for fi in known[:5]]
    med_lens = np.median(np.array(lens_prof), axis=0).astype(int).tolist()
    retain_m, retain_acc, _ = fact_margins(model, tok, retain[:100])
    f_rouge = gen_rouge(model, tok, forget)
    r_rouge = gen_rouge(model, tok, retain[:50])
    del model
    torch.cuda.empty_cache()

    m2 = AutoModelForCausalLM.from_pretrained(
        ckpt, torch_dtype=torch.float32).to(DEVICE)
    opt2 = torch.optim.AdamW(m2.parameters(), lr=1e-5)
    gg = torch.Generator().manual_seed(0)
    relearn = None
    for step in range(1, 41):
        fi = torch.randperm(len(forget), generator=gg)[:8].tolist()
        ids, labels, mask = make_batch(tok, [forget[j] for j in fi])
        opt2.zero_grad(set_to_none=True)
        t11.batch_ce(m2, ids, labels, mask).backward()
        opt2.step()
        if step % 5 == 0:
            _, ra, _ = fact_margins(m2, tok, [forget[int(i)]
                                              for i in known])
            if float(ra.mean()) >= 0.5:
                relearn = step
                break
    del m2, opt2
    torch.cuda.empty_cache()
    log({"scope": scope, "gamma": gamma,
         "forget_acc": round(float(acc[known].mean()), 2),
         "forget_margin_med": round(float(np.median(m[known])), 2),
         "forget_rougeL": round(f_rouge, 3),
         "retain_rougeL": round(r_rouge, 3),
         "retain_acc": round(float(retain_acc.mean()), 3),
         "retain_margin_med": round(float(np.median(retain_m)), 2),
         "lens_med_rank_per_layer": med_lens,
         "relearn_steps_to_half": relearn})


def main():
    import datasets
    tok = get_tok()
    forget = list(datasets.load_dataset("locuslab/TOFU", "forget01",
                                        split="train"))
    retain = list(datasets.load_dataset("locuslab/TOFU", "retain99",
                                        split="train"))[:400]
    base = AutoModelForCausalLM.from_pretrained(
        BASE_DIR, torch_dtype=torch.float32).to(DEVICE)
    base.eval()
    m0_r, _, _ = fact_margins(base, tok, retain)
    cap = float(np.median(m0_r))
    target_r = torch.tensor(np.minimum(m0_r, cap), dtype=torch.float32,
                            device=DEVICE)
    bm, bacc, bpos = fact_margins(base, tok, forget)
    known = np.where(bacc)[0][:10]
    log({"stage": "base", "forget_acc": round(float(bacc.mean()), 3),
         "forget_rougeL": round(gen_rouge(base, tok, forget), 3),
         "retain_rougeL": round(gen_rouge(base, tok, retain[:50]), 3)})
    for scope in ("min", "all"):
        for gamma in (0.5, 2.0, 8.0):
            run_config(scope, gamma, tok, forget, retain, base, target_r,
                       bpos, known)


if __name__ == "__main__":
    main()
