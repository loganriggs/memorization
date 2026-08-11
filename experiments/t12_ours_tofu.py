"""T12: our method on the Pythia/TOFU setup — head-to-head with GA/NPO.

Margin-pinned retension, sequence version: from the TOFU base model,
optimize
  relu(min_answer_margin_f + GAMMA)          forget pin (bounded, not
                                             unbounded ascent)
  + relu(min(m0_r, cap) - m_r)               retain margin restoration
  + LAMB * KL(retain seqs || base model)     distribution anchor
Then the same diagnosis battery as t11 (lens, relearn, retain damage).
Saves results/t11_tofu_ours/; appends to results/t11_tofu.jsonl."""

import json

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM

import t11_tofu as t11
from t11_tofu import (BASE_DIR, DEVICE, fact_margins, get_tok, lens_rank,
                      log, make_batch)

GAMMA, LAMB, STEPS, LR = 2.0, 1.0, 200, 1e-5


def seq_min_margins(model, ids, labels, mask):
    """Differentiable per-fact min answer-token margin."""
    lg = model(input_ids=ids, attention_mask=mask).logits
    out = []
    for b in range(ids.shape[0]):
        pos = (labels[b] != -100).nonzero().squeeze(1)
        lgb = lg[b, pos - 1]
        tg = labels[b, pos]
        own = lgb.gather(1, tg.unsqueeze(1)).squeeze(1)
        oth = lgb.scatter(1, tg.unsqueeze(1), float("-inf")).max(1).values
        out.append((own - oth).min())
    return torch.stack(out), lg


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

    model = AutoModelForCausalLM.from_pretrained(
        BASE_DIR, torch_dtype=torch.float32).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    g = torch.Generator().manual_seed(0)
    for step in range(STEPS):
        fi = torch.randperm(len(forget), generator=g)[:8].tolist()
        ri = torch.randperm(len(retain), generator=g)[:8].tolist()
        fids, flab, fm = make_batch(tok, [forget[j] for j in fi])
        rids, rlab, rm = make_batch(tok, [retain[j] for j in ri])
        opt.zero_grad(set_to_none=True)
        mf, _ = seq_min_margins(model, fids, flab, fm)
        mr, lg_r = seq_min_margins(model, rids, rlab, rm)
        with torch.no_grad():
            lg_rb = base(input_ids=rids, attention_mask=rm).logits
        kl = F.kl_div(F.log_softmax(lg_r, -1),
                      F.log_softmax(lg_rb, -1),
                      log_target=True, reduction="none").sum(-1)
        kl = (kl * rm).sum() / rm.sum()
        tr = target_r[torch.tensor(ri, device=DEVICE)]
        loss = (F.relu(mf + GAMMA).mean() + F.relu(tr - mr).mean()
                + LAMB * kl)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 25 == 0:
            _, acc, _ = fact_margins(model, tok, forget[:20])
            print(f"step {step} loss {float(loss):.3f} forget_acc "
                  f"{acc.mean():.2f}", flush=True)
    model.save_pretrained("results/t11_tofu_ours")

    # diagnosis (same battery as t11 stage_diagnose)
    bm, bacc, bpos = fact_margins(base, tok, forget)
    known = np.where(bacc)[0][:10]
    m, acc, _ = fact_margins(model, tok, forget)
    lens_prof = [lens_rank(model, tok, forget[int(fi)], bpos[int(fi)][0],
                           bpos[int(fi)][1]) for fi in known[:5]]
    med_lens = np.median(np.array(lens_prof), axis=0).astype(int).tolist()

    m2 = AutoModelForCausalLM.from_pretrained(
        "results/t11_tofu_ours", torch_dtype=torch.float32).to(DEVICE)
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
    del m2
    torch.cuda.empty_cache()
    retain_m, retain_acc, _ = fact_margins(model, tok,
                                           retain[:100])
    log({"stage": "diagnose", "model": "ours_retension",
         "forget_acc": round(float(acc[known].mean()), 2),
         "forget_margin_med": round(float(np.median(m[known])), 2),
         "lens_med_rank_per_layer": med_lens,
         "relearn_steps_to_half": relearn,
         "retain_acc": round(float(retain_acc.mean()), 3),
         "retain_margin_med": round(float(np.median(retain_m)), 2)})


if __name__ == "__main__":
    main()
