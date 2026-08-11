"""T21: seed replication for the relearn/jog claims (red-team #7).

Faithfully replicates each original training protocol at new seeds:
  pin_g8 — t13 run_config(all, gamma=8): 200 steps, AdamW 1e-5,
           bs 8+8, clip 1.0, pin + restoration hinge + KL.
  npo    — t11 stage_unlearn npo: 125 steps, AdamW 1e-5, bs 4+4,
           clip 1.0, NPO(beta 0.1) + retain-CE.

Usage: python t21_seeds.py train <pin_g8|npo> <seed>
   ->  results/t21_<method>_s<seed>/
Relearn/jog/para runs go through t20_methods.py relearn with its new
relearn-seed argument.
"""

import json
import sys

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM

import t11_tofu as t11
import t13_sweep as t13
import t17_methods as t17
from t11_tofu import BASE_DIR, DEVICE, fact_margins, get_tok, make_batch

OUT = "results/t21_seeds.jsonl"
GAMMA, NPO_BETA, LAMB = 8.0, 0.1, 1.0


def stage_train():
    import datasets
    method, seed = sys.argv[2], int(sys.argv[3])
    tok = get_tok()
    forget = list(datasets.load_dataset("locuslab/TOFU", "forget01",
                                        split="train"))
    retain = list(datasets.load_dataset("locuslab/TOFU", "retain99",
                                        split="train"))[:400]
    model = AutoModelForCausalLM.from_pretrained(
        BASE_DIR, torch_dtype=torch.float32).to(DEVICE)
    base = AutoModelForCausalLM.from_pretrained(
        BASE_DIR, torch_dtype=torch.float32).to(DEVICE)
    base.eval()
    m0_r, _, _ = fact_margins(base, tok, retain)
    cap = float(np.median(m0_r))
    target_r = torch.tensor(np.minimum(m0_r, cap), dtype=torch.float32,
                            device=DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-5)
    g = torch.Generator().manual_seed(seed)
    steps, bs = (200, 8) if method == "pin_g8" else (125, 4)

    for step in range(steps):
        fi = torch.randperm(len(forget), generator=g)[:bs].tolist()
        ri = torch.randperm(len(retain), generator=g)[:bs].tolist()
        fids, flab, fm = make_batch(tok, [forget[j] for j in fi])
        rids, rlab, rm = make_batch(tok, [retain[j] for j in ri])
        opt.zero_grad(set_to_none=True)
        if method == "pin_g8":
            fam, _ = t13.tok_margins(model, fids, flab, fm)
            floss = torch.stack([F.relu(m + GAMMA).mean()
                                 for m in fam]).mean()
            ram, lg_r = t13.tok_margins(model, rids, rlab, rm)
            mr = torch.stack([m.min() for m in ram])
            with torch.no_grad():
                lg_rb = base(input_ids=rids, attention_mask=rm).logits
            kl = F.kl_div(F.log_softmax(lg_r, -1),
                          F.log_softmax(lg_rb, -1), log_target=True,
                          reduction="none").sum(-1)
            kl = (kl * rm).sum() / rm.sum()
            tr = target_r[torch.tensor(ri, device=DEVICE)]
            loss = floss + F.relu(tr - mr).mean() + LAMB * kl
        else:  # npo, t11 protocol
            lp, _ = t17.seq_logps(model, fids, flab, fm)
            with torch.no_grad():
                lpr, _ = t17.seq_logps(base, fids, flab, fm)
            loss = ((-2 / NPO_BETA) * F.logsigmoid(
                -NPO_BETA * (lp - lpr)).mean()
                + t11.batch_ce(model, rids, rlab, rm))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 50 == 0:
            _, acc, _ = fact_margins(model, tok, forget[:20])
            print(f"{method} s{seed} step {step} loss {float(loss):.3f} "
                  f"forget_acc {acc.mean():.2f}", flush=True)
    model.save_pretrained(f"results/t21_{method}_s{seed}")
    with open(OUT, "a") as f:
        f.write(json.dumps({"stage": "train_done", "method": method,
                            "seed": seed}) + "\n")


if __name__ == "__main__":
    {"train": stage_train}[sys.argv[1]]()
