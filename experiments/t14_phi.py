"""T14: apples-to-apples — TOFU on microsoft/phi-1_5 (one of TOFU's two
official models), on the 16GB card via bf16 + 8-bit AdamW + gradient
checkpointing. Stages: train / unlearn (ga, npo, ours_alltok) / diagnose
(margins, lens, relearn, retain, paraphrase). Mirrors t11/t12.
Appends to results/t14_phi.jsonl."""

import json
import sys

import bitsandbytes as bnb
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM

import t11_tofu as t11
from t11_tofu import make_batch

DEVICE = "cuda"
OUT = "results/t14_phi.jsonl"
MODEL_ID = "microsoft/phi-1_5"
BASE_DIR = "results/t14_phi_base"


def log(rec):
    with open(OUT, "a") as f:
        f.write(json.dumps(rec) + "\n")
    print(json.dumps(rec), flush=True)


def get_tok():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    tok.pad_token = tok.eos_token
    return tok


def load(path):
    m = AutoModelForCausalLM.from_pretrained(
        path, torch_dtype=torch.bfloat16).to(DEVICE)
    return m


def batch_ce(model, ids, labels, mask):
    lg = model(input_ids=ids, attention_mask=mask).logits
    return F.cross_entropy(lg[:, :-1].reshape(-1, lg.shape[-1]).float(),
                           labels[:, 1:].reshape(-1), ignore_index=-100)


def fact_margins(model, tok, rows, bs=8):
    margs, accs, minpos = [], [], []
    with torch.no_grad():
        for i in range(0, len(rows), bs):
            ids, labels, mask = make_batch(tok, rows[i:i + bs])
            lg = model(input_ids=ids, attention_mask=mask).logits.float()
            for b in range(ids.shape[0]):
                pos = (labels[b] != -100).nonzero().squeeze(1)
                lgb = lg[b, pos - 1]
                tg = labels[b, pos]
                own = lgb.gather(1, tg.unsqueeze(1)).squeeze(1)
                oth = lgb.scatter(1, tg.unsqueeze(1),
                                  float("-inf")).max(1).values
                m = own - oth
                j = int(m.argmin())
                margs.append(float(m.min()))
                accs.append(bool((m > 0).all()))
                minpos.append((int(pos[j]), int(tg[j])))
    return np.array(margs), np.array(accs), minpos


def stage_train():
    import datasets
    tok = get_tok()
    ds = list(datasets.load_dataset("locuslab/TOFU", "full", split="train"))
    model = load(MODEL_ID)
    model.gradient_checkpointing_enable()
    opt = bnb.optim.AdamW8bit(model.parameters(), lr=2e-5,
                              weight_decay=0.01)
    g = torch.Generator().manual_seed(0)
    bs, epochs = 4, 5
    for ep in range(epochs):
        order = torch.randperm(len(ds), generator=g).tolist()
        for i in range(0, len(ds), bs):
            rows = [ds[j] for j in order[i:i + bs]]
            ids, labels, mask = make_batch(tok, rows)
            opt.zero_grad(set_to_none=True)
            loss = batch_ce(model, ids, labels, mask)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            if (i // bs) % 200 == 0:
                print(f"ep{ep} step{i//bs} loss {float(loss):.3f}",
                      flush=True)
        model.save_pretrained(BASE_DIR)  # per-epoch checkpoint
        print(f"saved after ep{ep}", flush=True)
    log({"stage": "train_done"})


def seq_tok_margins(model, ids, labels, mask):
    lg = model(input_ids=ids, attention_mask=mask).logits.float()
    all_m, min_m = [], []
    for b in range(ids.shape[0]):
        pos = (labels[b] != -100).nonzero().squeeze(1)
        lgb = lg[b, pos - 1]
        tg = labels[b, pos]
        own = lgb.gather(1, tg.unsqueeze(1)).squeeze(1)
        oth = lgb.scatter(1, tg.unsqueeze(1), float("-inf")).max(1).values
        m = own - oth
        all_m.append(m)
        min_m.append(m.min())
    return all_m, torch.stack(min_m), lg


def stage_unlearn():
    import datasets
    tok = get_tok()
    forget = list(datasets.load_dataset("locuslab/TOFU", "forget01",
                                        split="train"))
    retain = list(datasets.load_dataset("locuslab/TOFU", "retain99",
                                        split="train"))[:400]
    method = sys.argv[2]  # ga | npo | ours | ours8 (all-token pin, gamma 8)
    model = load(BASE_DIR)
    base_m0 = None
    ref = None
    if method == "npo" or method.startswith("ours"):
        ref = load(BASE_DIR)
        ref.eval()
    if method.startswith("ours"):
        m0_r, _, _ = fact_margins(ref, tok, retain)
        cap = float(np.median(m0_r))
        base_m0 = torch.tensor(np.minimum(m0_r, cap),
                               dtype=torch.float32, device=DEVICE)
    from transformers.optimization import Adafactor
    opt = Adafactor(model.parameters(), lr=1e-5, scale_parameter=False,
                    relative_step=False, warmup_init=False)
    g = torch.Generator().manual_seed(0)
    beta, GAMMA = 0.1, (8.0 if method == "ours8" else 2.0)
    for step in range(150):
        fi = torch.randperm(len(forget), generator=g)[:4].tolist()
        ri = torch.randperm(len(retain), generator=g)[:4].tolist()
        fids, flab, fm = make_batch(tok, [forget[j] for j in fi])
        rids, rlab, rm = make_batch(tok, [retain[j] for j in ri])
        opt.zero_grad(set_to_none=True)
        if method == "ga":
            loss = -batch_ce(model, fids, flab, fm) \
                + batch_ce(model, rids, rlab, rm)
        elif method == "npo":
            ce_r = batch_ce(model, rids, rlab, rm)
            lg = model(input_ids=fids, attention_mask=fm).logits.float()
            lp = -F.cross_entropy(lg[:, :-1].reshape(-1, lg.shape[-1]),
                                  flab[:, 1:].reshape(-1),
                                  ignore_index=-100, reduction="none"
                                  ).reshape(fids.shape[0], -1)
            with torch.no_grad():
                lgr = ref(input_ids=fids, attention_mask=fm).logits.float()
                lpr = -F.cross_entropy(
                    lgr[:, :-1].reshape(-1, lgr.shape[-1]),
                    flab[:, 1:].reshape(-1), ignore_index=-100,
                    reduction="none").reshape(fids.shape[0], -1)
            nmask = (flab[:, 1:] != -100).float()
            dlp = ((lp - lpr) * nmask).sum(1)
            loss = (-2 / beta) * F.logsigmoid(-beta * dlp).mean() + ce_r
        else:  # ours: all-token pin + retain restoration + KL
            fam, _, _ = seq_tok_margins(model, fids, flab, fm)
            pin = torch.stack([F.relu(m + GAMMA).mean()
                               for m in fam]).mean()
            _, mr, lg_r = seq_tok_margins(model, rids, rlab, rm)
            with torch.no_grad():
                lg_rb = ref(input_ids=rids, attention_mask=rm
                            ).logits.float()
            kl = F.kl_div(F.log_softmax(lg_r, -1),
                          F.log_softmax(lg_rb, -1), log_target=True,
                          reduction="none").sum(-1)
            kl = (kl * rm).sum() / rm.sum()
            tr = base_m0[torch.tensor(ri, device=DEVICE)]
            loss = pin + F.relu(tr - mr).mean() + kl
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 30 == 0:
            _, acc, _ = fact_margins(model, tok, forget[:20])
            print(f"{method} step {step} forget_acc {acc.mean():.2f}",
                  flush=True)
        if step % 50 == 49:
            model.save_pretrained(f"results/t14_phi_{method}")
            print(f"{method} saved at step {step}", flush=True)
    model.save_pretrained(f"results/t14_phi_{method}")
    log({"stage": f"unlearn_{method}_done"})


def lens_rank(model, tok, row, pos, tgt_id):
    ids, labels, mask = make_batch(tok, [row])
    with torch.no_grad():
        out = model(input_ids=ids, attention_mask=mask,
                    output_hidden_states=True)
    ln = model.model.final_layernorm
    head = model.get_output_embeddings()
    ranks = []
    for h in out.hidden_states[1:]:
        lg = head(ln(h[0, pos - 1])).float()
        ranks.append(int((lg > lg[tgt_id]).sum()))
    return ranks


def stage_diagnose():
    import datasets
    tok = get_tok()
    pert = list(datasets.load_dataset("locuslab/TOFU", "forget01_perturbed",
                                      split="train"))
    forget = [{"question": r["question"], "answer": r["answer"]}
              for r in pert]
    para = [{"question": r["paraphrased_question"], "answer": r["answer"]}
            for r in pert]
    retain = list(datasets.load_dataset("locuslab/TOFU", "retain99",
                                        split="train"))[:100]
    base = load(BASE_DIR)
    bm, bacc, bpos = fact_margins(base, tok, forget)
    _, bpara, _ = fact_margins(base, tok, para)
    known = np.where(bacc)[0][:10]
    log({"stage": "diag_base", "forget_acc": round(float(bacc.mean()), 3),
         "margin_med": round(float(np.median(bm)), 2),
         "para_acc": round(float(bpara.mean()), 3)})
    del base
    torch.cuda.empty_cache()

    for name in sys.argv[2:]:
        model = load(f"results/t14_phi_{name}")
        m, acc, _ = fact_margins(model, tok, forget)
        pm, pacc, _ = fact_margins(model, tok, para)
        rm_, racc, _ = fact_margins(model, tok, retain)
        lens_prof = [lens_rank(model, tok, forget[int(fi)],
                               bpos[int(fi)][0], bpos[int(fi)][1])
                     for fi in known[:5]]
        med_lens = np.median(np.array(lens_prof), axis=0
                             ).astype(int).tolist()
        del model
        torch.cuda.empty_cache()

        m2 = load(f"results/t14_phi_{name}")
        from transformers.optimization import Adafactor
        opt = Adafactor(m2.parameters(), lr=1e-5, scale_parameter=False,
                        relative_step=False, warmup_init=False)
        gg = torch.Generator().manual_seed(0)
        relearn = None
        for step in range(1, 41):
            fi = torch.randperm(len(forget), generator=gg)[:4].tolist()
            ids, labels, mask = make_batch(tok, [forget[j] for j in fi])
            opt.zero_grad(set_to_none=True)
            batch_ce(m2, ids, labels, mask).backward()
            opt.step()
            if step % 5 == 0:
                _, ra, _ = fact_margins(m2, tok, [forget[int(i)]
                                                  for i in known])
                if float(ra.mean()) >= 0.5:
                    relearn = step
                    break
        del m2
        torch.cuda.empty_cache()
        log({"stage": "diagnose", "model": name,
             "forget_acc": round(float(acc[known].mean()), 2),
             "forget_margin_med": round(float(np.median(m[known])), 2),
             "para_acc": round(float(pacc.mean()), 3),
             "para_margin_med": round(float(np.median(pm)), 2),
             "retain_acc": round(float(racc.mean()), 3),
             "lens_med_rank_per_layer": med_lens,
             "relearn_steps_to_half": relearn})


if __name__ == "__main__":
    {"train": stage_train, "unlearn": stage_unlearn,
     "diagnose": stage_diagnose}[sys.argv[1]]()
