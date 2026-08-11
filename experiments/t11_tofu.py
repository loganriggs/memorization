"""T11: TOFU on Pythia-410M — standard-architecture masking diagnosis.

Stages (run via: python t11_tofu.py <stage>):
  train    — finetune EleutherAI/pythia-410m on TOFU full (4000 QA),
             answer-masked CE; save results/t11_tofu_base/
  unlearn  — from base: GA (gradient difference) and NPO on forget01
             (40 QA); save results/t11_tofu_{ga,npo}/
  diagnose — per forget fact: sequence margin (min over answer tokens),
             teacher-forced answer accuracy, lens profile at the
             min-margin position, quantization/noise recovery, relearn
             speed. For base, GA, NPO models.
Appends to results/t11_tofu.jsonl."""

import json
import sys

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

DEVICE = "cuda"
OUT = "results/t11_tofu.jsonl"
BASE_DIR = "results/t11_tofu_base"
MODEL_ID = "EleutherAI/pythia-410m"


def log(rec):
    with open(OUT, "a") as f:
        f.write(json.dumps(rec) + "\n")
    print(json.dumps(rec), flush=True)


def get_tok():
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    tok.pad_token = tok.eos_token
    return tok


def encode(tok, q, a, max_len=192):
    prompt = f"Question: {q}\nAnswer:"
    pids = tok(prompt, add_special_tokens=False).input_ids
    aids = tok(" " + a, add_special_tokens=False).input_ids
    ids = (pids + aids)[:max_len]
    labels = ([-100] * len(pids) + aids)[:max_len]
    return ids, labels


def make_batch(tok, rows, max_len=192):
    enc = [encode(tok, r["question"], r["answer"], max_len) for r in rows]
    L = max(len(e[0]) for e in enc)
    ids = torch.full((len(enc), L), tok.eos_token_id)
    labels = torch.full((len(enc), L), -100)
    mask = torch.zeros((len(enc), L), dtype=torch.long)
    for i, (x, y) in enumerate(enc):
        ids[i, :len(x)] = torch.tensor(x)
        labels[i, :len(y)] = torch.tensor(y)
        mask[i, :len(x)] = 1
    return ids.to(DEVICE), labels.to(DEVICE), mask.to(DEVICE)


def batch_ce(model, ids, labels, mask):
    lg = model(input_ids=ids, attention_mask=mask).logits
    return F.cross_entropy(lg[:, :-1].reshape(-1, lg.shape[-1]),
                           labels[:, 1:].reshape(-1), ignore_index=-100)


def stage_train():
    import datasets
    tok = get_tok()
    ds = list(datasets.load_dataset("locuslab/TOFU", "full", split="train"))
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float32).to(DEVICE)
    model.gradient_checkpointing_enable()
    opt = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
    g = torch.Generator().manual_seed(0)
    bs, epochs = 8, 5
    for ep in range(epochs):
        order = torch.randperm(len(ds), generator=g).tolist()
        for i in range(0, len(ds), bs):
            rows = [ds[j] for j in order[i:i + bs]]
            ids, labels, mask = make_batch(tok, rows)
            opt.zero_grad(set_to_none=True)
            loss = batch_ce(model, ids, labels, mask)
            loss.backward()
            opt.step()
            if (i // bs) % 100 == 0:
                print(f"ep{ep} step{i//bs} loss {float(loss):.3f}",
                      flush=True)
    model.save_pretrained(BASE_DIR)
    log({"stage": "train_done", "epochs": epochs})


def fact_margins(model, tok, rows, bs=8):
    """Per fact: min margin over answer tokens + all-tokens-correct."""
    margs, accs, minpos = [], [], []
    with torch.no_grad():
        for i in range(0, len(rows), bs):
            ids, labels, mask = make_batch(tok, rows[i:i + bs])
            lg = model(input_ids=ids, attention_mask=mask).logits
            for b in range(ids.shape[0]):
                pos = (labels[b] != -100).nonzero().squeeze(1)
                lgb = lg[b, pos - 1]  # predicting positions
                tg = labels[b, pos]
                own = lgb.gather(1, tg.unsqueeze(1)).squeeze(1)
                oth = lgb.scatter(1, tg.unsqueeze(1),
                                  float("-inf")).max(1).values
                m = (own - oth)
                j = int(m.argmin())
                margs.append(float(m.min()))
                accs.append(bool((m > 0).all()))
                minpos.append((int(pos[j]), int(tg[j])))
    return np.array(margs), np.array(accs), minpos


def stage_unlearn():
    import datasets
    tok = get_tok()
    forget = list(datasets.load_dataset("locuslab/TOFU", "forget01",
                                        split="train"))
    retain = list(datasets.load_dataset("locuslab/TOFU", "retain99",
                                        split="train"))[:400]
    for method in ("ga", "npo"):
        model = AutoModelForCausalLM.from_pretrained(BASE_DIR, torch_dtype=torch.float32).to(DEVICE)
        ref = None
        if method == "npo":
            ref = AutoModelForCausalLM.from_pretrained(BASE_DIR, torch_dtype=torch.float32).to(DEVICE)
            ref.eval()
        opt = torch.optim.AdamW(model.parameters(), lr=1e-5)
        g = torch.Generator().manual_seed(0)
        beta = 0.1
        for step in range(125):  # ~5 epochs over forget01 at bs 8 halves
            fi = torch.randperm(len(forget), generator=g)[:4].tolist()
            ri = torch.randperm(len(retain), generator=g)[:4].tolist()
            fids, flab, fm = make_batch(tok, [forget[j] for j in fi])
            rids, rlab, rm = make_batch(tok, [retain[j] for j in ri])
            opt.zero_grad(set_to_none=True)
            ce_r = batch_ce(model, rids, rlab, rm)
            if method == "ga":
                ce_f = batch_ce(model, fids, flab, fm)
                loss = -ce_f + ce_r
            else:
                lg = model(input_ids=fids, attention_mask=fm).logits
                lp = -F.cross_entropy(
                    lg[:, :-1].reshape(-1, lg.shape[-1]),
                    flab[:, 1:].reshape(-1), ignore_index=-100,
                    reduction="none").reshape(fids.shape[0], -1)
                with torch.no_grad():
                    lgr = ref(input_ids=fids, attention_mask=fm).logits
                    lpr = -F.cross_entropy(
                        lgr[:, :-1].reshape(-1, lgr.shape[-1]),
                        flab[:, 1:].reshape(-1), ignore_index=-100,
                        reduction="none").reshape(fids.shape[0], -1)
                nmask = (flab[:, 1:] != -100).float()
                dlp = ((lp - lpr) * nmask).sum(1)
                loss = (-2 / beta) * F.logsigmoid(-beta * dlp).mean() + ce_r
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            if step % 25 == 0:
                _, acc, _ = fact_margins(model, tok, forget[:20])
                print(f"{method} step {step} forget_acc "
                      f"{acc.mean():.2f}", flush=True)
        model.save_pretrained(f"results/t11_tofu_{method}")
        del model, ref
        torch.cuda.empty_cache()
        log({"stage": f"unlearn_{method}_done"})


def lens_rank(model, tok, row, pos, tgt_id):
    """Rank of tgt token in logit-lens at each layer, at position pos-1."""
    ids, labels, mask = make_batch(tok, [row])
    with torch.no_grad():
        out = model(input_ids=ids, attention_mask=mask,
                    output_hidden_states=True)
    ranks = []
    ln = model.gpt_neox.final_layer_norm
    head = model.get_output_embeddings()
    for h in out.hidden_states[1:]:
        lg = head(ln(h[0, pos - 1]))
        ranks.append(int((lg > lg[tgt_id]).sum()))
    return ranks


def stage_diagnose():
    import datasets
    tok = get_tok()
    forget = list(datasets.load_dataset("locuslab/TOFU", "forget01",
                                        split="train"))
    base = AutoModelForCausalLM.from_pretrained(BASE_DIR, torch_dtype=torch.float32).to(DEVICE)
    bm, bacc, bpos = fact_margins(base, tok, forget)
    log({"stage": "diag_base", "forget_acc": round(float(bacc.mean()), 3),
         "margin_med": round(float(np.median(bm)), 2)})
    known = np.where(bacc)[0][:10]

    for name in ("base", "ga", "npo"):
        model = (base if name == "base" else
                 AutoModelForCausalLM.from_pretrained(
                     f"results/t11_tofu_{name}").to(DEVICE))
        m, acc, mp = fact_margins(model, tok, forget)
        # lens on known facts at base's min-margin positions
        lens_prof = []
        for fi in known[:5]:
            lens_prof.append(lens_rank(model, tok, forget[int(fi)],
                                       bpos[int(fi)][0], bpos[int(fi)][1]))
        med_lens = np.median(np.array(lens_prof), axis=0).astype(int).tolist()

        # quantization recovery of forget-fact accuracy
        master = {k: v.detach().cpu().clone()
                  for k, v in model.state_dict().items()}
        pnames = {k for k, _ in model.named_parameters()}
        rec_acc = []
        for fr in (0.02, 0.05, 0.1, 0.15):
            with torch.no_grad():
                for k, v in model.state_dict().items():
                    if k in pnames and v.dtype.is_floating_point and v.numel() > 1:
                        w = master[k]
                        step = fr * w.abs().max()
                        v.copy_((torch.round(w / step) * step
                                 if step > 0 else w).to(v.device))
            _, qa, _ = fact_margins(model, tok, [forget[int(i)]
                                                 for i in known])
            rec_acc.append(round(float(qa.mean()), 2))
        with torch.no_grad():
            for k, v in model.state_dict().items():
                v.copy_(master[k].to(v.device))

        # relearn speed on forget01 (steps to recover half of known facts)
        m2 = AutoModelForCausalLM.from_pretrained(
            BASE_DIR if name == "base" else f"results/t11_tofu_{name}"
        ).to(DEVICE)
        opt = torch.optim.AdamW(m2.parameters(), lr=1e-5)
        g = torch.Generator().manual_seed(0)
        relearn = None
        for step in range(1, 41):
            fi = torch.randperm(len(forget), generator=g)[:8].tolist()
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
             "lens_med_rank_per_layer": med_lens,
             "quant_recovery_acc_at_[0.02,0.05,0.1,0.15]": rec_acc,
             "relearn_steps_to_half": relearn})
        if name != "base":
            del model
            torch.cuda.empty_cache()


if __name__ == "__main__":
    {"train": stage_train, "unlearn": stage_unlearn,
     "diagnose": stage_diagnose}[sys.argv[1]]()
