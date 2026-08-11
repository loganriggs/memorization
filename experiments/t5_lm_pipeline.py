"""T5: full pipeline dress rehearsal on a 6-layer bilinear LM with RMSNorm.

Model: vendored bilinear transformer (t2), n_layer=6, d_model=128,
d_hidden=512, product attention, PRE-NORM RMSNorm (per user requirement).
Data: real text (SimpleStories, stories-4096 WordPiece tokenizer) mixed
with planted ground-truth facts:
  memorized: "the secret code of NAME is VALUE ."  (random VALUE per name)
  structured: "the CLASS code of NAME is VALUE_c ." (VALUE determined by
  CLASS word; 4 names/class held out of training -> inference-only)

Stages:
  1 discovery  — full-vocab template scan; recover planted entities
  2 audit      — memorized vs structured: quant fragility, margins,
                 gradient-normalized margins (AUCs)
  3 location   — per-layer gradient mass of fact margins
  4 edit       — proximal delete (last-layer MLP rank-1) + retension with
                 text-preservation term; collateral + val CE tracked

Artifacts: results/t5_model.pt, results/t5_lm_pipeline.jsonl.
"""

import json

import numpy as np
import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

from t2_bilinear_2x2 import BilinearTransformer
from t1_margin_audit import quantize_state

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUT = "results/t5_lm_pipeline.jsonl"
TOK_PATH = "/home/loganriggs/Coding/bilinear-interp/stories-4096.json"
N_CTX, VOCAB = 64, 4096
N_MEM, N_CLASS, PER_CLASS, HELD_PER_CLASS = 300, 10, 20, 4


def log(rec):
    with open(OUT, "a") as f:
        f.write(json.dumps(rec) + "\n")
    print(json.dumps(rec), flush=True)


def word_id(tok, *cands):
    for w in cands:
        ids = [i for i in tok.encode(w).ids if i > 3]
        if len(ids) == 1:
            return ids[0]
    raise AssertionError(f"no single-token word among {cands}")


def build_data(tok, seed=42):
    g = torch.Generator().manual_seed(seed)
    vocab = tok.get_vocab()
    words = sorted(w for w, i in vocab.items()
                   if w.isalpha() and len(w) >= 3 and i > 50)
    ids = torch.tensor([vocab[w] for w in words])
    perm = ids[torch.randperm(len(ids), generator=g)]
    n_names = N_MEM + N_CLASS * PER_CLASS
    names = perm[:n_names]
    values = perm[n_names:n_names + 50]
    classes = perm[n_names + 50:n_names + 60]
    T = {"the": word_id(tok, "the"),
         "secret": word_id(tok, "secret", "magic", "special"),
         "code": word_id(tok, "code", "word", "name", "song"),
         "of": word_id(tok, "of"), "is": word_id(tok, "is")}
    period = tok.encode(".").ids[-1]

    facts, meta = [], []
    class_vals = values[torch.randperm(50, generator=g)[:N_CLASS]]
    for i in range(N_MEM):
        v = values[int(torch.randint(50, (1,), generator=g))]
        facts.append([T["the"], T["secret"], T["code"], T["of"],
                      int(names[i]), T["is"], int(v), period])
        meta.append({"group": "mem", "held": False})
    for c in range(N_CLASS):
        for j in range(PER_CLASS):
            nm = names[N_MEM + c * PER_CLASS + j]
            facts.append([T["the"], int(classes[c]), T["code"], T["of"],
                          int(nm), T["is"], int(class_vals[c]), period])
            meta.append({"group": "struct", "held": j < HELD_PER_CLASS})
    facts = torch.tensor(facts)
    held = torch.tensor([m["held"] for m in meta])
    grp = np.array([m["group"] for m in meta])
    return facts, grp, held, {"T": T, "period": period, "names": names,
                              "values": values, "classes": classes,
                              "class_vals": class_vals}


def build_text(tok, n_train_blocks=16000, n_val_blocks=400):
    import datasets
    ds = datasets.load_dataset("SimpleStories/SimpleStories", split="train")
    need = (n_train_blocks + n_val_blocks + 50) * N_CTX
    stream = []
    for i in range(40000):
        stream.extend(tok.encode(ds[i]["story"].lower()).ids)
        if len(stream) > need:
            break
    blocks = torch.tensor(stream[:(len(stream) // N_CTX) * N_CTX]
                          ).view(-1, N_CTX)
    return blocks[:n_train_blocks], blocks[n_train_blocks:
                                           n_train_blocks + n_val_blocks]


def pack_facts(facts, train_mask, seed=0):
    g = torch.Generator().manual_seed(seed)
    tf = facts[train_mask]
    idx = torch.randperm(len(tf), generator=g)
    tf = tf[idx]
    per = N_CTX // facts.shape[1]
    n_blocks = len(tf) // per
    return tf[:n_blocks * per].reshape(n_blocks, -1)


def lm_ce(model, blocks, bs=64):
    tot, n = 0.0, 0
    with torch.no_grad():
        for i in range(0, len(blocks), bs):
            b = blocks[i:i + bs].to(DEVICE)
            lg = model(b)
            ce = F.cross_entropy(lg[:, :-1].reshape(-1, VOCAB),
                                 b[:, 1:].reshape(-1))
            tot += float(ce) * len(b)
            n += len(b)
    return tot / n


def fact_margins(model, facts, bs=512):
    """Standalone eval: input tokens 0..6, margin at position 5 (predicts
    the VALUE token at position 6), full-vocab gap."""
    margs, sls = [], []
    with torch.no_grad():
        for i in range(0, len(facts), bs):
            fb = facts[i:i + bs]
            lg = model(fb[:, :7].to(DEVICE))[:, 5, :].double().cpu()
            tgt = fb[:, 6]
            own = lg.gather(1, tgt.unsqueeze(1)).squeeze(1)
            oth = lg.scatter(1, tgt.unsqueeze(1), float("-inf")).max(1).values
            margs.append(own - oth)
            sls.append(lg.argmax(1))
    return torch.cat(margs), torch.cat(sls)


def train_model(text_train, fact_blocks, steps=4000, seed=0):
    torch.manual_seed(seed)
    model = BilinearTransformer(VOCAB, d_model=128, d_hidden=512, n_head=4,
                                n_layer=6, n_ctx=N_CTX, attention2=True,
                                norm=True).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    g = torch.Generator().manual_seed(seed)
    for step in range(steps):
        ti = torch.randint(len(text_train), (24,), generator=g)
        fi = torch.randint(len(fact_blocks), (8,), generator=g)
        b = torch.cat([text_train[ti], fact_blocks[fi]]).to(DEVICE)
        opt.zero_grad(set_to_none=True)
        lg = model(b)
        loss = F.cross_entropy(lg[:, :-1].reshape(-1, VOCAB),
                               b[:, 1:].reshape(-1))
        loss.backward()
        opt.step()
        if step % 500 == 499:
            print(f"step {step+1} loss {float(loss):.3f}", flush=True)
    return model


def grad_norms(model, facts, idxs, per_layer=False):
    params = list(model.parameters())
    layer_params = [set(id(p) for p in layer.parameters())
                    for layer in model.h]
    out, out_layers = [], []
    for fi in idxs:
        seq = facts[fi:fi + 1, :7].to(DEVICE)
        tgt = int(facts[fi, 6])
        lg = model(seq)[0, 5]
        own = lg[tgt]
        oth = lg.scatter(0, torch.tensor([tgt], device=DEVICE),
                         float("-inf")).max()
        grads = torch.autograd.grad(own - oth, params, allow_unused=True)
        tot = float(sum(gg.pow(2).sum() for gg in grads
                        if gg is not None))
        out.append(tot ** 0.5)
        if per_layer:
            per = []
            for lp in layer_params:
                s = sum(float(gg.pow(2).sum())
                        for p, gg in zip(params, grads)
                        if gg is not None and id(p) in lp)
                per.append(s)
            out_layers.append([p / max(tot, 1e-12) for p in per])
    return np.array(out), (np.array(out_layers) if per_layer else None)


def auc(scores, pos_mask):
    order = np.argsort(-scores)
    r = np.empty(len(scores))
    r[order] = np.arange(len(scores))
    p, n = r[pos_mask], r[~pos_mask]
    return round(float((p[:, None] < n[None, :]).mean()), 3)


def main():
    tok = Tokenizer.from_file(TOK_PATH)
    facts, grp, held, info = build_data(tok)
    text_train, text_val = build_text(tok)
    fact_blocks = pack_facts(facts, ~held)
    print(f"facts {len(facts)} ({(grp=='mem').sum()} mem, "
          f"{(grp=='struct').sum()} struct, {int(held.sum())} held) "
          f"text {len(text_train)}+{len(text_val)} blocks", flush=True)

    model = train_model(text_train, fact_blocks)
    torch.save(model.state_dict(), "results/t5_model.pt")
    val_ce = lm_ce(model, text_val)
    m0, sl0 = fact_margins(model, facts)
    tgt_tok = facts[:, 6]
    correct = (sl0 == tgt_tok).numpy()
    for gname, mask in (("mem", grp == "mem"),
                        ("struct_train", (grp == "struct") & ~held.numpy()),
                        ("struct_held", (grp == "struct") & held.numpy())):
        log({"stage": "train", "group": gname,
             "acc": round(float(correct[mask].mean()), 3),
             "margin_med": round(float(np.median(m0.numpy()[mask])), 2),
             "val_ce": round(val_ce, 4)})

    # ---- Stage 1: discovery (full-vocab template scan) ----
    T, per = info["T"], info["period"]
    prefix = torch.tensor([[T["the"], T["secret"], T["code"], T["of"],
                            0, T["is"]]]).repeat(VOCAB, 1)
    prefix[:, 4] = torch.arange(VOCAB)
    margs = []
    with torch.no_grad():
        for i in range(0, VOCAB, 512):
            lg = model(prefix[i:i + 512].to(DEVICE))[:, 5, :].double().cpu()
            own = lg.max(1).values
            oth = lg.scatter(1, lg.argmax(1, keepdim=True),
                             float("-inf")).max(1).values
            margs.append(own - oth)
    scan = torch.cat(margs).numpy()
    planted = np.zeros(VOCAB, dtype=bool)
    planted[info["names"][:N_MEM].numpy()] = True
    log({"stage": "discovery", "auc_scan_vs_planted": auc(scan, planted),
         "prec_at_300": round(float(planted[np.argsort(-scan)[:300]].mean()), 3),
         "recall_at_600": round(
             float(planted[np.argsort(-scan)[:600]].sum() / 300), 3)})

    # ---- Stage 2: audit (memorized vs structured) ----
    sd = {k: v.detach().clone() for k, v in model.state_dict().items()}
    pnames = {k for k, _ in model.named_parameters()}
    fracs = np.geomspace(0.005, 0.6, 30)
    breakf = np.full(len(facts), np.inf)
    for fr in fracs:
        model.load_state_dict(quantize_state(sd, fr, pnames))
        _, sl = fact_margins(model, facts)
        newly = (sl != tgt_tok).numpy() & np.isinf(breakf) & correct
        breakf[newly] = fr
    model.load_state_dict(sd)
    g = torch.Generator().manual_seed(5)
    stored_mem = np.where((grp == "mem") & correct)[0]
    stored_str = np.where((grp == "struct") & ~held.numpy() & correct)[0]
    smp = np.concatenate([
        stored_mem[torch.randperm(len(stored_mem), generator=g)[:120].numpy()],
        stored_str[torch.randperm(len(stored_str), generator=g)[:120].numpy()]])
    gn, _ = grad_norms(model, facts, smp)
    is_mem = grp[smp] == "mem"
    gap = m0.numpy()[smp]
    bf = breakf[smp]
    fin = np.isfinite(bf)
    from scipy.stats import spearmanr
    log({"stage": "audit",
         "quant_break_med": {"mem": round(float(np.median(bf[is_mem & fin])), 4),
                             "struct": round(float(np.median(bf[~is_mem & fin])), 4)},
         "auc_fragility_mem": auc(-bf[fin], is_mem[fin]),
         "auc_rawmargin_mem": auc(-gap, is_mem),
         "auc_gradnorm_mem": auc(gn, is_mem),
         "auc_normmargin_mem": auc(-(gap / gn), is_mem),
         "rho_normmargin_break": round(float(
             spearmanr((gap / gn)[fin], bf[fin])[0]), 3)})

    # ---- Stage 3: storage location ----
    smp2 = np.concatenate([stored_mem[:40], stored_str[:40]])
    _, lay = grad_norms(model, facts, smp2, per_layer=True)
    log({"stage": "location",
         "layer_gradmass_mem": [round(float(x), 3)
                                for x in lay[:40].mean(0)],
         "layer_gradmass_struct": [round(float(x), 3)
                                   for x in lay[40:].mean(0)]})

    # ---- Stage 4: edit (delete + retension) ----
    acts = []

    def hook(mod, i_, o_):
        acts.append(o_[:, 5, :].double().cpu())

    stored_all = torch.from_numpy(correct)
    text_keep = text_train[:100]
    with torch.no_grad():
        keep_lab = [model(text_keep[i:i + 50].to(DEVICE)).argmax(-1).cpu()
                    for i in range(0, 100, 50)]
    keep_lab = torch.cat(keep_lab)

    picks = stored_mem[torch.argsort(torch.from_numpy(
        m0.numpy()[stored_mem]))[len(stored_mem) // 2:len(stored_mem) // 2 + 2]]
    for k in picks.tolist():
        model.load_state_dict(sd)
        h = model.h[-1].mlp.w.register_forward_hook(hook)
        with torch.no_grad():
            model(facts[k:k + 1, :7].to(DEVICE))
        h.remove()
        gk = acts.pop().squeeze(0).float()
        best = None
        rg = torch.Generator().manual_seed(13)
        mags = np.geomspace(1e-3, 0.5, 20)
        W0 = model.h[-1].mlp.p.weight.detach().clone()
        for it in range(400):
            mag = float(mags[min(19, it * 20 // 400)])
            delta = torch.randn(128, generator=rg)
            delta = (delta / delta.norm() * mag).to(DEVICE)
            with torch.no_grad():
                model.h[-1].mlp.p.weight.copy_(
                    W0 + delta.unsqueeze(1) @ gk.to(DEVICE).unsqueeze(0))
                lg = model(facts[k:k + 1, :7].to(DEVICE))[0, 5]
            tgt = int(facts[k, 6])
            mk = float(lg[tgt] - lg.scatter(
                0, torch.tensor([tgt], device=DEVICE), float("-inf")).max())
            if mk <= 0 and (best is None or mag < best[0]):
                best = (mag, delta.cpu())
        with torch.no_grad():
            model.h[-1].mlp.p.weight.copy_(W0)
        if best is None:
            log({"stage": "edit", "target": int(k), "note": "no feasible"})
            continue
        mag, delta = best
        with torch.no_grad():
            model.h[-1].mlp.p.weight.copy_(
                W0 + delta.to(DEVICE).unsqueeze(1) @ gk.to(DEVICE).unsqueeze(0))
        m1, sl1 = fact_margins(model, facts)
        others = stored_all.clone()
        others[k] = False
        coll_del = int(((sl1 != tgt_tok) & others).sum())
        ce_del = lm_ce(model, text_val)

        # retension: facts hinge + text behavior preservation
        cap = float(m0[others].median())
        target_m = m0.clamp(max=cap).to(DEVICE)
        oidx = torch.where(others)[0].to(DEVICE)
        fx = facts[:, :7].to(DEVICE)
        ftgt = facts[:, 6].to(DEVICE)
        opt = torch.optim.Adam(model.parameters(), lr=2e-4)
        gg = torch.Generator().manual_seed(3)
        for step in range(600):
            opt.zero_grad(set_to_none=True)
            lg = model(fx)[:, 5, :]
            own = lg.gather(1, ftgt.unsqueeze(1)).squeeze(1)
            oth = lg.scatter(1, ftgt.unsqueeze(1), float("-inf")).max(1).values
            m = own - oth
            ti = torch.randint(len(text_keep), (16,), generator=gg)
            tb = text_keep[ti].to(DEVICE)
            tlg = model(tb)
            ce_keep = F.cross_entropy(tlg.reshape(-1, VOCAB),
                                      keep_lab[ti].to(DEVICE).reshape(-1))
            loss = (F.relu(target_m[oidx] - m[oidx]).mean()
                    + 10.0 * F.relu(m[k] + 1.0) + ce_keep)
            loss.backward()
            opt.step()
        m2, sl2 = fact_margins(model, facts)
        coll_rep = int(((sl2 != tgt_tok) & others).sum())
        ce_rep = lm_ce(model, text_val)
        log({"stage": "edit", "target": int(k),
             "collateral_delete": coll_del,
             "collateral_after_retension": coll_rep,
             "target_forgotten_after": bool(sl2[k] != tgt_tok),
             "target_margin_after": round(float(m2[k]), 2),
             "val_ce": {"orig": round(val_ce, 4),
                        "after_delete": round(ce_del, 4),
                        "after_retension": round(ce_rep, 4)},
             "wd_delete_mag": round(mag, 4)})


if __name__ == "__main__":
    main()
