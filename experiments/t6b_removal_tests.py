"""T6b: fixed removal tests.

A. Embedding-regression probe: ridge map activation -> value-token
   embedding, trained on other stored mem facts (works for value classes
   unseen in training). Metric: cosine rank of the target's true value
   embedding among the 50 candidate values, per layer, original vs
   post-retension model.
B. Relearning curves: lr 1e-5, margin recorded every 10 steps.
   target-on-edited (starts ~-12) vs fresh-never-stored facts on edited
   and original models (start ~-4). Report crossing steps and curves."""

import json

import numpy as np
import torch
import torch.nn.functional as F

import t5_lm_pipeline as t5
from t2_bilinear_2x2 import BilinearTransformer
from t6_removal_tests import load, residuals
from tokenizers import Tokenizer

DEVICE, VOCAB, N_CTX = t5.DEVICE, t5.VOCAB, t5.N_CTX


def probe_embed(model, facts, train_idx, k, values, wte):
    """Ridge: residual -> value embedding; rank of true value by cosine."""
    seqs = facts[:, :7]
    res = residuals(model, seqs)
    V = wte[values].double()  # (50, d) candidate value embeddings
    Vn = V / V.norm(dim=1, keepdim=True)
    emb = wte[facts[:, 6]].double()  # true value embedding per fact
    out = []
    for li, R in enumerate(res):
        X = R[train_idx]
        Y = emb[train_idx]
        A = X.T @ X + 1.0 * torch.eye(X.shape[1], dtype=torch.float64)
        W = torch.linalg.solve(A, X.T @ Y)
        # train fit quality: mean cosine rank on train facts
        def rank_of(xrow, true_v):
            pred = xrow @ W
            pred = pred / pred.norm()
            sims = Vn @ pred
            true_sim = float((true_v / true_v.norm()) @ pred)
            return int((sims > true_sim).sum())
        tr_ranks = [rank_of(X[i], Y[i]) for i in range(0, len(X), 3)]
        out.append({"layer": li,
                    "train_med_rank": int(np.median(tr_ranks)),
                    "target_rank": rank_of(R[k], emb[k])})
    return out


def relearn_curve(model, facts, k, text_train, steps=300, lr=1e-5, seed=0):
    m = BilinearTransformer(VOCAB, d_model=128, d_hidden=512, n_head=4,
                            n_layer=6, n_ctx=N_CTX, attention2=True,
                            norm=True).to(DEVICE)
    m.load_state_dict(model.state_dict())
    opt = torch.optim.Adam(m.parameters(), lr=lr)
    g = torch.Generator().manual_seed(seed)
    seq = facts[k:k + 1].to(DEVICE)
    tgt = int(seq[0, 6])
    curve = []

    def marg():
        with torch.no_grad():
            lg = m(seq[:, :7])[0, 5]
        return float(lg[tgt] - lg.scatter(
            0, torch.tensor([tgt], device=DEVICE), float("-inf")).max())

    curve.append(round(marg(), 2))
    for step in range(1, steps + 1):
        opt.zero_grad(set_to_none=True)
        ti = torch.randint(len(text_train), (8,), generator=g)
        tb = text_train[ti].to(DEVICE)
        ce_f = F.cross_entropy(m(seq[:, :7])[0, :6], seq[0, 1:7])
        ce_t = F.cross_entropy(m(tb)[:, :-1].reshape(-1, VOCAB),
                               tb[:, 1:].reshape(-1))
        (ce_f + ce_t).backward()
        opt.step()
        if step % 10 == 0:
            curve.append(round(marg(), 2))
    cross0 = next((10 * i for i, v in enumerate(curve) if v > 0), None)
    cross2 = next((10 * i for i, v in enumerate(curve) if v > 2), None)
    return {"start": curve[0], "cross0": cross0, "cross2": cross2,
            "curve_every10": curve[:16]}


def main():
    tok = Tokenizer.from_file(t5.TOK_PATH)
    facts, grp, held, info = t5.build_data(tok)
    text_train, _ = t5.build_text(tok)
    orig = load("results/t5_model_ft.pt")
    edited = load("results/t5_model_edited.pt")
    wte = orig.wte.weight.detach().double().cpu()

    m0, sl0 = t5.fact_margins(orig, facts)
    correct = (sl0 == facts[:, 6]).numpy()
    stored_mem = np.where((grp == "mem") & correct)[0]
    order = stored_mem[np.argsort(m0.numpy()[stored_mem])]
    k = int(order[len(order) // 2])
    train_idx = np.array([i for i in stored_mem if i != k])

    for name, model in (("original", orig), ("post_retension", edited)):
        t5.log({"stage": "probe_v2", "model": name, "target": k,
                "layers": probe_embed(model, facts, train_idx, k,
                                      info["values"], wte)})

    unstored = np.where((grp == "mem") & ~correct)[0]
    g = torch.Generator().manual_seed(9)
    fresh = unstored[torch.randperm(len(unstored), generator=g)[:3].numpy()]
    t5.log({"stage": "relearn_v2", "target": k,
            "target_on_edited": relearn_curve(edited, facts, k, text_train),
            "fresh_on_original": [
                relearn_curve(orig, facts, int(f), text_train)
                for f in fresh],
            "fresh_on_edited": [
                relearn_curve(edited, facts, int(f), text_train)
                for f in fresh]})


if __name__ == "__main__":
    main()
