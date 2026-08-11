"""T3c: weights-only neighbor finding — can we predict WHICH facts a
deletion will break, from the weights of the edit alone (no labels, no
margins), by folding the function change back to input space?

For a delete edit L -> L', the per-label function change is
Delta M_c = -(sym(l'_c l'_c^T) - sym(l_c l_c^T)) — an input-space operator
computable from weights alone. Its top right-singular directions span the
inputs the edit can affect. Rank all datapoints by projection onto that
subspace; measure retrieval of the actual delete-only collateral set (AUC,
precision@|C|), vs the data-side baseline |x_i . x_k|.
Appends to results/t3c_neighbors.jsonl.
"""

import json

import numpy as np
import torch

import p3_corr_funcdist as p3
from t3b_dense_retension import margins_w, proximal_delete

OUT = "results/t3c_neighbors.jsonl"
p3.N = 768


def edit_subspace(L, L2, rank=6):
    stack = []
    for c in range(p3.M):
        M0 = -torch.outer(L[c], L[c])
        M1 = -torch.outer(L2[c], L2[c])
        dM = (M1 + M1.T) / 2 - (M0 + M0.T) / 2
        stack.append(dM)
    S = torch.cat(stack, dim=0)  # (C*dim, dim)
    _, _, Vh = torch.linalg.svd(S, full_matrices=False)
    return Vh[:rank]  # (rank, dim)


def auc(scores, positives):
    order = np.argsort(-scores)
    ranks = np.empty(len(scores))
    ranks[order] = np.arange(len(scores))
    pos, neg = ranks[positives], ranks[~positives]
    wins = (pos[:, None] < neg[None, :]).mean() if len(pos) and len(neg) else None
    return None if wins is None else round(float(wins), 3)


def main():
    x, y, _ = p3.data(0.5)
    ws = p3.train(x, y, sym=True, epochs=40000)
    L = ws[0]
    labels = ((x @ L.T) ** 2 @ (-torch.eye(p3.C, dtype=L.dtype))).argmax(1)
    pre_marg = margins_w(L, x, labels)
    stored = pre_marg > 0

    sidx = torch.where(stored)[0]
    order = sidx[torch.argsort(pre_marg[sidx])]
    picks = [int(order[len(order) // 2 + j * 30]) for j in range(-2, 3)]

    for k in picks:
        res = proximal_delete(L, x, labels, stored, k)
        if res is None:
            continue
        _, _, L2 = res
        m2 = margins_w(L2, x, labels)
        others = stored.clone()
        others[k] = False
        broke = (others & (m2 <= 0)).numpy()
        n_broke = int(broke.sum())
        if n_broke == 0:
            print(json.dumps({"target": int(k), "n_broke": 0}))
            continue

        mask = np.ones(p3.N, dtype=bool)
        mask[k] = False
        V = edit_subspace(L, L2)
        s_weights = (x @ V.T).norm(dim=1).numpy()
        s_overlap = (x @ x[k]).abs().numpy()

        topn = np.argsort(-s_weights[mask])[:n_broke]
        prec_w = float(broke[mask][topn].mean())
        topn_o = np.argsort(-s_overlap[mask])[:n_broke]
        prec_o = float(broke[mask][topn_o].mean())
        rec = {"target": int(k), "n_broke": n_broke,
               "auc_weights_subspace": auc(s_weights[mask], broke[mask]),
               "auc_input_overlap": auc(s_overlap[mask], broke[mask]),
               "prec_at_n_weights": round(prec_w, 3),
               "prec_at_n_overlap": round(prec_o, 3)}
        with open(OUT, "a") as f:
            f.write(json.dumps(rec) + "\n")
        print(json.dumps(rec), flush=True)


if __name__ == "__main__":
    main()
