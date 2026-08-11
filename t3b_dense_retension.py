"""T3b: does delete-then-retension break the deletion-only collateral floor
in the DENSE regime? (P3-D, crowded n=768: oracle deletion ~23 collateral,
proximal ~25, random ~258.)

Two-stage edit: (1) proximal rank-1 delete of target fact, (2) retension =
hinge repair of bystander margins (self-labeled: labels read from the
model's own argmax, inputs cached — dense inputs are not enumerable from
weights, unlike the token toy) with the target pinned negative.
Appends to results/t3b_dense_retension.jsonl.
"""

import json

import numpy as np
import torch
import torch.nn.functional as F

import p3_corr_funcdist as p3
from capacity import DEVICE

OUT = "results/t3b_dense_retension.jsonl"
p3.N = 768


def margins_w(L, x, labels):
    h = (x @ L.T) ** 2
    logits = h @ (-torch.eye(p3.C, dtype=L.dtype, device=L.device))
    own = logits.gather(1, labels.unsqueeze(1)).squeeze(1)
    oth = logits.scatter(1, labels.unsqueeze(1), float("-inf")).max(1).values
    return own - oth


def proximal_delete(L, x, labels, stored, k, n_cand=3000, seed=0):
    xk = x[k]
    g = torch.Generator().manual_seed(seed)
    scale = float(L.norm())
    mags = np.geomspace(1e-3, 2.0, 30)
    best = None
    others = stored.clone()
    others[k] = False
    for it in range(n_cand):
        mag = float(mags[min(29, it * 30 // n_cand)]) * scale
        delta = torch.randn(p3.M, generator=g, dtype=L.dtype)
        delta = delta / delta.norm() * mag
        L2 = L + delta.unsqueeze(1) * xk.unsqueeze(0)
        if float(margins_w(L2, x[k:k + 1], labels[k:k + 1])) <= 0:
            coll = int((margins_w(L2, x, labels)[others] <= 0).sum())
            wd = float((L2 - L).norm())
            if best is None or (wd, coll) < (best[0], best[1]):
                best = (wd, coll, L2)
    return best


def retension(L0, x, labels, stored, k, pre_marg, steps=4000, lr=1e-2):
    L = L0.clone().requires_grad_(True)
    others = stored.clone()
    others[k] = False
    idx = torch.where(others)[0]
    xb, lb = x[idx], labels[idx]
    target_m = pre_marg[idx].clamp(max=float(pre_marg[idx].median()))
    opt = torch.optim.Adam([L], lr=lr)
    for _ in range(steps):
        opt.zero_grad(set_to_none=True)
        mb = margins_w(L, xb, lb)
        mk = margins_w(L, x[k:k + 1], labels[k:k + 1])
        loss = F.relu(target_m - mb).mean() + 10.0 * F.relu(mk + 0.05).sum()
        loss.backward()
        opt.step()
    return L.detach()


def main():
    x, y, mean_ov = p3.data(0.5)
    ws = p3.train(x, y, sym=True, epochs=40000)
    L = ws[0]
    labels = ((x @ L.T) ** 2 @ (-torch.eye(p3.C, dtype=L.dtype))).argmax(1)
    pre_marg = margins_w(L, x, labels)
    stored = pre_marg > 0
    acc_vs_y = float((labels == y).float().mean())
    n_stored = int(stored.sum())
    print(f"trained: {n_stored}/{p3.N} stored (argmax==y {acc_vs_y:.3f}, "
          f"overlap {mean_ov:.3f})", flush=True)

    sidx = torch.where(stored)[0]
    order = sidx[torch.argsort(pre_marg[sidx])]
    picks = [int(order[len(order) // 2 + j * 30]) for j in range(-2, 3)]

    for k in picks:
        res = proximal_delete(L, x, labels, stored, k)
        if res is None:
            print(json.dumps({"target": k, "note": "no feasible delete"}))
            continue
        wd_del, coll_del, L_del = res
        L_rep = retension(L_del, x, labels, stored, k, pre_marg)
        others = stored.clone()
        others[k] = False
        m_rep = margins_w(L_rep, x, labels)
        coll_rep = int((m_rep[others] <= 0).sum())
        mk = float(m_rep[k])
        rec = {"target": int(k), "pre_margin": round(float(pre_marg[k]), 4),
               "n_stored": n_stored,
               "collateral_delete_only": coll_del,
               "collateral_after_retension": coll_rep,
               "target_margin_after_retension": round(mk, 4),
               "target_stays_forgotten": mk <= 0,
               "wd_delete": round(wd_del, 3),
               "wd_total": round(float((L_rep - L).norm()), 3)}
        with open(OUT, "a") as f:
            f.write(json.dumps(rec) + "\n")
        print(json.dumps(rec), flush=True)


if __name__ == "__main__":
    main()
