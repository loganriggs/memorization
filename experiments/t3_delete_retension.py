"""T3: weights-only delete-then-retension.

Claim under test: in the token toy the ENTIRE unlearning loop can run from
weights alone — (1) read the stored-fact ledger out of the weight tensor
(enumerate all (2d)^2 pairs, self-label = argmax, stored = confident
margin), (2) delete a target fact via proximal rank-1 edit, (3) RETENSION:
restore the broken bystanders' margins (hinge repair on the self-labeled
ledger) while pinning the target's margin negative. No training data used
anywhere after step 0.

Reports collateral after deletion-only vs after delete+retension, whether
the target stays forgotten, and weight-change costs.
Appends to results/t3_delete_retension.jsonl.
"""

import json

import numpy as np
import torch
import torch.nn.functional as F

from analysis import train_single
from capacity import DEVICE, _forward, width_for

OUT = "results/t3_delete_retension.jsonl"
D = 8
N_FACTS = 200


def all_pair_inputs(d):
    v_in = 2 * d
    pairs = torch.cartesian_prod(torch.arange(v_in), torch.arange(v_in))
    x = torch.cat([F.one_hot(pairs[:, 0], v_in).float(),
                   F.one_hot(pairs[:, 1], v_in).float()], dim=-1)
    return pairs, x.to(DEVICE)


def logits_of(weights, x):
    ws = {k: v.unsqueeze(0) for k, v in weights.items()}
    return _forward("bilinear", ws, x)[0]


def margins_vs(weights, x, labels):
    lg = logits_of(weights, x)
    own = lg.gather(1, labels.unsqueeze(1)).squeeze(1)
    oth = lg.scatter(1, labels.unsqueeze(1), float("-inf")).max(1).values
    return own - oth


def ledger_from_weights(weights, x):
    """Self-labels + confident stored set, from weights alone.
    Threshold = largest relative gap in the sorted positive margins."""
    lg = logits_of(weights, x)
    labels = lg.argmax(1)
    marg = margins_vs(weights, x, labels)
    s, _ = marg.sort()
    logs = s.clamp_min(1e-6).log()
    gaps = logs[1:] - logs[:-1]
    cut = int(gaps.argmax()) + 1
    tau = float(s[cut - 1] * (s[cut] / s[cut - 1]).sqrt()) if cut < len(s) else 0.0
    stored = marg > tau
    return labels.detach(), marg.detach(), stored, tau


def proximal_delete(weights, x, labels, stored, k, n_cand=3000, seed=0):
    """Rank-1 edits on L along x_k; feasible = target margin <= 0;
    select min ||delta L|| among feasible."""
    L = weights["L"]
    m = L.shape[0]
    xk = x[k]
    g = torch.Generator().manual_seed(seed)
    scale = float(L.norm())
    best = None
    mags = np.geomspace(1e-3, 1.0, 30)
    for it in range(n_cand):
        mag = float(mags[min(29, it * 30 // n_cand)]) * scale
        delta = torch.randn(m, generator=g).to(DEVICE)
        delta = delta / delta.norm() * mag
        w2 = dict(weights)
        w2["L"] = L + delta.unsqueeze(1) * xk.unsqueeze(0)
        mk = margins_vs(w2, x[k:k + 1], labels[k:k + 1])
        if float(mk) <= 0:
            others = stored.clone()
            others[k] = False
            coll = int((margins_vs(w2, x, labels)[others] <= 0).sum())
            wd = float(delta.norm())
            if best is None or (wd, coll) < (best[0], best[1]):
                best = (wd, coll, w2)
    return best


def retension(weights, x, labels, stored, k, pre_marg, steps=2000, lr=3e-3):
    """Weights-only repair: raise bystander margins back toward their
    pre-edit values (capped), pin target margin <= -eps."""
    w = {kk: v.clone().requires_grad_(True) for kk, v in weights.items()}
    others = stored.clone()
    others[k] = False
    idx = torch.where(others)[0]
    xb, lb = x[idx], labels[idx]
    target_m = pre_marg[idx].clamp(max=float(pre_marg[idx].median()))
    opt = torch.optim.Adam(list(w.values()), lr=lr)
    for step in range(steps):
        opt.zero_grad(set_to_none=True)
        mb = margins_vs(w, xb, lb)
        mk = margins_vs(w, x[k:k + 1], labels[k:k + 1])
        loss = F.relu(target_m - mb).mean() + 10.0 * F.relu(mk + 0.1).sum()
        loss.backward()
        opt.step()
    return {kk: v.detach() for kk, v in w.items()}


def main():
    m_width = width_for("bilinear", D)
    state, acc, _ = train_single("bilinear", D, m_width, N_FACTS)
    weights = {k: v.to(DEVICE).double() for k, v in state.items()}
    pairs, x = all_pair_inputs(D)
    x = x.double()

    labels, pre_marg, stored, tau = ledger_from_weights(weights, x)
    n_stored = int(stored.sum())
    print(f"trained acc {acc:.3f}; weights-only ledger: {n_stored} stored "
          f"(tau={tau:.3f}) of {len(pairs)} pairs", flush=True)

    g = torch.Generator().manual_seed(7)
    sidx = torch.where(stored)[0]
    order = sidx[torch.argsort(pre_marg[sidx])]
    picks = [int(order[len(order) // 2 + j * 8]) for j in range(-2, 3)]

    for k in picks:
        res = proximal_delete(weights, x, labels, stored, k)
        if res is None:
            print(json.dumps({"target": k, "note": "no feasible delete"}))
            continue
        wd_del, coll_del, w_del = res
        w_rep = retension(w_del, x, labels, stored, k, pre_marg)

        others = stored.clone()
        others[k] = False
        m_rep = margins_vs(w_rep, x, labels)
        coll_rep = int((m_rep[others] <= 0).sum())
        mk_rep = float(m_rep[k])
        wd_total = float(sum((w_rep[kk] - weights[kk]).norm() ** 2
                             for kk in weights) ** 0.5)
        rec = {"target": int(k), "pre_margin": round(float(pre_marg[k]), 3),
               "n_stored": n_stored,
               "collateral_delete_only": coll_del,
               "collateral_after_retension": coll_rep,
               "target_margin_after_retension": round(mk_rep, 3),
               "target_stays_forgotten": mk_rep <= 0,
               "wd_delete": round(wd_del, 3),
               "wd_total": round(wd_total, 3)}
        with open(OUT, "a") as f:
            f.write(json.dumps(rec) + "\n")
        print(json.dumps(rec), flush=True)


if __name__ == "__main__":
    main()
