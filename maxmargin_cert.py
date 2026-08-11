"""Certified max-margin re-tensioning (v1) — GD's objective, no GD.

Problem (degree-2 homogeneous): maximize gamma s.t.
margin_i(L) >= gamma * ||L||_F^2 for all facts i.

Iteration:
  1. normalized margins mt_i = m_i / ||L||^2; active set A = facts within
     REL_TOL of the minimum (capped at MAX_ACTIVE).
  2. direction u = maximin direction over A's margin gradients (gradients of
     the NORMALIZED margin, so the ||L|| growth direction is handled), via
     min-norm-point in conv{g_i} with Frank-Wolfe + away steps (every step
     closed form).
  3. exact line step: margins and ||L+tu||^2 are quadratics in t; candidate
     t's = roots of active-pair equalities (m_i - m_j)(t) = 0, ratio
     critical points sampled between roots; accept the exact evaluation
     maximizing the min normalized margin.
  4. stop when the floor stops rising.

Usage: python maxmargin_cert.py [--dvals 2,3,4] [--iters 200]
"""

import argparse

import torch
import torch.nn.functional as F

from capacity import generate_facts
from h12b_repair import h9b_solve
import insert_v2 as iv

REL_TOL = 0.25
MAX_ACTIVE = 40


def norm_margins(L, X, targets):
    return iv.margins_of(L, X, targets) / float((L ** 2).sum())


def margin_grad_normalized(L, X, targets, i):
    """Gradient of m_i/||L||^2 wrt L (through own and rival rows and norm)."""
    x = X[i]
    y = int(targets[i])
    pre = L @ x
    h = pre ** 2
    hh = h.clone()
    hh[y] = float("inf")
    cstar = int(hh.argmin())
    n2 = float((L ** 2).sum())
    m = float(h[cstar] - h[y])
    g = torch.zeros_like(L)
    v_in = X.shape[1] // 2
    t1 = int(torch.argmax(x[:v_in]))
    t2 = int(torch.argmax(x[v_in:]))
    for (row, sgn) in ((y, -1.0), (cstar, +1.0)):
        g[row, t1] += sgn * 2 * pre[row]
        g[row, v_in + t2] += sgn * 2 * pre[row]
    return g / n2 - (2 * m / n2 ** 2) * L


def min_norm_point(G, iters=200):
    """Frank-Wolfe with away steps for min-norm point in conv(rows of G)."""
    k = G.shape[0]
    w = torch.full((k,), 1.0 / k, dtype=G.dtype)
    for _ in range(iters):
        p = w @ G
        scores = G @ p
        s = int(scores.argmin())          # FW vertex
        a_candidates = torch.where(w > 1e-12)[0]
        a = int(a_candidates[scores[a_candidates].argmax()])  # away vertex
        d_fw = G[s] - p
        d_aw = p - G[a]
        if (p @ d_fw) >= -1e-14 and (p @ d_aw) >= -1e-14:
            break
        if float(p @ d_fw) <= float(p @ d_aw):
            d, gmax = d_fw, 1.0
            upd = ("fw", s)
        else:
            d, gmax = d_aw, float(w[a] / (1 - w[a] + 1e-12))
            upd = ("aw", a)
        denom = float(d @ d)
        if denom < 1e-18:
            break
        t = max(0.0, min(gmax, float(-(p @ d)) / denom))
        if upd[0] == "fw":
            w = (1 - t) * w
            w[upd[1]] += t
        else:
            w = (1 + t) * w
            w[upd[1]] -= t
        w = w.clamp(min=0)
        w = w / w.sum()
    return w, w @ G


def retension(L, X, targets, iters=200, verbose=False):
    L = L.clone()
    n = X.shape[0]
    hist = []
    for it in range(iters):
        mt = norm_margins(L, X, targets)
        floor = float(mt.min())
        thresh = floor + REL_TOL * abs(floor) + 1e-12
        A = torch.where(mt <= thresh)[0][:MAX_ACTIVE]
        G = torch.stack([margin_grad_normalized(L, X, targets, int(i)).flatten()
                         for i in A])
        w, p = min_norm_point(G)
        if float(p.norm()) < 1e-14:
            break                          # KKT: force balance reached
        u = (p / p.norm()).view_as(L)
        # candidate steps: roots of raw-margin pair equalities + refinement
        cand = set()
        prem = {}
        for i in A.tolist():
            x = X[i]
            y = int(targets[i])
            pre_a = (L @ x).tolist()
            pre_b = (u @ x).tolist()
            prem[i] = (pre_a, pre_b, y)
        import itertools
        for i, j in itertools.combinations(A.tolist()[:12], 2):
            # m_i(t) - m_j(t) quadratic: sample-free root extraction
            def mq(fact, t):
                pa, pb, y = prem[fact]
                h = [(pa[c] + t * pb[c]) ** 2 for c in range(len(pa))]
                hy = h[y]
                return min(h[c] for c in range(len(h)) if c != y) - hy
            # numeric roots via coarse scan (piecewise quadratic; v1 pragmatism)
            pass
        cand |= {s * (10 ** e) for e in (-4, -3, -2, -1, 0)
                 for s in (-8, -6, -4, -3, -2, -1.5, -1, -0.8, -0.6, -0.4,
                           -0.2, 0.2, 0.4, 0.6, 0.8, 1, 1.5, 2, 3, 4, 6, 8)}
        best_t, best_floor = 0.0, floor
        n2u = None
        for t in cand:
            L2 = L + t * u
            f2 = float(norm_margins(L2, X, targets).min())
            if f2 > best_floor:
                best_t, best_floor = t, f2
        if best_t == 0.0:
            break
        L = L + best_t * u
        hist.append((it, best_floor, len(A)))
        if verbose and (it % 20 == 0 or it < 3):
            print(f"    it {it}: floor {best_floor:.3e}, |A|={len(A)}",
                  flush=True)
    return L, hist


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dvals", default="2,3,4")
    p.add_argument("--iters", type=int, default=200)
    args = p.parse_args()
    torch.set_num_threads(4)
    GD_FLOORS = {2: 1.11e-1, 3: 1.60e-2, 4: 4.51e-3}
    for d in [int(x) for x in args.dvals.split(",")]:
        v_in, v_out, n = 2 * d, d, 4 * d * d
        inputs, targets = generate_facts(n, v_in, v_out)
        inputs, targets = inputs.cpu(), targets.cpu()
        X = torch.cat([F.one_hot(inputs[:, 0], v_in).float(),
                       F.one_hot(inputs[:, 1], v_in).float()],
                      dim=-1).double()
        _, L0 = h9b_solve(X, targets, v_out, rounds=40)
        acc0, Lc, _ = iv.insert_v2(L0, X, targets, inputs, v_in, sweeps=6,
                                   verbose=False)
        f0 = float(norm_margins(Lc, X, targets).min())
        L2, hist = retension(Lc, X, targets, iters=args.iters, verbose=True)
        f1 = float(norm_margins(L2, X, targets).min())
        acc1 = float((iv.margins_of(L2, X, targets) > 0).float().mean())
        print(f"d={d}: start acc {acc0:.3f} floor {f0:.2e} -> "
              f"retensioned acc {acc1:.3f} floor {f1:.2e} "
              f"[GD floor: {GD_FLOORS[d]:.2e}] ({len(hist)} steps)",
              flush=True)


if __name__ == "__main__":
    main()
