"""Exact-interval fact insertion (user's interference idea, v1).

Along any update direction g, every fact j's rivalry condition
q_{j,c}(t) = h_c(x_j; L+tg) - h_y(x_j; L+tg) is an exact quadratic in t
(difference of squares of linear functions). The roots of all affected
facts' quadratics partition the t-axis into segments on which the global
correct-count is constant; we evaluate one candidate t per segment
boundary/midpoint and take the best. No gradient descent, no blind line
search: candidates come from closed-form roots.

Direction candidates for inserting fact i = (t1, t2) -> y (all supported
on columns t1 and V+t2 only, so only token-sharing facts are affected):
  1. negated CE gradient (rows ~ alpha_i on both columns)
  2. own-row silence move (drive v_y . x_i toward 0)
  3. strongest-rival loudening (grow v_c* . x_i)

Usage: python interval_insert.py [--dvals 4,8] [--stage 1|2]
"""

import argparse

import torch
import torch.nn.functional as F

from capacity import generate_facts
from h12b_repair import h9b_solve


def margins(L, X, targets):
    h = (X @ L.T) ** 2
    own = h.gather(1, targets.unsqueeze(1)).squeeze(1)
    oth = h.scatter(1, targets.unsqueeze(1), float("inf")).min(dim=1).values
    return oth - own


def candidate_dirs(L, X, targets, i, inputs, v_in):
    x = X[i]
    pre = L @ x                                  # (m,)
    y = int(targets[i])
    m = L.shape[0]
    dirs = []
    # 1. negated CE gradient
    z = -(pre ** 2)
    p = F.softmax(z, dim=0)
    onehot = torch.zeros(m, dtype=L.dtype)
    onehot[y] = 1.0
    alpha = 2 * (p - onehot) * pre               # descent = -grad -> +alpha
    g = torch.zeros_like(L)
    g[:, inputs[i, 0]] += alpha
    g[:, v_in + inputs[i, 1]] += alpha
    if g.norm() > 1e-9:
        dirs.append(g / g.norm())
    # 2. own-row silence
    g = torch.zeros_like(L)
    g[y, inputs[i, 0]] = -pre[y] / 2
    g[y, v_in + inputs[i, 1]] = -pre[y] / 2
    if g.norm() > 1e-9:
        dirs.append(g / g.norm())
    # 3. strongest-rival loudening (quietest rival)
    h = pre ** 2
    h[y] = float("inf")
    cstar = int(h.argmin())
    g = torch.zeros_like(L)
    s = 1.0 if pre[cstar] >= 0 else -1.0
    g[cstar, inputs[i, 0]] = s
    g[cstar, v_in + inputs[i, 1]] = s
    dirs.append(g / g.norm())
    # 4. v3 category-B move: un-quiet the rival on x_i with least-squares-
    #    minimal disturbance of the rival's own facts:
    #    w = (X_own(c*)^T X_own(c*) + eps I)^-1 x_i, applied to row c* only.
    own_mask = targets == cstar
    if own_mask.any():
        Xo = X[own_mask]
        G2 = Xo.T @ Xo + 1e-3 * torch.eye(X.shape[1], dtype=L.dtype)
        wdir = torch.linalg.solve(G2, x)
        g = torch.zeros_like(L)
        g[cstar] = wdir
        if g.norm() > 1e-9:
            dirs.append(g / g.norm())
    return dirs


def root_candidates(L, g, X, targets, affected, t_max=4.0):
    """Roots of all rivalry quadratics for affected facts -> candidate ts."""
    ts = set()
    m = L.shape[0]
    for j in affected:
        x = X[j]
        a = (L @ x).tolist()   # pre at t=0 (python floats: avoids a torch
        b = (g @ x).tolist()   # scalar-op segfault in hot loops, and is faster)
        y = int(targets[j])
        for c in range(m):
            if c == y:
                continue
            # q(t) = (a_c + t b_c)^2 - (a_y + t b_y)^2
            A = b[c] * b[c] - b[y] * b[y]
            B = 2 * (a[c] * b[c] - a[y] * b[y])
            C = a[c] * a[c] - a[y] * a[y]
            if abs(A) < 1e-12:
                if abs(B) > 1e-12:
                    ts.add(-C / B)
                continue
            disc = B * B - 4 * A * C
            if disc >= 0:
                r = disc ** 0.5
                ts.add((-B + r) / (2 * A))
                ts.add((-B - r) / (2 * A))
    ts = sorted(t for t in ts if -t_max <= t <= t_max)
    cands = []
    prev = -t_max
    for t in ts + [t_max]:
        cands.append((prev + t) / 2)             # segment midpoints
        prev = t
    cands += [t + 1e-6 for t in ts] + [t - 1e-6 for t in ts]
    return [t for t in cands if t != 0.0]


def insert_loop(L, X, targets, inputs, v_in, max_sweeps=12, verbose=True):
    L = L.clone()
    n = X.shape[0]
    obstructions = {}
    for sweep in range(max_sweeps):
        marg = margins(L, X, targets)
        count = int((marg > 0).sum())
        improved = False
        order = torch.argsort(marg)              # most-negative first? closest?
        for i in [int(j) for j in order if marg[j] <= 0]:
            share = ((inputs[:, 0] == inputs[i, 0])
                     | (inputs[:, 1] == inputs[i, 1])).nonzero().flatten()
            best = (count, None, None)
            for g in candidate_dirs(L, X, targets, i, inputs, v_in):
                for t in root_candidates(L, g, X, targets, share.tolist()):
                    cand = L + t * g
                    c = int((margins(cand, X, targets) > 0).sum())
                    if c > best[0]:
                        best = (c, g, t)
            if best[1] is not None:
                L = L + best[2] * best[1]
                count = best[0]
                improved = True
                obstructions.pop(i, None)
            else:
                # record which facts block every candidate (structural info)
                obstructions[i] = len(share)
        acc = count / n
        if verbose:
            print(f"  sweep {sweep}: acc {acc:.3f} "
                  f"({len(obstructions)} obstructed)", flush=True)
        if not improved:
            break
    return count / n, L, obstructions


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dvals", default="4,8")
    p.add_argument("--stage", type=int, default=1)
    args = p.parse_args()
    torch.set_num_threads(4)
    for d in [int(x) for x in args.dvals.split(",")]:
        v_in, v_out, n = 2 * d, d, 4 * d * d
        inputs, targets = generate_facts(n, v_in, v_out)
        inputs, targets = inputs.cpu(), targets.cpu()
        X = torch.cat([F.one_hot(inputs[:, 0], v_in).float(),
                       F.one_hot(inputs[:, 1], v_in).float()],
                      dim=-1).double()
        rounds = 1 if args.stage == 1 else 40
        _, L = h9b_solve(X, targets, v_out, rounds=rounds)
        acc0 = float((margins(L, X, targets) > 0).float().mean())
        print(f"d={d}: start (stage {args.stage}) acc {acc0:.3f}")
        acc, L2, obs = insert_loop(L, X, targets, inputs, v_in)
        print(f"d={d}: FINAL acc {acc:.3f}, obstructed facts: "
              f"{sorted(obs.keys())}")


if __name__ == "__main__":
    main()
