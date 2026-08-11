"""H12b: hinge-margin greedy repair on top of the H9b silence construction.

Silence code: logits = -(v_c . x)^2, margin_i = min_{c != y} h_c - h_y.
Score = sum_i min(margin_i, tau) — a hinge that rewards fixing near-misses
without over-rewarding already-safe facts. Moves: single-entry grid scans
plus paired moves (a[t1] += delta, b[t2] -= delta along own-edges, which
preserve that fact's response while shifting others). Restarts from noisy
H9b. Greedy accept-if-better only — no gradients.

Usage: python h12b_repair.py [--dvals 3,4,5,6,8] [--restarts 3]
"""

import argparse

import torch
import torch.nn.functional as F

from capacity import generate_facts

EPS = 1e-3
TAU = 0.3


def h9b_solve(X, targets, v_out, rounds=40):
    dim = X.shape[1]
    wts = torch.ones(len(targets), dtype=torch.float64)
    best, bestL = 0.0, None
    for _ in range(rounds):
        L = torch.zeros(v_out, dim, dtype=torch.float64)
        for c in range(v_out):
            mp, mn = targets == c, targets != c
            P, N = X[mp], X[mn]
            wp, wn = wts[mp].unsqueeze(1), wts[mn].unsqueeze(1)
            A = (P * wp).T @ P / wp.sum()
            B = (N * wn).T @ N / wn.sum() + EPS * torch.eye(
                dim, dtype=torch.float64)
            R = torch.linalg.cholesky(B, upper=True)
            Ri = torch.linalg.inv(R)
            _, V = torch.linalg.eigh(Ri.T @ A @ Ri)
            v = Ri @ V[:, 0]
            L[c] = v / (v @ B @ v).sqrt()
        pred = (-(X @ L.T) ** 2).argmax(-1)
        acc = (pred == targets).float().mean().item()
        if acc > best:
            best, bestL = acc, L.clone()
        wrong = pred != targets
        wts[wrong] = (wts[wrong] * 2.0).clamp(max=1e6)
        wts = wts / wts.mean()
    return best, bestL


def hinge_score(L, X, targets, v_out):
    h = (X @ L.T) ** 2
    own = h.gather(1, targets.unsqueeze(1)).squeeze(1)
    other = h.clone()
    other.scatter_(1, targets.unsqueeze(1), float("inf"))
    margin = other.min(dim=1).values - own
    return margin.clamp(max=TAU).sum().item(), \
        int((margin > 0).sum())


def repair(L, X, targets, v_out, inputs, v_in, passes=8, n_cand=17,
           gen=None):
    L = L.clone()
    n, dim = X.shape
    score, ncorr = hinge_score(L, X, targets, v_out)
    own_edges = [(int(t[0]), int(t[1]), int(y))
                 for t, y in zip(inputs, targets)]
    for p in range(passes):
        improved = False
        # single-entry moves
        for c in range(v_out):
            row_scale = float(L[c].abs().max())
            for j in range(dim):
                base = L[c, j].item()
                width = max(row_scale, 0.3)
                best_v, best_s, best_n = base, score, ncorr
                for dv in torch.linspace(-width, width, n_cand).tolist():
                    L[c, j] = base + dv
                    s, nc = hinge_score(L, X, targets, v_out)
                    if (nc, s) > (best_n, best_s):
                        best_v, best_s, best_n = base + dv, s, nc
                L[c, j] = best_v
                if (best_n, best_s) > (ncorr, score):
                    score, ncorr = best_s, best_n
                    improved = True
        # paired moves along own-edges (keep that fact's response fixed)
        for (t1, t2, y) in own_edges:
            j, k = t1, v_in + t2
            base_j, base_k = L[y, j].item(), L[y, k].item()
            width = max(float(L[y].abs().max()), 0.3)
            best = (base_j, base_k, score, ncorr)
            for dv in torch.linspace(-width, width, n_cand).tolist():
                L[y, j] = base_j + dv
                L[y, k] = base_k - dv
                s, nc = hinge_score(L, X, targets, v_out)
                if (nc, s) > (best[3], best[2]):
                    best = (base_j + dv, base_k - dv, s, nc)
            L[y, j], L[y, k] = best[0], best[1]
            if (best[3], best[2]) > (ncorr, score):
                score, ncorr = best[2], best[3]
                improved = True
        if not improved:
            break
    return ncorr / n, L


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dvals", default="3,4,5,6,8")
    p.add_argument("--restarts", type=int, default=3)
    args = p.parse_args()
    gen = torch.Generator().manual_seed(0)
    print(f"{'d':>3} {'H9b':>6} {'H12b best':>10}")
    for d in [int(x) for x in args.dvals.split(",")]:
        v_in, v_out, n = 2 * d, d, 4 * d * d
        inputs, targets = generate_facts(n, v_in, v_out)
        inputs, targets = inputs.cpu(), targets.cpu()
        X = torch.cat([F.one_hot(inputs[:, 0], v_in).float(),
                       F.one_hot(inputs[:, 1], v_in).float()],
                      dim=-1).double()
        acc0, L0 = h9b_solve(X, targets, v_out)
        best = acc0
        for r in range(args.restarts):
            Lr = L0 if r == 0 else L0 + 0.1 * torch.randn(
                L0.shape, generator=gen, dtype=torch.float64)
            acc, _ = repair(Lr, X, targets, v_out, inputs, v_in)
            best = max(best, acc)
        print(f"{d:>3} {acc0:>6.3f} {best:>10.3f}", flush=True)


if __name__ == "__main__":
    main()
