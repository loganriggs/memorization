"""H1b: iterative reweighted Rayleigh construction.

Round t: per-fact weights w_i (init 1). For each label c:
  A_c = sum_{i: y_i=c} w_i x_i x_i^T / sum w,  B_c = sum_{i: y_i!=c} w_i x_i
  x_i^T / sum w + eps*I; v_c = top generalized eigenvector, v^T B v = 1.
Evaluate argmax_c (v_c.x)^2; multiply w_i by BETA for every misclassified
fact; repeat. Keep the best round. Eigen-solving + reweighting only — no GD.

Usage: python h1b_reweighted.py [--dvals ...] [--rounds 40] [--beta 2.0]
"""

import argparse

import torch
import torch.nn.functional as F

from capacity import generate_facts

EPS = 1e-3


def solve_L(X, targets, wts, v_out):
    dim = X.shape[1]
    L = torch.zeros(v_out, dim, dtype=torch.float64)
    for c in range(v_out):
        mp = targets == c
        mn = ~mp
        wp = wts[mp].unsqueeze(1)
        wn = wts[mn].unsqueeze(1)
        P = X[mp].double()
        N = X[mn].double()
        A = (P * wp).T @ P / wp.sum()
        B = (N * wn).T @ N / wn.sum() + EPS * torch.eye(dim,
                                                        dtype=torch.float64)
        R = torch.linalg.cholesky(B, upper=True)
        Rinv = torch.linalg.inv(R)
        evals, evecs = torch.linalg.eigh(Rinv.T @ A @ Rinv)
        v = Rinv @ evecs[:, -1]
        L[c] = v / (v @ B @ v).sqrt()
    return L.float()


def run(d, n_facts, rounds, beta, w_cap=1e6):
    v_in, v_out = 2 * d, d
    inputs, targets = generate_facts(n_facts, v_in, v_out)
    inputs, targets = inputs.cpu(), targets.cpu()
    X = torch.cat([F.one_hot(inputs[:, 0], v_in).float(),
                   F.one_hot(inputs[:, 1], v_in).float()], dim=-1)
    wts = torch.ones(n_facts, dtype=torch.float64)
    best_acc, best_L, hist = 0.0, None, []
    for t in range(rounds):
        L = solve_L(X, targets, wts, v_out)
        pred = ((X @ L.T) ** 2).argmax(-1)
        acc = (pred == targets).float().mean().item()
        hist.append(acc)
        if acc > best_acc:
            best_acc, best_L = acc, L.clone()
        wrong = pred != targets
        wts[wrong] = (wts[wrong] * beta).clamp(max=w_cap)
        wts = wts / wts.mean()
    return best_acc, best_L, hist


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dvals", default="2,3,4,5,6,7,8")
    p.add_argument("--rounds", type=int, default=40)
    p.add_argument("--beta", type=float, default=2.0)
    p.add_argument("--nfrac", type=float, default=1.0)
    args = p.parse_args()
    print(f"{'d':>3} {'n':>6} {'round0':>7} {'best':>7} {'last':>7}")
    for d in [int(x) for x in args.dvals.split(",")]:
        n = max(1, int(round(args.nfrac * 4 * d * d)))
        best, L, hist = run(d, n, args.rounds, args.beta)
        print(f"{d:>3} {n:>6} {hist[0]:>7.3f} {best:>7.3f} {hist[-1]:>7.3f}",
              flush=True)


if __name__ == "__main__":
    main()
