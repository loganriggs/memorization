"""H1: Rayleigh-quotient construction for the symmetric bilinear memorizer.

For each label c, build neuron v_c = top generalized eigenvector of
(A_c, B_c) where A_c = mean_{facts with label c} x x^T and
B_c = mean_{other facts} x x^T + eps*I, with x = [onehot(t1); onehot(t2)].
Normalize v_c^T B_c v_c = 1 (so mean off-label squared response = 1 for every
label) and set D = I. Prediction: argmax_c (v_c . x)^2.

No gradient descent anywhere. Usage: python rayleigh.py [--dvals 2..8]
[--nfrac 1.0] (fraction of the 4d^2 ceiling to store).
"""

import argparse

import torch
import torch.nn.functional as F

from capacity import generate_facts

EPS = 1e-3


def build(d, n_facts):
    v_in, v_out = 2 * d, d
    inputs, targets = generate_facts(n_facts, v_in, v_out)
    inputs, targets = inputs.cpu(), targets.cpu()
    X = torch.cat([F.one_hot(inputs[:, 0], v_in).float(),
                   F.one_hot(inputs[:, 1], v_in).float()], dim=-1)  # (n, 4d)
    dim = X.shape[1]
    L = torch.zeros(v_out, dim, dtype=torch.float64)
    Xd = X.double()
    for c in range(v_out):
        pos = Xd[targets == c]
        neg = Xd[targets != c]
        A = pos.T @ pos / len(pos)
        B = neg.T @ neg / len(neg) + EPS * torch.eye(dim, dtype=torch.float64)
        # generalized eig via Cholesky whitening: B = R^T R
        R = torch.linalg.cholesky(B, upper=True)
        Rinv = torch.linalg.inv(R)
        M = Rinv.T @ A @ Rinv
        evals, evecs = torch.linalg.eigh(M)
        v = Rinv @ evecs[:, -1]
        v = v / (v @ B @ v).sqrt()      # v^T B v = 1
        L[c] = v
    return L.float(), (inputs, targets, X)


def evaluate(L, data):
    inputs, targets, X = data
    h = (X @ L.T) ** 2                  # (n, V_out); D = I
    acc = (h.argmax(-1) == targets).float().mean().item()
    return acc


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dvals", default="2,3,4,5,6,7,8")
    p.add_argument("--nfrac", type=float, default=1.0)
    args = p.parse_args()
    print(f"{'d':>3} {'n':>6} {'acc':>7}")
    for d in [int(x) for x in args.dvals.split(",")]:
        n = max(1, int(round(args.nfrac * 4 * d * d)))
        L, data = build(d, n)
        acc = evaluate(L, data)
        print(f"{d:>3} {n:>6} {acc:>7.3f}", flush=True)


if __name__ == "__main__":
    main()
