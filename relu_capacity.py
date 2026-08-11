"""Task 22: capacity (max facts, acc>=0.9) of the ReLU-MLP silence
construction at d=16 — head-to-head with their hand-coded MLP (80-92),
hybrid, and trained (784) on the same architecture.

Appends to results/handcoded_relu.jsonl.
"""

import json
import os

import torch
import torch.nn.functional as F

from capacity import generate_facts, PRECISION_FRACTION, RESULTS_DIR
from h12c_fast import fast_repair

torch.set_num_threads(6)
RESULTS_PATH = os.path.join(RESULTS_DIR, "handcoded_relu.jsonl")
GEN = torch.Generator().manual_seed(0)
S = 3


def assign(d, S, seed=0):
    g = torch.Generator().manual_seed(seed)
    A = torch.zeros(d, d, dtype=torch.float64)
    load = torch.zeros(d)
    for c in torch.randperm(d, generator=g).tolist():
        picks = torch.argsort(load + 0.01 * torch.rand(d, generator=g))[:S]
        A[c, picks] = 1.0
        load[picks] += 1
    return A


def acc_at(d, n, n_assign=3, n_restart=3):
    v_in, v_out = 2 * d, d
    inputs, targets = generate_facts(n, v_in, v_out)
    inputs, targets = inputs.cpu(), targets.cpu()
    X = torch.cat([F.one_hot(inputs[:, 0], v_in).float(),
                   F.one_hot(inputs[:, 1], v_in).float()], dim=-1).double()
    dim = X.shape[1]
    G = X.T @ X + 1e-3 * len(X) * torch.eye(dim, dtype=torch.float64)
    best = 0.0
    for a_seed in range(n_assign):
        A = assign(d, S, a_seed)
        D = -A
        uses = A.T
        W = torch.zeros(d, dim, dtype=torch.float64)
        for nn in range(d):
            t = torch.where(uses[nn][targets] > 0, -1.0, 1.0).double()
            W[nn] = torch.linalg.solve(G, X.T @ t)
        for r in range(n_restart):
            Wr = W if r == 0 else W + 0.1 * torch.randn(
                W.shape, generator=GEN, dtype=torch.float64)
            acc, _ = fast_repair(Wr, D, X, targets, inputs, v_in,
                                 passes=12, n_cand=25, act="relu")
            best = max(best, acc)
            if best >= 0.9:
                return best
    return best


def main():
    d = 16
    max_possible = 4 * d * d
    lo, hi, best, best_score = 1, max_possible, 0, None
    cache = {}
    while hi - lo >= PRECISION_FRACTION * hi:
        mid = (lo + hi) // 2
        if mid not in cache:
            cache[mid] = acc_at(d, mid)
            print(f"  d={d} n={mid}: acc {cache[mid]:.3f}", flush=True)
        if cache[mid] >= 0.9:
            best, best_score = mid, cache[mid]
            lo = mid + 1
        else:
            hi = mid - 1
    rec = {"arch": "handcoded_relu_silence", "d": d, "threshold": 0.9,
           "max_facts": best, "best_score": best_score, "S": S,
           "ceiling": max_possible}
    with open(RESULTS_PATH, "a") as f:
        f.write(json.dumps(rec) + "\n")
    print(f"RESULT {rec}", flush=True)


if __name__ == "__main__":
    main()
