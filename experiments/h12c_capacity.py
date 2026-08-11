"""Task 19 (fast engine): capacity of the H12b/H12c construction.

Binary search max n at acc>=0.9 using h9b spectral solve + fast_repair,
6 restarts. Appends to results/handcoded_h12c.jsonl.
"""

import json
import os

import torch
import torch.nn.functional as F

from capacity import generate_facts, PRECISION_FRACTION, RESULTS_DIR
from h12b_repair import h9b_solve
from h12c_fast import fast_repair

torch.set_num_threads(6)
RESULTS_PATH = os.path.join(RESULTS_DIR, "handcoded_h12c.jsonl")
GEN = torch.Generator().manual_seed(0)


def acc_at(d, n, restarts=6):
    v_in, v_out = 2 * d, d
    inputs, targets = generate_facts(n, v_in, v_out)
    inputs, targets = inputs.cpu(), targets.cpu()
    X = torch.cat([F.one_hot(inputs[:, 0], v_in).float(),
                   F.one_hot(inputs[:, 1], v_in).float()], dim=-1).double()
    acc0, L0 = h9b_solve(X, targets, v_out)
    D = -torch.eye(v_out, dtype=torch.float64)
    best = acc0
    for r in range(restarts):
        Lr = L0 if r == 0 else L0 + 0.15 * torch.randn(
            L0.shape, generator=GEN, dtype=torch.float64)
        acc, _ = fast_repair(Lr, D, X, targets, inputs, v_in,
                             passes=12, n_cand=25)
        best = max(best, acc)
        if best >= 0.9:
            break
    return best


def main():
    for d in (6, 8, 12, 16):
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
        if hi == max_possible and max_possible not in cache:
            s = acc_at(d, max_possible)
            if s >= 0.9:
                best, best_score = max_possible, s
        rec = {"arch": "handcoded_h12c", "d": d, "threshold": 0.9,
               "max_facts": best, "best_score": best_score,
               "ceiling": max_possible}
        with open(RESULTS_PATH, "a") as f:
            f.write(json.dumps(rec) + "\n")
        print(f"RESULT {rec}", flush=True)


if __name__ == "__main__":
    main()
