"""Capacity of the H16 pipeline (null-structured init + hinge repair) at
d=8 and d=16. Noise/variance handling: 8 restarts per eval alternating
sigma in {0.02, 0.05}; boundary points (within 5% of the running frontier)
re-evaluated with 16 restarts. Appends to results/handcoded_h16.jsonl.
"""

import json
import os

import torch

from capacity import PRECISION_FRACTION, RESULTS_DIR
from h12c_fast import fast_repair
from h16 import h16_build

torch.set_num_threads(6)
RESULTS_PATH = os.path.join(RESULTS_DIR, "handcoded_h16.jsonl")
GEN = torch.Generator().manual_seed(0)
SIGMAS = (0.02, 0.05)


def acc_at(d, n, restarts=8):
    acc0, L0, (X, inputs, targets) = h16_build(d, n)
    D = -torch.eye(d, dtype=torch.float64)
    best = acc0
    for r in range(restarts):
        sigma = SIGMAS[r % len(SIGMAS)]
        Lr = L0 if r == 0 else L0 + sigma * torch.randn(
            L0.shape, generator=GEN, dtype=torch.float64)
        acc, _ = fast_repair(Lr, D, X, targets, inputs, 2 * d,
                             passes=12, n_cand=25)
        best = max(best, acc)
        if best >= 0.9:
            break
    return best


def main():
    for d in (8, 16):
        max_possible = 4 * d * d
        lo, hi, best, best_score = 1, max_possible, 0, None
        cache = {}
        while hi - lo >= PRECISION_FRACTION * hi:
            mid = (lo + hi) // 2
            if mid not in cache:
                cache[mid] = acc_at(d, mid)
                # near-boundary: double-check with more restarts before
                # declaring failure (variance handling)
                if 0.85 <= cache[mid] < 0.9:
                    cache[mid] = max(cache[mid], acc_at(d, mid, restarts=16))
                print(f"  d={d} n={mid}: acc {cache[mid]:.3f}", flush=True)
            if cache[mid] >= 0.9:
                best, best_score = mid, cache[mid]
                lo = mid + 1
            else:
                hi = mid - 1
        if hi == max_possible and max_possible not in cache:
            s = acc_at(d, max_possible, restarts=16)
            if s >= 0.9:
                best, best_score = max_possible, s
        rec = {"arch": "handcoded_h16", "d": d, "threshold": 0.9,
               "max_facts": best, "best_score": best_score,
               "ceiling": max_possible}
        with open(RESULTS_PATH, "a") as f:
            f.write(json.dumps(rec) + "\n")
        print(f"RESULT {rec}", flush=True)


if __name__ == "__main__":
    main()
