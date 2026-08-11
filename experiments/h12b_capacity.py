"""Task 19: capacity (max facts at acc>=0.9) of the H12b construction.

Binary search over n_facts (same 2%-relative protocol as everything else),
evaluating the full non-GD pipeline (H9b spectral solve + hinge repair) at
each candidate n. Appends results to results/handcoded_h12b.jsonl.

Usage: python h12b_capacity.py [--dvals 6,8,12] [--restarts 2]
"""

import argparse
import json
import os

import torch
import torch.nn.functional as F

from capacity import generate_facts, PRECISION_FRACTION, RESULTS_DIR
from h12b_repair import h9b_solve, repair

RESULTS_PATH = os.path.join(RESULTS_DIR, "handcoded_h12b.jsonl")


def acc_at(d, n, restarts, gen):
    v_in, v_out = 2 * d, d
    inputs, targets = generate_facts(n, v_in, v_out)
    inputs, targets = inputs.cpu(), targets.cpu()
    X = torch.cat([F.one_hot(inputs[:, 0], v_in).float(),
                   F.one_hot(inputs[:, 1], v_in).float()], dim=-1).double()
    acc0, L0 = h9b_solve(X, targets, v_out)
    best = acc0
    for r in range(restarts):
        Lr = L0 if r == 0 else L0 + 0.1 * torch.randn(
            L0.shape, generator=gen, dtype=torch.float64)
        acc, _ = repair(Lr, X, targets, v_out, inputs, v_in, passes=6)
        best = max(best, acc)
        if best >= 0.9:
            break
    return best


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dvals", default="6,8,12")
    p.add_argument("--restarts", type=int, default=2)
    args = p.parse_args()
    gen = torch.Generator().manual_seed(0)
    for d in [int(x) for x in args.dvals.split(",")]:
        max_possible = 4 * d * d
        lo, hi, best, best_score = 1, max_possible, 0, None
        cache = {}
        while hi - lo >= PRECISION_FRACTION * hi:
            mid = (lo + hi) // 2
            if mid not in cache:
                cache[mid] = acc_at(d, mid, args.restarts, gen)
                print(f"  d={d} n={mid}: acc {cache[mid]:.3f}", flush=True)
            if cache[mid] >= 0.9:
                best, best_score = mid, cache[mid]
                lo = mid + 1
            else:
                hi = mid - 1
        if hi == max_possible and max_possible not in cache:
            s = acc_at(d, max_possible, args.restarts, gen)
            if s >= 0.9:
                best, best_score = max_possible, s
        rec = {"arch": "handcoded_h12b", "d": d, "threshold": 0.9,
               "max_facts": best, "best_score": best_score,
               "ceiling": max_possible}
        with open(RESULTS_PATH, "a") as f:
            f.write(json.dumps(rec) + "\n")
        print(f"RESULT {rec}", flush=True)


if __name__ == "__main__":
    main()
