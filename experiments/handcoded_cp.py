"""Hand-coded bilinear layer via CP decomposition of the fact tensor.

Construction: a width-m bilinear layer y = D(Lx . Rx) with the folded
two-position encoding can exactly implement a rank-m CP decomposition of the
(V_in, V_in, V_out) fact tensor:

    L[n] = [a_n ; 0],  R[n] = [0 ; b_n]  =>  hidden_n = a_n[t1] * b_n[t2]
    D[c, n] = weight_n * c_n[c]          =>  logits = CP reconstruction.

So the max-facts of "parafac fit -> bilinear weights" lower-bounds what a
hand-coded bilinear layer can do. ALS (parafac) is a sequence of least-squares
solves — in the spirit of the challenge's "ridge regression is allowed" rule,
though reasonable people could call it a numeric optimizer; flagged in the
writeup.

The fact tensor is 1 at (t1, t2, label) for each stored fact, 0 elsewhere.
Success criterion: argmax over the label axis of the reconstruction at each
stored fact's (t1, t2) equals its label for >= threshold of facts.

Usage: python handcoded_cp.py --dvals 16,32,64 --thresholds 0.9,1.0
"""

import argparse
import json
import os
import time

import torch
import tensorly as tl
from tensorly.decomposition import parafac

from capacity import (generate_facts, width_for, param_count, RESULTS_DIR,
                      PRECISION_FRACTION)

tl.set_backend("pytorch")

RESULTS_PATH = os.path.join(RESULTS_DIR, "handcoded_cp.jsonl")
N_RESTARTS = 3  # random-init restarts (first try is svd init)


def cp_accuracy(d, m, n_facts, verbose=False):
    """Fit rank-m parafac to the fact tensor of the first n_facts facts;
    return the best argmax accuracy over restarts."""
    v_in, v_out = 2 * d, d
    inputs, targets = generate_facts(n_facts, v_in, v_out)
    fact_tensor = torch.zeros(v_in, v_in, v_out, device=inputs.device)
    fact_tensor[inputs[:, 0], inputs[:, 1], targets] = 1.0

    best_acc = 0.0
    for restart in range(N_RESTARTS):
        try:
            cp = parafac(
                fact_tensor,
                rank=m,
                n_iter_max=800,
                init="svd" if restart == 0 else "random",
                random_state=restart,
                tol=1e-9,
                l2_reg=1e-8,
                linesearch=True,
            )
        except Exception as e:  # svd init can fail on degenerate tensors
            if verbose:
                print(f"    restart {restart} failed: {e}")
            continue
        recon = tl.cp_to_tensor(cp)
        logits = recon[inputs[:, 0], inputs[:, 1]]  # (n_facts, V_out)
        acc = (logits.argmax(-1) == targets).float().mean().item()
        best_acc = max(best_acc, acc)
        if best_acc == 1.0:
            break
    return best_acc


def find_max_facts_cp(d, m, threshold, verbose=True):
    max_possible = 4 * d ** 2
    lo, hi = 1, max_possible
    best, best_score = 0, None
    cache = {}
    while hi - lo >= PRECISION_FRACTION * hi:
        mid = (lo + hi) // 2
        if mid not in cache:
            t0 = time.time()
            cache[mid] = cp_accuracy(d, m, mid)
            if verbose:
                print(f"    n={mid}: acc={cache[mid]:.4f} "
                      f"({time.time()-t0:.0f}s)", flush=True)
        if cache[mid] >= threshold:
            best, best_score = mid, cache[mid]
            lo = mid + 1
        else:
            hi = mid - 1
    if verbose:
        print(f"  => cp d={d} m={m} thr={threshold}: max_facts={best}", flush=True)
    return best, best_score


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dvals", default="16,32,64")
    p.add_argument("--thresholds", default="0.9,1.0")
    args = p.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    for d in [int(x) for x in args.dvals.split(",")]:
        m = width_for("bilinear", d, "param_matched")
        for thr in [float(x) for x in args.thresholds.split(",")]:
            best, score = find_max_facts_cp(d, m, thr)
            rec = {"arch": "cp_bilinear", "d": d, "m": m, "threshold": thr,
                   "max_facts": best, "best_score": score,
                   "params": param_count("bilinear", d, m)}
            with open(RESULTS_PATH, "a") as f:
                f.write(json.dumps(rec) + "\n")
            print(f"RESULT {rec}", flush=True)


if __name__ == "__main__":
    main()
