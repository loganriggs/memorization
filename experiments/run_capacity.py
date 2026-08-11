"""Driver: run all capacity searches and append results to results/capacity.jsonl.

Usage: python run_capacity.py [--archs mlp,bilinear,swiglu] [--dvals 16,32,64,128]
       [--thresholds 0.9,1.0] [--width-modes param_matched,equal_width]
"""

import argparse
import json
import os
import time

import capacity
from capacity import (GridCache, find_max_facts, width_for, param_count,
                      RESULTS_DIR)

RESULTS_PATH = os.path.join(RESULTS_DIR, "capacity.jsonl")


def already_done(done, rec_key):
    return rec_key in done


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--archs", default="mlp,bilinear,swiglu")
    p.add_argument("--dvals", default="16,32,64,128")
    p.add_argument("--thresholds", default="0.9,1.0")
    p.add_argument("--width-modes", default="param_matched")
    args = p.parse_args()

    archs = args.archs.split(",")
    dvals = [int(x) for x in args.dvals.split(",")]
    thresholds = [float(x) for x in args.thresholds.split(",")]
    width_modes = args.width_modes.split(",")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    done = set()
    if os.path.exists(RESULTS_PATH):
        with open(RESULTS_PATH) as f:
            for line in f:
                r = json.loads(line)
                done.add((r["arch"], r["d"], r["m"], r["threshold"]))

    cache = GridCache()
    t_start = time.time()
    for d in dvals:                       # small sizes first: fast feedback
        for wm in width_modes:
            for arch in archs:
                m = width_for(arch, d, wm)
                if arch == "mlp" and wm != width_modes[0]:
                    continue              # mlp width doesn't depend on mode
                for thr in thresholds:
                    if (arch, d, m, thr) in done:
                        print(f"skip {arch} d={d} m={m} thr={thr} (done)")
                        continue
                    best, score = find_max_facts(cache, arch, d, m, thr)
                    rec = {
                        "arch": arch, "d": d, "m": m, "width_mode": wm,
                        "threshold": thr, "max_facts": best,
                        "best_score": score,
                        "params": param_count(arch, d, m),
                        "n_attempts": capacity.N_ATTEMPTS,
                        "elapsed_total_s": round(time.time() - t_start),
                    }
                    with open(RESULTS_PATH, "a") as f:
                        f.write(json.dumps(rec) + "\n")
                    done.add((arch, d, m, thr))
                    print(f"RESULT {rec}", flush=True)
    print("all done")


if __name__ == "__main__":
    main()
