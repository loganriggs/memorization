"""T21: forget quality against open-unlearning's PUBLISHED retain logs.

Two modes:

  selftest <ours_tr_json> <model_dirname> <split>
      Convention check: our full-model truth ratios vs their published
      full-model log for the same split. Same model on both sides, so if the
      transform is right the KS p-value must be ~1. Run before any FQ number
      is trusted. Example:
        python t21_fq_published.py selftest \
            results/t15_truthratios/p2_llama_ours_v2.json \
            tofu_Llama-3.2-1B-Instruct_full forget01

  fq <ours_tr_json> <retain_dirname> <split>
      Real forget quality: an unlearned model's TRs vs the published retain
      reference log. Example:
        python t21_fq_published.py fq results/t15_truthratios/<tag>.json \
            tofu_Llama-3.2-1B-Instruct_retain95 forget05

Transform: our stored R = correct/wrong (paraphrased over mean-perturbed);
their per-example score = wrong/correct (memorization.py:163-171, identical
aggregation). So theirs = 1/ours, applied to OUR sample before the KS. The KS
statistic is invariant under a strictly monotonic transform applied to BOTH
samples, but not to one -- hence the explicit inversion.
"""
import glob
import json
import sys

import numpy as np
from scipy.stats import ks_2samp

EVAL_DS = "/workspace/.hf_home/hub/datasets--open-unlearning--eval/snapshots/*"


def published_trs(model_dirname, split):
    # Layout differs by model kind: the full model carries per-split subdirs
    # (evals_forget01/05/10); each retain reference pairs with exactly one
    # forget split, so its TOFU_EVAL.json sits at the top level.
    pats = [f"{EVAL_DS}/{model_dirname}/evals_{split}/TOFU_EVAL.json",
            f"{EVAL_DS}/{model_dirname}/TOFU_EVAL.json"]
    fs = [f for pat in pats for f in glob.glob(pat)]
    if not fs:
        sys.exit(f"no published log matches any of {pats}")
    j = json.load(open(fs[0]))
    vals = [v["score"] for v in j["forget_truth_ratio"]["value_by_index"].values()]
    return np.array(vals, dtype=np.float64)


def ours_trs(path):
    return np.array(json.load(open(path))["forget"], dtype=np.float64)


def main():
    mode, ours_path, dirname, split = sys.argv[1:5]
    ours = 1.0 / (ours_trs(ours_path) + 1e-10)   # -> their wrong/correct convention
    theirs = published_trs(dirname, split)
    stat, p = ks_2samp(ours, theirs)
    print(f"n_ours={len(ours)} n_theirs={len(theirs)} "
          f"mean_ours={ours.mean():.4f} mean_theirs={theirs.mean():.4f}")
    print(f"KS stat={stat:.4f} p={p:.6f}")
    if mode == "selftest":
        ok = p > 0.5
        print("SELFTEST " + ("PASS" if ok else
              "FAIL -- transform or span mismatch; fix ours, never resample"))
        sys.exit(0 if ok else 1)
    print(f"forget_quality p={p:.6f} "
          f"({'PASS' if p > 0.05 else 'FAIL'} at 0.05)")


if __name__ == "__main__":
    main()
