"""Analyze the multiseed fleet: Hungarian-align canonical solutions across
seeds, measure similarity, and census the D motifs and ratios (H2/H3/H7).

Neuron matching: labels are fixed across seeds (same facts), only neurons
permute. Feature per neuron n = [unit-norm L row ; unit-norm D column].
Hungarian (scipy) maximizes total cosine similarity; report L-part and D-part
similarities separately for every seed pair.

Outputs: results/multiseed_similarity.png, console census, and appends
findings to research_log.md manually (done by the caller).
"""

import glob
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "tiny_models", "sym_random", "multiseed")
RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


def load(d):
    out = {}
    for p in sorted(glob.glob(os.path.join(DIR, f"d{d}_seed*.pt"))):
        s = int(re.search(r"seed(\d+)", p).group(1))
        out[s] = torch.load(p)
    return out


def neuron_features(canon):
    L, D = canon["L"], canon["down"]
    Ln = L / L.norm(dim=1, keepdim=True).clamp_min(1e-9)
    Dc = D.T  # (m, V_out) columns as rows
    Dn = Dc / Dc.norm(dim=1, keepdim=True).clamp_min(1e-9)
    return Ln, Dn


def align_pair(ca, cb):
    """Hungarian on combined cos sim; returns (mean L cos, mean D cos)
    over matched neurons, using |cos| for L (row-sign gauge is only fixed
    up to the argmax heuristic)."""
    La, Da = neuron_features(ca)
    Lb, Db = neuron_features(cb)
    simL = (La @ Lb.T).abs()
    simD = Da @ Db.T
    cost = -(simL + simD)
    r, c = linear_sum_assignment(cost.numpy())
    return float(simL[r, c].mean()), float(simD[r, c].mean())


def census(d, runs):
    tap_counts, defaults, ratios, neg_frac = [], 0, [], []
    for s, rec in runs.items():
        D = rec["canonical"]["down"]
        per_label = (D != 0).sum(dim=1).int().tolist()
        tap_counts.append(sorted(per_label))
        defaults += sum(1 for t in per_label if t == 0)
        nz = D[D != 0]
        neg_frac.append(float((nz < 0).float().mean()))
        for c in range(D.shape[0]):
            row = D[c]
            pos = row[row > 0]
            neg = row[row < 0]
            if len(pos) == 1 and len(neg) == 1:
                ratios.append(float(pos.abs() / neg.abs()))
    return tap_counts, defaults, ratios, neg_frac


def main():
    dvals = range(2, 9)
    fig, axes = plt.subplots(2, 4, figsize=(14, 6.4))
    summary = []
    for i, d in enumerate(dvals):
        runs = load(d)
        seeds = sorted(runs)
        n = len(seeds)
        ML = np.eye(n)
        MD = np.eye(n)
        for a in range(n):
            for b in range(a + 1, n):
                sl, sd = align_pair(runs[seeds[a]]["canonical"],
                                    runs[seeds[b]]["canonical"])
                ML[a, b] = ML[b, a] = sl
                MD[a, b] = MD[b, a] = sd
        off = ~np.eye(n, dtype=bool)
        tap_counts, defaults, ratios, neg_frac = census(d, runs)
        summary.append({
            "d": d, "n_seeds": n,
            "L_cos": (ML[off].mean(), ML[off].std()),
            "D_cos": (MD[off].mean(), MD[off].std()),
            "mean_taps_per_label": np.mean([np.mean(t) for t in tap_counts]),
            "default_labels": defaults,
            "neg_frac": np.mean(neg_frac),
            "ratios": ratios,
        })
        ax = axes[i // 4][i % 4]
        im = ax.imshow((ML + MD) / 2, vmin=0, vmax=1, cmap="Blues")
        ax.set_title(f"d={d}  L̄cos {ML[off].mean():.2f} "
                     f"D̄cos {MD[off].mean():.2f}", fontsize=9)
        ax.set_xticks(range(n)), ax.set_yticks(range(n))
        ax.set_xticklabels(seeds, fontsize=6)
        ax.set_yticklabels(seeds, fontsize=6)
    axes[1][3].axis("off")
    # pooled ratio histogram
    axr = axes[1][3]
    axr.axis("on")
    all_r = [r for s in summary for r in s["ratios"]]
    axr.hist(all_r, bins=30, range=(0, 1.5), color="#eb6834")
    axr.set_title("pos/|neg| ratio, 2-tap rows (all d)", fontsize=9)
    fig.suptitle("Seed×seed similarity of Hungarian-aligned canonical "
                 "solutions (mean of L-cos and D-cos)", y=1.0)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, "multiseed_similarity.png"),
                bbox_inches="tight", dpi=150)

    print(f"{'d':>3} {'L cos':>12} {'D cos':>12} {'taps/lab':>9} "
          f"{'defaults':>9} {'neg frac':>9} {'2tap ratios (median)':>21}")
    for s in summary:
        med = np.median(s["ratios"]) if s["ratios"] else float("nan")
        print(f"{s['d']:>3} {s['L_cos'][0]:6.3f}±{s['L_cos'][1]:.3f} "
              f"{s['D_cos'][0]:6.3f}±{s['D_cos'][1]:.3f} "
              f"{s['mean_taps_per_label']:>9.2f} {s['default_labels']:>9} "
              f"{s['neg_frac']:>9.2f} {med:>21.3f}"
              f"   (n={len(s['ratios'])})")


if __name__ == "__main__":
    main()
