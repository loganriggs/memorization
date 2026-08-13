"""T30: tuning-budget Pareto frontiers, forget05 — the campaign's final figure.

Two panels, two threat models:
  A. auditor-indistinguishability: forget quality (KS p) vs model utility.
     Best corner: top-right (admissible AND functional).
  B. content-removal: forget-set generation leakage vs model utility.
     Best corner: bottom-right; the never-knew floor (0.395) marks where
     lower leakage stops meaning "removed" and starts meaning "suppressed".

Every method is drawn as a CURVE over its own tuning knob (ours: gamma;
NPO: lr; GA: epochs; RMU: steering_coeff; SimNPO: gamma), each point a
3-seed mean. This is the fair-comparison answer to fixed-config leaderboards:
compare frontiers, not single published points.

Writes reports/remote/fig_pareto_forget05.png/.svg and PARETO.md (table).
"""
import json
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RR = "../reports/remote"
P_CLIP = 1e-14
FLOOR_LEAK = 0.3950   # measured never-knew leakage floor, forget05
REF_UTIL = 0.5961     # retain95 reference utility under the frozen protocol


def rows(path):
    return [json.loads(l) for l in open(f"{RR}/{path}")]


def cellmean(rs, key):
    return sum(r[key] for r in rs) / len(rs)


def group(all_rows, cellmap):
    """cellmap: label -> cell prefix (without _s<seed>). Returns
    label -> dict(fq, util, leak, n, seeds[fq])."""
    out = {}
    for label, prefix in cellmap.items():
        rs = [r for r in all_rows if r["cell"].rsplit("_s", 1)[0] == prefix]
        if not rs:
            continue
        out[label] = {
            "fq": cellmean(rs, "fq_p_vs_retain95"),
            "util": cellmean(rs, "model_utility"),
            "leak": cellmean(rs, "forget_rouge"),
            "n": len(rs),
            "fq_seeds": [r["fq_p_vs_retain95"] for r in rs],
        }
    return out


def load_all():
    ours = rows("t20_forget05_sweep.jsonl")
    pub = rows("t23_baselines_forget05.jsonl")
    tuned = rows("t23p_pareto_forget05.jsonl")
    allr = pub + tuned

    curves = {}
    for scope in ("min", "all"):
        curves[f"ours ({scope}-token), γ"] = group(
            ours, {f"γ{g:g}": f"t20_forget05_{scope}_g{g:g}"
                   for g in (0.5, 1, 2, 4)})
    curves["NPO, lr"] = group(allr, {
        "1e-5 (pub)": "t23_forget05_npo",
        "2e-5": "t23_forget05_npo_lr2e-05",
        "5e-5": "t23_forget05_npo_lr5e-05"})
    curves["GA, epochs"] = group(allr, {
        "2": "t23p_ga_2ep", "5": "t23p_ga_5ep",
        "10 (pub)": "t23_forget05_gradascent"})
    curves["RMU, steer coeff"] = group(allr, {
        "2 (pub)": "t23_forget05_rmu", "5": "t23p_rmu_sc5",
        "20": "t23p_rmu_sc20"})
    curves["SimNPO, γ"] = group(allr, {
        "0.125 (pub)": "t23_forget05_simnpo", "1.0": "t23p_simnpo_g1"})
    return curves


STYLE = {  # curve -> (color, marker)
    "ours (min-token), γ": ("#d62728", "s"),
    "ours (all-token), γ": ("#1f77b4", "o"),
    "NPO, lr": ("#9467bd", "v"),
    "GA, epochs": ("#2ca02c", "^"),
    "RMU, steer coeff": ("#e377c2", "P"),
    "SimNPO, γ": ("#8c564b", "D"),
}


def draw(ax, curves, ykey):
    for name, pts in curves.items():
        c, m = STYLE[name]
        # order points along the curve by their knob position in the cellmap
        labs = list(pts)
        xs = [pts[l]["util"] for l in labs]
        ys = [max(pts[l][ykey], P_CLIP) if ykey == "fq" else pts[l][ykey]
              for l in labs]
        ax.plot(xs, ys, marker=m, color=c, lw=1.3, ms=6, label=name,
                alpha=0.9)
        for l, x, y in zip(labs, xs, ys):
            ax.annotate(l, (x, y), textcoords="offset points", xytext=(5, 4),
                        fontsize=6.5, color=c)
    ax.axvline(REF_UTIL, color="#888", lw=1, ls=":")
    ax.set_xlabel("model utility (retain-ref 0.596)")


def main():
    curves = load_all()
    fig, (a, b) = plt.subplots(1, 2, figsize=(13.2, 5.4), dpi=150)

    a.set_yscale("log")
    a.axhline(0.05, color="#888", lw=1, ls="--")
    a.text(0.02, 0.065, "admissible p = 0.05", fontsize=8, color="#555")
    draw(a, curves, "fq")
    a.set_ylim(P_CLIP / 3, 3)
    a.set_ylabel("forget quality (KS p vs retain95 log)")
    a.set_title("A — auditor framing: indistinguishability vs utility\n"
                "(want top-right)")
    a.legend(fontsize=7, loc="lower left", framealpha=0.9)

    draw(b, curves, "leak")
    b.axhline(FLOOR_LEAK, color="#888", lw=1, ls="--")
    b.text(0.02, FLOOR_LEAK + 0.012, "never-knew floor 0.395 (below = "
           "over-suppression)", fontsize=7.5, color="#555")
    b.set_ylim(0, 0.8)
    b.set_ylabel("forget-set gen ROUGE-L recall (leakage)")
    b.set_title("B — content-removal framing: leakage vs utility\n"
                "(want bottom-right, at or above floor)")

    fig.suptitle("TOFU forget05, Llama-3.2-1B — tuning-budget Pareto "
                 "frontiers (3-seed means, frozen protocol)", y=1.00)
    fig.tight_layout()
    for ext in ("png", "svg"):
        fig.savefig(f"{RR}/fig_pareto_forget05.{ext}", bbox_inches="tight")
    print("wrote fig_pareto_forget05.png/.svg")

    with open(f"{RR}/PARETO.md", "w") as f:
        f.write("# Tuning-budget Pareto sweep — forget05 (3-seed means)\n\n"
                "| method | knob | FQ p (seeds) | utility | leakage |\n"
                "|---|---|---|---|---|\n")
        for name, pts in curves.items():
            for l, d in pts.items():
                seeds = ", ".join(f"{s:.2g}" for s in d["fq_seeds"])
                f.write(f"| {name.split(',')[0]} | {l} | "
                        f"**{d['fq']:.3g}** ({seeds}) | {d['util']:.3f} | "
                        f"{d['leak']:.3f} |\n")
    print("wrote PARETO.md")


if __name__ == "__main__":
    main()
