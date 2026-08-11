"""Scaling figure: max facts vs model size for all architectures.

Produces results/fig_scaling.png (two panels: acc>=0.9, acc=1.0, log-log)
with power-law fits, plus a markdown capacity table on stdout.
"""

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from capacity import RESULTS_DIR

C = {"mlp": "#2a78d6", "bilinear": "#eb6834", "swiglu": "#1baf7a",
     "cp_bilinear": "#eda100"}
LABEL = {"mlp": "MLP (ReLU)", "bilinear": "Bilinear", "swiglu": "SwiGLU",
         "cp_bilinear": "Hand-coded bilinear (CP/ALS)"}
GRAY = "#8a8a85"

# Their published trained numbers (any-of-11), from
# ref_repo/hand_coded_models/hc2_sweep_results/capacity_search_results_fulltrain.json
PUBLISHED = {
    0.9: {16: 784, 32: 2560, 64: 8320, 128: 27648, 256: 93184},
    1.0: {16: 568, 32: 2176, 64: 7680, 128: 25600, 256: 88064},
}

plt.rcParams.update({
    "figure.dpi": 150, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.6,
    "font.size": 9, "axes.titlesize": 10,
})


def load(path, extra=None):
    rows = []
    if os.path.exists(path):
        with open(path) as f:
            rows = [json.loads(l) for l in f]
    return rows


def fit_power(ds, ns):
    ds, ns = np.asarray(ds, float), np.asarray(ns, float)
    if len(ds) < 2 or (ns <= 0).any():
        return None
    b, a = np.polyfit(np.log(ds), np.log(ns), 1)
    return np.exp(a), b  # N = A * d^b


def main():
    rows = load(os.path.join(RESULTS_DIR, "capacity.jsonl"))
    rows += load(os.path.join(RESULTS_DIR, "handcoded_cp.jsonl"))
    rows = [r for r in rows if r.get("width_mode", "param_matched")
            == "param_matched"]

    fig = plt.figure(figsize=(10.5, 6.4))
    gs = fig.add_gridspec(2, 2, height_ratios=[2.2, 1], hspace=0.32)
    axes = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])]
    raxes = [fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])]
    for ax, rax, thr in zip(axes, raxes, (0.9, 1.0)):
        for arch in ("mlp", "bilinear", "swiglu", "cp_bilinear"):
            pts = sorted((r["d"], r["max_facts"]) for r in rows
                         if r["arch"] == arch and r["threshold"] == thr)
            if not pts:
                continue
            ds, ns = zip(*pts)
            fit = fit_power(ds, ns)
            lbl = LABEL[arch]
            if fit:
                lbl += f"  ($\\propto d^{{{fit[1]:.2f}}}$)"
            ax.plot(ds, ns, "o-", color=C[arch], lw=2, ms=5, label=lbl)
        pd_, pn = zip(*sorted(PUBLISHED[thr].items()))
        ax.plot(pd_, pn, "s--", color=GRAY, lw=1.5, ms=4,
                label="Their trained MLP (published)")
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_title(f"acc ≥ {thr:g}" if thr < 1 else "acc = 1.0")
        ax.legend(frameon=False, fontsize=8, loc="upper left")

        # ratio panel: capacity relative to my trained MLP
        mlp_vals = {r["d"]: r["max_facts"] for r in rows
                    if r["arch"] == "mlp" and r["threshold"] == thr}
        for arch in ("bilinear", "swiglu"):
            pts = sorted((r["d"], r["max_facts"]) for r in rows
                         if r["arch"] == arch and r["threshold"] == thr
                         and r["d"] in mlp_vals)
            if not pts:
                continue
            ds, ns = zip(*pts)
            ratio = [n / mlp_vals[dd] for dd, n in pts]
            rax.plot(ds, ratio, "o-", color=C[arch], lw=2, ms=5)
        rax.axhline(1.0, color=GRAY, lw=1, ls="--")
        rax.set_xscale("log", base=2)
        rax.set_xlabel("model size $d$   ($V_{in}=2d$, $V_{out}=d$)")
        rax.set_ylabel("× MLP capacity")
        rax.set_ylim(0.9, 1.85)
    axes[0].set_ylabel("max facts memorized (any of 11 seeds)")
    fig.suptitle("Sequence-memorization capacity, parameter-matched "
                 "(≈$5d^2$ params)", y=1.0)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "fig_scaling.png"),
                bbox_inches="tight")

    # ── random-features figure ───────────────────────────────────────────────
    rf = load(os.path.join(RESULTS_DIR, "randfeat.jsonl"))
    if rf:
        # Their published rand-emb (random up, Adam-trained down), acc>=0.9 any
        pub_randemb = {16: 36, 32: 196, 64: 800, 128: 2624}
        fig2, ax = plt.subplots(figsize=(5.2, 4))
        styles = {
            "mlp_randfeat_gd": (C["mlp"], "-", "o", "MLP feats + Adam readout"),
            "mlp_randfeat_ridge": (C["mlp"], ":", "^",
                                   "MLP feats + ridge readout"),
            "bilinear_randfeat_gd": (C["bilinear"], "-", "o",
                                     "Bilinear feats + Adam readout"),
            "bilinear_randfeat_ridge": (C["bilinear"], ":", "^",
                                        "Bilinear feats + ridge readout"),
        }
        for name, (color, ls, mk, lbl) in styles.items():
            pts = sorted((r["d"], r["max_facts"]) for r in rf
                         if r["arch"] == name and r["threshold"] == 0.9)
            if not pts:
                continue
            ds, ns = zip(*pts)
            ax.plot(ds, ns, ls, marker=mk, color=color, lw=2, ms=5, label=lbl)
        pd_, pn = zip(*sorted(pub_randemb.items()))
        ax.plot(pd_, pn, "s--", color=GRAY, lw=1.5, ms=4,
                label="Their rand-emb (published)")
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xlabel("model size $d$")
        ax.set_ylabel("max facts (acc ≥ 0.9, any of 11 seeds)")
        ax.set_title("Frozen random input weights, readout only")
        ax.legend(frameon=False, fontsize=8)
        fig2.tight_layout()
        fig2.savefig(os.path.join(RESULTS_DIR, "fig_randfeat.png"),
                     bbox_inches="tight")

    # markdown table
    print("| arch | thr | " + " | ".join(f"d={d}" for d in (16, 32, 64, 128))
          + " |")
    print("|---|---|" + "---|" * 4)
    for thr in (0.9, 1.0):
        for arch in ("mlp", "bilinear", "swiglu", "cp_bilinear"):
            vals = {r["d"]: r["max_facts"] for r in rows
                    if r["arch"] == arch and r["threshold"] == thr}
            if not vals:
                continue
            cells = " | ".join(str(vals.get(d, "—")) for d in (16, 32, 64, 128))
            print(f"| {LABEL[arch]} | {thr:g} | {cells} |")
        pub = PUBLISHED[thr]
        cells = " | ".join(str(pub.get(d, "—")) for d in (16, 32, 64, 128))
        print(f"| their trained MLP | {thr:g} | {cells} |")


if __name__ == "__main__":
    main()
