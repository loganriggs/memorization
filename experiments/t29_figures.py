"""T29: frontier figure — forget quality vs model utility, forget05.

The matrix's headline visual: every method wants the top-right (admissible FQ,
intact utility). Renders from the jsonls; regenerate any time. Writes
reports/remote/fig_frontier_forget05.png (+ .svg).
"""
import json
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RR = "../reports/remote"
FLOOR_P = 0.05
P_CLIP = 1e-6           # display clip for p-values at/near 0
REF = {"util": 0.5961}  # retain95 under the frozen protocol


def load(path, prefix):
    if not os.path.exists(path):
        return {}
    groups = defaultdict(list)
    for l in open(path):
        r = json.loads(l)
        name = r["cell"].replace(prefix, "").rsplit("_", 1)[0]
        groups[name].append(r)
    return groups


def mean(rs, k):
    vs = [(x.get("fq_p_vs_retain95") or 0.0) if k == "fq" else x[k] for x in rs]
    return sum(vs) / len(vs)


def main():
    ours = load(f"{RR}/t20_forget05_sweep.jsonl", "t20_forget05_")
    base = load(f"{RR}/t23_baselines_forget05.jsonl", "t23_forget05_")

    fig, ax = plt.subplots(figsize=(7.2, 5.2), dpi=150)
    ax.set_yscale("log")
    ax.axhline(FLOOR_P, color="#888", lw=1, ls="--")
    ax.text(0.015, FLOOR_P * 1.25, "admissibility p = 0.05", fontsize=8,
            color="#555")
    ax.axvline(REF["util"], color="#888", lw=1, ls=":")
    ax.text(REF["util"] - 0.005, 2.2e-6, "retain-reference utility",
            fontsize=8, color="#555", rotation=90, va="bottom", ha="right")

    # ours: gamma traces a curve per scope
    for scope, marker, color in (("all", "o", "#1f77b4"), ("min", "s", "#d62728")):
        pts = []
        for name, rs in sorted(ours.items()):
            if not name.startswith(scope):
                continue
            g = float(name.split("_g")[1])
            pts.append((g, mean(rs, "util_placeholder" if False else "model_utility"),
                        max(mean(rs, "fq"), P_CLIP)))
        pts.sort()
        xs = [p[1] for p in pts]
        ys = [p[2] for p in pts]
        ax.plot(xs, ys, marker=marker, color=color, lw=1.2, ms=6,
                label=f"ours ({scope}-token), γ ∈ {{0.5,1,2,4}}")
        for (g, x, y) in pts:
            ax.annotate(f"γ{g:g}", (x, y), textcoords="offset points",
                        xytext=(5, 4), fontsize=7, color=color)

    # baselines: one point each
    bmark = {"gradascent": ("GA", "^", "#2ca02c"), "npo": ("NPO", "v", "#9467bd"),
             "simnpo": ("SimNPO", "D", "#8c564b"), "rmu": ("RMU", "P", "#e377c2")}
    for name, rs in sorted(base.items()):
        lbl, m, c = bmark.get(name, (name, "x", "#333"))
        x, y = mean(rs, "model_utility"), max(mean(rs, "fq"), P_CLIP)
        n = len(rs)
        ax.scatter([x], [y], marker=m, c=c, s=70, zorder=5,
                   label=f"{lbl} (n={n} seed{'s' if n > 1 else ''})")
        ax.annotate(lbl, (x, y), textcoords="offset points", xytext=(6, -3),
                    fontsize=8, color=c)

    ax.set_xlabel("model utility (harmonic mean of 9; retain-ref = 0.596)")
    ax.set_ylabel("forget quality (KS p vs published retain95 log)")
    ax.set_title("TOFU forget05, Llama-3.2-1B — forget quality vs utility\n"
                 "(means over seeds; frozen protocol; p clipped at 1e-6)")
    ax.legend(fontsize=7.5, loc="lower left", framealpha=0.9)
    ax.set_ylim(P_CLIP / 2, 1)
    fig.tight_layout()
    for ext in ("png", "svg"):
        fig.savefig(f"{RR}/fig_frontier_forget05.{ext}")
    print("wrote fig_frontier_forget05.png/.svg")


if __name__ == "__main__":
    main()
