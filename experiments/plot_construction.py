"""Figure: hand-coded construction progression (accuracy at the 4d^2
ceiling vs d) and capacity fraction of the best construction."""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from capacity import RESULTS_DIR

DV = [2, 3, 4, 5, 6, 7, 8, 12, 16]
SERIES = {  # acc at ceiling; None = not run
    "H1 Rayleigh detectors (D=I)":
        {2: .875, 3: .778, 4: .688, 5: .530, 6: .444, 7: .362, 8: .391,
         12: .240, 16: .204},
    "H9 anti-Rayleigh silence (D=-I)":
        {2: .812, 3: .861, 4: .672, 5: .560, 6: .465, 7: .485, 8: .387,
         12: .309, 16: .276},
    "H9b + fact reweighting":
        {2: 1.0, 3: .889, 4: .828, 5: .720, 6: .597, 7: .551, 8: .465,
         12: .373, 16: .322},
    "H12b + hinge greedy repair":
        {2: 1.0, 3: 1.0, 4: .938, 5: .910, 6: .854, 8: .758},
    "H13a shared-neuron tap graph":
        {4: 1.0},
}
COLORS = ["#8a8a85", "#eda100", "#1baf7a", "#eb6834", "#2a78d6"]

plt.rcParams.update({
    "figure.dpi": 150, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "font.size": 9,
    "axes.titlesize": 10,
})
fig, ax = plt.subplots(figsize=(6.4, 4.2))
for (name, vals), color in zip(SERIES.items(), COLORS):
    ds = [d for d in DV if d in vals]
    ax.plot(ds, [vals[d] for d in ds], "o-", color=color, lw=2, ms=4,
            label=name)
ax.axhline(1.0, color="#444", ls="--", lw=1)
ax.text(12.6, 1.012, "trained (saturates ceiling)", fontsize=8, color="#444")
ax.axhline(0.9, color="#8a8a85", ls=":", lw=1)
ax.set_xlabel("model size $d$")
ax.set_ylabel("accuracy at the full $4d^2$ fact ceiling")
ax.set_xscale("log", base=2)
ax.set_xticks(DV)
ax.set_xticklabels(DV)
ax.set_ylim(0, 1.05)
ax.set_title("Hand-coded (no-GD) constructions for the symmetric bilinear "
             "memorizer")
ax.legend(frameon=False, fontsize=8, loc="lower left")
fig.tight_layout()
fig.savefig(os.path.join(RESULTS_DIR, "fig_construction.png"),
            bbox_inches="tight")
print("wrote results/fig_construction.png")
