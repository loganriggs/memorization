"""T36: RWKU pilot figure — forget depth vs neighbor collateral on REAL
pretrained knowledge (10 targets, base-normalized). Panel B: probe-level
breakdown (cloze / QA / adversarial, plus neighbor levels)."""
import json
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RR = "../reports/remote"
rows = [json.loads(l) for l in open("results/t35_rwku.jsonl")
        if '"forget_mean"' in l or "forget_mean" in l]
rows = [r for r in rows if "forget_mean" in r]
base = {r["k"]: r for r in rows if r["tag"].startswith("t35_base")}
COLORS = {"ga": "#2ca02c", "npo": "#9467bd", "ours": "#d62728", "hybrid": "#e6873c"}

pts = defaultdict(list)   # method -> [(neighbor_kept, forget_drop)]
for r in rows:
    if r["tag"].startswith("t35_base"):
        continue
    m = r["tag"].split("_")[1]
    b = base[r["k"]]
    pts[m].append((r["neighbor_mean"] / max(b["neighbor_mean"], 1e-9),
                   1 - r["forget_mean"] / max(b["forget_mean"], 1e-9)))

fig, (a, bx) = plt.subplots(1, 2, figsize=(12.8, 5.0), dpi=150)
for m, v in pts.items():
    v = np.array(v)
    a.scatter(v[:, 0] * 100, v[:, 1] * 100, c=COLORS[m], s=28, alpha=0.45)
    a.scatter([v[:, 0].mean() * 100], [v[:, 1].mean() * 100], c=COLORS[m],
              s=220, marker="*", edgecolors="black", linewidths=0.8,
              label=f"{m.upper() if m != 'ours' else 'ours'} (mean of 10)",
              zorder=5)
a.set_xlabel("neighbor knowledge kept (% of base)")
a.set_ylabel("forget-target knowledge removed (% of base)")
a.set_title("A — RWKU pilot: forget depth vs collateral\n"
            "(real pretrained knowledge; want top-right)")
a.set_xlim(-5, 105)
a.set_ylim(-5, 105)
a.grid(alpha=0.25)
a.legend(fontsize=8, loc="lower left")

keys = [("forget_l1", "cloze"), ("forget_l2", "QA"),
        ("forget_l3", "adversarial"), ("neighbor_l1", "neigh cloze"),
        ("neighbor_l2", "neigh QA")]
methods = [m for m in ("ga", "npo", "ours", "hybrid") if m in pts]
x = np.arange(len(keys))
w = 0.8 / max(len(methods), 1)
for i, m in enumerate(methods):
    vals = []
    for key, _ in keys:
        fr = [r[key] / max(base[r["k"]][key], 1e-9) for r in rows
              if not r["tag"].startswith("t35_base")
              and r["tag"].split("_")[1] == m]
        vals.append(100 * np.mean(fr))
    bx.bar(x + (i - (len(methods)-1)/2) * w, vals, w, color=COLORS[m], label=m)
bx.axhline(100, color="#888", lw=1, ls=":")
bx.set_xticks(x)
bx.set_xticklabels([lbl for _, lbl in keys], fontsize=9)
bx.set_ylabel("score retained (% of base model)")
bx.set_title("B — probe-level breakdown\n"
             "(left 3 lower = better forgetting; right 2 higher = better)")
bx.legend(fontsize=9)
bx.grid(axis="y", alpha=0.25)

fig.suptitle("RWKU pilot, Llama-3.2-1B — unlearning real-world knowledge "
             "(10 targets, single declared config per method)", y=1.02)
fig.tight_layout()
for ext in ("png", "svg"):
    fig.savefig(f"{RR}/fig_rwku_pilot.{ext}", bbox_inches="tight")
print("wrote fig_rwku_pilot")
