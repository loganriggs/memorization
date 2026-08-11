"""D-only iterative L1 pruning for sym_random d in {2,3,4,5}.

Same protocol as sparsity_d8_donly.py. For each d: curve figure + weights of
the sparsest model with finetuned acc >= 0.9, appended to that d's report.
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

from capacity import generate_facts
from sparsity_d8 import (accuracy, clone, LAMBDA, PRUNE_FRAC, L1_EPOCHS,
                         FT_EPOCHS, DEVICE, DIR)
from sparsity_d8_donly import train_d_penalty
import tiny_report

DVALS = [2, 3, 4, 5]

plt.rcParams.update({
    "figure.dpi": 150, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "font.size": 9,
    "axes.titlesize": 10,
})


def run_for_d(d):
    v_in, v_out, n = 2 * d, d, 4 * d * d
    saved = torch.load(os.path.join(DIR, f"d{d}_weights.pt"))
    w = {k: saved[k].to(DEVICE).clone().requires_grad_(True)
         for k in ("L", "down")}
    mask_d = torch.ones_like(w["down"])
    inputs, targets = generate_facts(n, v_in, v_out)
    x = torch.cat([F.one_hot(inputs[:, 0], v_in).float(),
                   F.one_hot(inputs[:, 1], v_in).float()], dim=-1).to(DEVICE)
    targets = targets.to(DEVICE)
    total_d = mask_d.numel()

    base = accuracy(w, x, targets)
    curve = [(0.0, base, base, total_d)]
    snaps = {total_d: {k: v.detach().cpu() for k, v in w.items()}}
    print(f"-- d={d}: start acc {base:.3f}, {total_d} D weights", flush=True)

    while True:
        train_d_penalty(w, mask_d, x, targets, L1_EPOCHS, l1=LAMBDA)
        vals = (w["down"].detach().abs() + (1 - mask_d) * 1e9).flatten()
        k_drop = max(1, int(round(PRUNE_FRAC * int(mask_d.sum()))))
        thresh = vals.kthvalue(k_drop).values
        mask_d *= (w["down"].detach().abs() > thresh).float()
        with torch.no_grad():
            w["down"].mul_(mask_d)
        nnz = int(mask_d.sum())
        frac_abl = 1 - nnz / total_d
        acc_pruned = accuracy(w, x, targets)
        wf = clone(w)
        train_d_penalty(wf, mask_d, x, targets, FT_EPOCHS, l1=0.0)
        acc_ft = accuracy(wf, x, targets)
        curve.append((frac_abl, acc_pruned, acc_ft, nnz))
        snaps[nnz] = {k: v.detach().cpu() for k, v in wf.items()}
        print(f"   d={d}: {nnz:>3} D left ({frac_abl:5.1%})  "
              f"pruned {acc_pruned:.3f}  finetuned {acc_ft:.3f}", flush=True)
        if acc_ft < 0.3 or nnz <= 1:
            break

    ok = [c for c in curve if c[2] >= 0.9]
    best = max(ok, key=lambda c: c[0]) if ok else curve[0]
    wf_best = snaps[best[3]]
    per_label = (wf_best["down"] != 0).sum(dim=1).int().tolist()

    # curve figure
    fig, ax = plt.subplots(figsize=(5.6, 3.6))
    fr = [c[0] * 100 for c in curve]
    ax.plot(fr, [c[1] for c in curve], "o-", color="#eb6834", lw=2, ms=3.5,
            label="after prune (under L1)")
    ax.plot(fr, [c[2] for c in curve], "o-", color="#2a78d6", lw=2, ms=3.5,
            label="after no-L1 finetune")
    ax.axhline(0.9, color="#8a8a85", ls=":", lw=1)
    ax.axvline(best[0] * 100, color="#8a8a85", ls="--", lw=1)
    ax.set_xlabel(f"% of D weights ablated (of {total_d}; L never pruned)")
    ax.set_ylabel("fact accuracy")
    ax.set_ylim(0, 1.05)
    ax.set_title(f"D-only L1 pruning, sym_random d={d} (n={n} facts)")
    ax.legend(frameon=False, fontsize=8, loc="lower left")
    fig.tight_layout()
    img_curve = f"img/d{d}_sparsity_donly.png"
    fig.savefig(os.path.join(DIR, img_curve), bbox_inches="tight")
    plt.close(fig)

    img_w = f"img/d{d}_donly_frontier_weights.png"
    tiny_report.fig_weights(wf_best, d, os.path.join(DIR, img_w))
    torch.save(wf_best, os.path.join(DIR, f"d{d}_donly_frontier_weights.pt"))

    md_path = os.path.join(DIR, f"d{d}.md")
    with open(md_path) as f:
        content = f.read()
    marker = "## Sparsity: D only"
    if marker in content:
        content = content[:content.index(marker)].rstrip() + "\n"
    section = [
        "", marker, "",
        f"Iterative L1 (λ={LAMBDA}) on D only + prune "
        f"{PRUNE_FRAC:.0%} of remaining (min 1) per round; L trains freely. "
        f"At each level a {FT_EPOCHS}-epoch no-L1 finetune measures "
        "recoverable accuracy.",
        "",
        f"Sparsest level with finetuned acc ≥ 0.9: **{best[3]} of {total_d} "
        f"D weights** ({best[0]:.0%} ablated), finetuned acc {best[2]:.3f}; "
        f"nonzero D weights per label: {per_label}.",
        "", f"![curve]({img_curve})", "",
        "Weights of that frontier model:",
        "", f"![frontier weights]({img_w})", ""]
    with open(md_path, "w") as f:
        f.write(content + "\n".join(section))
    return {"d": d, "total_d": total_d, "left": best[3],
            "frac": round(best[0], 3), "acc": round(best[2], 3),
            "per_label": per_label}


def main():
    results = [run_for_d(d) for d in DVALS]
    print()
    for r in results:
        print(r)


if __name__ == "__main__":
    main()
