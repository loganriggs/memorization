"""Iterative L1 + magnitude pruning applied ONLY to D (sym_random d=8).

Same loop as sparsity_d8.py, but the L1 penalty and the pruning mask touch
only the readout D; L trains freely throughout. Appends a section to
tiny_models/sym_random/d8.md.
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

from capacity import generate_facts
from sparsity_d8 import (forward, accuracy, clone,
                         V_IN, V_OUT, N, LAMBDA, PRUNE_FRAC, L1_EPOCHS,
                         FT_EPOCHS, LR, DEVICE, DIR)


def train_d_penalty(w, mask_d, x, targets, epochs, l1=0.0):
    opt = torch.optim.Adam(list(w.values()), lr=LR)
    for _ in range(epochs):
        opt.zero_grad(set_to_none=True)
        loss = F.cross_entropy(forward(w, x), targets)
        if l1:
            loss = loss + l1 * w["down"].abs().sum()
        loss.backward()
        opt.step()
        with torch.no_grad():
            w["down"].mul_(mask_d)


def main():
    saved = torch.load(os.path.join(DIR, "d8_weights.pt"))
    w = {k: saved[k].to(DEVICE).clone().requires_grad_(True)
         for k in ("L", "down")}
    mask_d = torch.ones_like(w["down"])
    inputs, targets = generate_facts(N, V_IN, V_OUT)
    x = torch.cat([F.one_hot(inputs[:, 0], V_IN).float(),
                   F.one_hot(inputs[:, 1], V_IN).float()], dim=-1).to(DEVICE)
    targets = targets.to(DEVICE)
    total_d = mask_d.numel()

    base = accuracy(w, x, targets)
    curve = [(0.0, base, base, total_d)]
    print(f"start: acc {base:.3f}, {total_d} D weights")

    for rnd in range(1, 200):
        train_d_penalty(w, mask_d, x, targets, L1_EPOCHS, l1=LAMBDA)
        vals = (w["down"].detach().abs() + (1 - mask_d) * 1e9).flatten()
        nnz = int(mask_d.sum())
        k_drop = max(1, int(round(PRUNE_FRAC * nnz)))
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
        print(f"round {rnd:>3}: D ablated {frac_abl:5.1%} ({nnz} left)  "
              f"acc(pruned+L1) {acc_pruned:.3f}  acc(finetuned) {acc_ft:.3f}",
              flush=True)
        if acc_ft < 0.3 or nnz <= V_OUT:
            break

    torch.save({"curve": curve}, os.path.join(DIR, "d8_sparsity_donly.pt"))
    ok = [c for c in curve if c[2] >= 0.9]
    best = max(ok, key=lambda c: c[0]) if ok else curve[0]

    plt.rcParams.update({
        "figure.dpi": 150, "axes.spines.top": False,
        "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.25,
        "font.size": 9, "axes.titlesize": 10,
    })
    fig, ax = plt.subplots(figsize=(6.2, 3.9))
    fr = [c[0] * 100 for c in curve]
    ax.plot(fr, [c[1] for c in curve], "o-", color="#eb6834", lw=2, ms=3.5,
            label="after prune (still under L1)")
    ax.plot(fr, [c[2] for c in curve], "o-", color="#2a78d6", lw=2, ms=3.5,
            label="after no-L1 finetune (L free throughout)")
    ax.axhline(0.9, color="#8a8a85", ls=":", lw=1)
    ax.text(1, 0.905, "acc 0.9", color="#8a8a85", fontsize=8)
    ax.axvline(best[0] * 100, color="#8a8a85", ls="--", lw=1)
    ax.text(best[0] * 100 - 1.5, 0.32,
            f"sparsest ≥0.9: {best[0]:.0%} of D ablated\n"
            f"({best[3]} D weights left, {best[3]/V_OUT:.1f}/label)",
            color="#555", fontsize=8, ha="right")
    ax.set_xlabel("% of D weights ablated (L never pruned)")
    ax.set_ylabel("fact accuracy")
    ax.set_ylim(0, 1.05)
    ax.set_title(f"L1 (λ={LAMBDA}) + prune {PRUNE_FRAC:.0%} of remaining — "
                 f"D only ({total_d} weights), sym_random d=8")
    ax.legend(frameon=False, fontsize=8, loc="lower left")
    fig.tight_layout()
    img = os.path.join(DIR, "img", "d8_sparsity_donly.png")
    fig.savefig(img, bbox_inches="tight")
    print(f"wrote {img}")

    md_path = os.path.join(DIR, "d8.md")
    with open(md_path) as f:
        content = f.read()
    marker = "## Sparsity: D only"
    if marker in content:
        content = content[:content.index(marker)].rstrip() + "\n"
    section = [
        "", marker, "",
        "Same iterative loop, but the L1 penalty and pruning apply only to "
        "the readout D; L trains freely (never penalized or masked).",
        "",
        f"Sparsest level keeping finetuned acc ≥ 0.9: **{best[0]:.0%} of D "
        f"ablated** ({best[3]} of {total_d} D weights remain — "
        f"{best[3]/V_OUT:.1f} per label).",
        "", "![sparsity D only](img/d8_sparsity_donly.png)", ""]
    with open(md_path, "w") as f:
        f.write(content + "\n".join(section))
    print(f"appended section to {md_path}")


if __name__ == "__main__":
    main()
