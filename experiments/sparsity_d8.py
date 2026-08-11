"""Iterative L1 + magnitude pruning of the sym_random d=8 model.

Loop: train with CE + lambda*L1 (masked), then permanently zero the 5% of
remaining nonzero weights with smallest |w| (pooled over L and D). At each
sparsity level, also snapshot a no-L1 masked finetune to measure recoverable
accuracy. Stops when finetuned accuracy < 0.3. Appends a section to
tiny_models/sym_random/d8.md.
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

from capacity import generate_facts

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DIR = os.path.join(HERE, "tiny_models", "sym_random")
V_IN, V_OUT, N, M = 16, 8, 256, 8
LAMBDA = 3e-3
PRUNE_FRAC = 0.05          # of *remaining* nonzero weights per round
L1_EPOCHS = 300
FT_EPOCHS = 1500
LR = 1e-2
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def forward(w, x):
    hL = x @ w["L"].T
    return (hL * hL) @ w["down"].T


def accuracy(w, x, targets):
    with torch.no_grad():
        return (forward(w, x).argmax(-1) == targets).float().mean().item()


def train(w, masks, x, targets, epochs, l1=0.0):
    params = list(w.values())
    opt = torch.optim.Adam(params, lr=LR)
    for _ in range(epochs):
        opt.zero_grad(set_to_none=True)
        loss = F.cross_entropy(forward(w, x), targets)
        if l1:
            loss = loss + l1 * sum(v.abs().sum() for v in w.values())
        loss.backward()
        opt.step()
        with torch.no_grad():
            for k in w:
                w[k].mul_(masks[k])


def clone(w):
    return {k: v.detach().clone().requires_grad_(True) for k, v in w.items()}


def main():
    saved = torch.load(os.path.join(DIR, "d8_weights.pt"))
    w = {k: v.to(DEVICE).clone().requires_grad_(True)
         for k, v in {"L": saved["L"], "down": saved["down"]}.items()}
    masks = {k: torch.ones_like(v) for k, v in w.items()}
    inputs, targets = generate_facts(N, V_IN, V_OUT)
    x = torch.cat([F.one_hot(inputs[:, 0], V_IN).float(),
                   F.one_hot(inputs[:, 1], V_IN).float()], dim=-1).to(DEVICE)
    targets = targets.to(DEVICE)

    total = sum(m.numel() for m in masks.values())
    curve = []  # (frac_ablated, acc_pruned, acc_finetuned, nnz_L, nnz_D)
    base = accuracy(w, x, targets)
    print(f"start: acc {base:.3f}, {total} weights")
    curve.append((0.0, base, base,
                  int(masks['L'].sum()), int(masks['down'].sum())))

    for rnd in range(1, 200):
        train(w, masks, x, targets, L1_EPOCHS, l1=LAMBDA)
        # prune 5% of remaining nonzero weights, pooled by |w|
        vals = torch.cat([ (w[k].detach().abs() + (1 - masks[k]) * 1e9
                           ).flatten() for k in ("L", "down")])
        nnz = int(sum(m.sum() for m in masks.values()))
        k_drop = max(1, int(round(PRUNE_FRAC * nnz)))
        thresh = vals.kthvalue(k_drop).values
        for kk in ("L", "down"):
            masks[kk] *= (w[kk].detach().abs() > thresh).float()
        with torch.no_grad():
            for kk in w:
                w[kk].mul_(masks[kk])
        nnz = int(sum(m.sum() for m in masks.values()))
        frac_abl = 1 - nnz / total
        acc_pruned = accuracy(w, x, targets)

        wf = clone(w)
        train(wf, masks, x, targets, FT_EPOCHS, l1=0.0)
        acc_ft = accuracy(wf, x, targets)
        curve.append((frac_abl, acc_pruned, acc_ft,
                      int(masks['L'].sum()), int(masks['down'].sum())))
        print(f"round {rnd:>3}: ablated {frac_abl:5.1%}  "
              f"acc(pruned+L1) {acc_pruned:.3f}  acc(finetuned) {acc_ft:.3f}"
              f"  nnz L/D {int(masks['L'].sum())}/{int(masks['down'].sum())}",
              flush=True)
        if acc_ft < 0.3:
            break

    torch.save({"curve": curve}, os.path.join(DIR, "d8_sparsity_curve.pt"))

    # sparsest level whose finetuned accuracy is still >= 0.9
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
            label="after no-L1 finetune")
    ax.axhline(0.9, color="#8a8a85", ls=":", lw=1)
    ax.text(1, 0.905, "acc 0.9", color="#8a8a85", fontsize=8)
    ax.axvline(best[0] * 100, color="#8a8a85", ls="--", lw=1)
    ax.text(best[0] * 100 - 1.5, 0.35,
            f"sparsest ≥0.9: {best[0]:.0%} ablated\n"
            f"({best[3]}+{best[4]} weights left)",
            color="#555", fontsize=8, ha="right")
    ax.set_xlabel("% of weights ablated (cumulative)")
    ax.set_ylabel("fact accuracy")
    ax.set_ylim(0, 1.05)
    ax.set_title(f"Iterative L1 (λ={LAMBDA}) + prune 5% of remaining, "
                 f"sym_random d=8 (320 weights)")
    ax.legend(frameon=False, fontsize=8, loc="lower left")
    fig.tight_layout()
    img = os.path.join(DIR, "img", "d8_sparsity.png")
    fig.savefig(img, bbox_inches="tight")
    print(f"wrote {img}")

    md_path = os.path.join(DIR, "d8.md")
    with open(md_path) as f:
        content = f.read()
    marker = "## Sparsity (iterative L1 + magnitude pruning)"
    if marker in content:
        content = content[:content.index(marker)].rstrip() + "\n"
    section = [
        "", marker, "",
        f"Loop: {L1_EPOCHS} epochs CE+L1 (λ={LAMBDA}) → permanently zero "
        f"the {PRUNE_FRAC:.0%} smallest-|w| of the *remaining* weights "
        "(pooled over L and D). At each level, the blue curve shows a "
        f"{FT_EPOCHS}-epoch no-L1 finetune of the masked model.",
        "",
        f"Sparsest level keeping finetuned acc ≥ 0.9: **{best[0]:.0%} "
        f"ablated** ({best[3]} of {int(masks['L'].numel())} L weights and "
        f"{best[4]} of {int(masks['down'].numel())} D weights remain).",
        "", "![sparsity](img/d8_sparsity.png)", ""]
    with open(md_path, "w") as f:
        f.write(content + "\n".join(section))
    print(f"appended section to {md_path}")


if __name__ == "__main__":
    main()
