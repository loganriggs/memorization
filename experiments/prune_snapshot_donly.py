"""Snapshot the D-only pruning run at the acc>=0.9 frontier (round 17,
57.8% of D ablated) and render L (shared, R = L) and D. Appends the figure
to the 'Sparsity: D only' section of tiny_models/sym_random/d8.md."""

import os

import torch
import torch.nn.functional as F

from capacity import generate_facts
from sparsity_d8 import (accuracy, clone, V_IN, V_OUT, N, LAMBDA,
                         PRUNE_FRAC, L1_EPOCHS, FT_EPOCHS, DEVICE, DIR)
from sparsity_d8_donly import train_d_penalty
import tiny_report

SNAP_ROUND = 17


def main():
    saved = torch.load(os.path.join(DIR, "d8_weights.pt"))
    w = {k: saved[k].to(DEVICE).clone().requires_grad_(True)
         for k in ("L", "down")}
    mask_d = torch.ones_like(w["down"])
    inputs, targets = generate_facts(N, V_IN, V_OUT)
    x = torch.cat([F.one_hot(inputs[:, 0], V_IN).float(),
                   F.one_hot(inputs[:, 1], V_IN).float()], dim=-1).to(DEVICE)
    targets = targets.to(DEVICE)

    for rnd in range(1, SNAP_ROUND + 1):
        train_d_penalty(w, mask_d, x, targets, L1_EPOCHS, l1=LAMBDA)
        vals = (w["down"].detach().abs() + (1 - mask_d) * 1e9).flatten()
        k_drop = max(1, int(round(PRUNE_FRAC * int(mask_d.sum()))))
        thresh = vals.kthvalue(k_drop).values
        mask_d *= (w["down"].detach().abs() > thresh).float()
        with torch.no_grad():
            w["down"].mul_(mask_d)

    wf = clone(w)
    train_d_penalty(wf, mask_d, x, targets, FT_EPOCHS, l1=0.0)
    acc = accuracy(wf, x, targets)
    nnz = int(mask_d.sum())
    per_label = mask_d.sum(dim=1).int().tolist()
    print(f"round {SNAP_ROUND}: {nnz} D weights left, finetuned acc {acc:.3f}")
    print(f"nonzero D weights per label: {per_label}")

    wf_cpu = {k: v.detach().cpu() for k, v in wf.items()}
    img = "img/d8_donly_frontier_weights.png"
    tiny_report.fig_weights(wf_cpu, 8, os.path.join(DIR, img))
    torch.save(wf_cpu, os.path.join(DIR, "d8_donly_frontier_weights.pt"))

    md_path = os.path.join(DIR, "d8.md")
    with open(md_path) as f:
        content = f.read()
    marker = "### D-only frontier model"
    if marker in content:
        content = content[:content.index(marker)].rstrip() + "\n"
    section = [
        "", marker, "",
        f"The model at the acc ≥ 0.9 frontier of D-only pruning "
        f"(round {SNAP_ROUND}: {nnz}/64 D weights, finetuned acc {acc:.3f}; "
        f"nonzero D weights per label: {per_label}). L is dense — never "
        "pruned — but has been retrained to suit the sparse readout.",
        "", f"![frontier weights]({img})", ""]
    with open(md_path, "w") as f:
        f.write(content + "\n".join(section))
    print(f"appended to {md_path}")


if __name__ == "__main__":
    main()
