"""Re-run the d=8 iterative-L1 pruning loop, snapshot the finetuned model at
~30%, ~37%, and ~62% ablation, and render weights + full logit tensor for
each. Appends a subsection to tiny_models/sym_random/d8.md."""

import os

import torch
import torch.nn.functional as F

from capacity import generate_facts
from sparsity_d8 import (forward, accuracy, train, clone,
                         V_IN, V_OUT, N, LAMBDA, PRUNE_FRAC, L1_EPOCHS,
                         FT_EPOCHS, DEVICE, DIR)
import tiny_report

SNAP_ROUNDS = {7: "30", 9: "37", 19: "62"}   # round -> label used in filenames


def main():
    saved = torch.load(os.path.join(DIR, "d8_weights.pt"))
    w = {k: saved[k].to(DEVICE).clone().requires_grad_(True)
         for k in ("L", "down")}
    masks = {k: torch.ones_like(v) for k, v in w.items()}
    inputs, targets = generate_facts(N, V_IN, V_OUT)
    x = torch.cat([F.one_hot(inputs[:, 0], V_IN).float(),
                   F.one_hot(inputs[:, 1], V_IN).float()], dim=-1).to(DEVICE)
    targets = targets.to(DEVICE)
    total = sum(m.numel() for m in masks.values())

    snaps = {}
    for rnd in range(1, max(SNAP_ROUNDS) + 1):
        train(w, masks, x, targets, L1_EPOCHS, l1=LAMBDA)
        vals = torch.cat([(w[k].detach().abs() + (1 - masks[k]) * 1e9
                           ).flatten() for k in ("L", "down")])
        nnz = int(sum(m.sum() for m in masks.values()))
        k_drop = max(1, int(round(PRUNE_FRAC * nnz)))
        thresh = vals.kthvalue(k_drop).values
        for kk in ("L", "down"):
            masks[kk] *= (w[kk].detach().abs() > thresh).float()
        with torch.no_grad():
            for kk in w:
                w[kk].mul_(masks[kk])
        if rnd in SNAP_ROUNDS:
            wf = clone(w)
            train(wf, masks, x, targets, FT_EPOCHS, l1=0.0)
            frac = 1 - int(sum(m.sum() for m in masks.values())) / total
            acc = accuracy(wf, x, targets)
            snaps[rnd] = ({k: v.detach().cpu() for k, v in wf.items()},
                          frac, acc)
            print(f"snapshot round {rnd}: {frac:.1%} ablated, "
                  f"finetuned acc {acc:.3f}", flush=True)

    md_lines = ["", "### Pruned models: weights and logit tensors", ""]
    for rnd, (wf, frac, acc) in sorted(snaps.items()):
        tag = SNAP_ROUNDS[rnd]
        nnz_L = int((wf["L"] != 0).sum())
        nnz_D = int((wf["down"] != 0).sum())
        img_w = f"img/d8_pruned_{tag}_weights.png"
        img_t = f"img/d8_pruned_{tag}_tensor.png"
        tiny_report.fig_weights(wf, 8, os.path.join(DIR, img_w))
        with torch.no_grad():
            L = wf["L"]
            hL = L[:, :V_IN].unsqueeze(2) + L[:, V_IN:].unsqueeze(1)
            T = torch.einsum("cn,nab->abc", wf["down"], hL * hL)
            pred = T[inputs[:, 0].cpu(), inputs[:, 1].cpu()].argmax(-1)
        tiny_report.fig_tensor(T, (inputs.cpu(), targets.cpu(), pred),
                               os.path.join(DIR, img_t), 8)
        md_lines += [
            f"#### {frac:.0%} ablated (finetuned acc {acc:.3f}; "
            f"{nnz_L}/{wf['L'].numel()} L and {nnz_D}/{wf['down'].numel()} "
            "D weights remain)",
            "", f"![weights]({img_w})", "", f"![tensor]({img_t})", ""]

    md_path = os.path.join(DIR, "d8.md")
    with open(md_path) as f:
        content = f.read()
    marker = "### Pruned models: weights and logit tensors"
    if marker in content:
        content = content[:content.index(marker)].rstrip() + "\n"
    with open(md_path, "w") as f:
        f.write(content + "\n".join(md_lines))
    print(f"appended to {md_path}")


if __name__ == "__main__":
    main()
