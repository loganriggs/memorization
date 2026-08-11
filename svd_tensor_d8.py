"""Per-label SVD of the sym_random d=8 logit tensor.

For each label c, take the V_in x V_in matrix T[:, :, c] (the model's logit
for label c over every input pair), SVD it, and plot: the full slice, the
top-3 rank-1 components individually, and the cumulative rank-1/2/3
reconstructions. Appends a section to tiny_models/sym_random/d8.md.
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from capacity import generate_facts

HERE = os.path.dirname(os.path.abspath(__file__))
DIR = os.path.join(HERE, "tiny_models", "sym_random")
V_IN, V_OUT, N = 16, 8, 256


def main():
    w = torch.load(os.path.join(DIR, "d8_weights.pt"))
    L, D = w["L"].cpu(), w["down"].cpu()
    inputs, targets = generate_facts(N, V_IN, V_OUT)
    inputs, targets = inputs.cpu(), targets.cpu()

    hL = L[:, :V_IN].unsqueeze(2) + L[:, V_IN:].unsqueeze(1)   # (m, V, V)
    T = torch.einsum("cn,nab->abc", D, hL * hL)                # (V, V, V_out)

    cols = ["full $T_c$", "comp 1", "comp 2", "comp 3",
            "cum 1", "cum 1–2", "cum 1–3"]
    fig, axes = plt.subplots(V_OUT, len(cols),
                             figsize=(2.0 * len(cols), 2.05 * V_OUT))
    plt.rcParams.update({"font.size": 8})
    for c in range(V_OUT):
        Tc = T[:, :, c]
        U, S, Vh = torch.linalg.svd(Tc)
        comps = [S[k] * torch.outer(U[:, k], Vh[k]) for k in range(3)]
        cums = [sum(comps[:k + 1]) for k in range(3)]
        energy = (S ** 2) / (S ** 2).sum()
        panels = [Tc] + comps + cums
        vmax = float(Tc.abs().max())
        for j, (ax, mat) in enumerate(zip(axes[c], panels)):
            ax.imshow(mat.numpy(), cmap="RdBu", vmin=-vmax, vmax=vmax)
            ax.set_xticks([]), ax.set_yticks([])
            if j == 0:
                mask = (targets == c).numpy()
                pts = inputs[mask].numpy()
                ax.scatter(pts[:, 1], pts[:, 0], s=6, facecolors="none",
                           edgecolors="black", linewidths=0.5)
                ax.set_ylabel(f"label {c}", fontsize=9)
                ax.set_title(cols[0], fontsize=8)
            elif 1 <= j <= 3:
                ax.set_title(f"{cols[j]}  σ={S[j-1]:.0f} "
                             f"({energy[j-1]:.0%})", fontsize=7)
            else:
                k = j - 3
                ax.set_title(f"{cols[j]}  ({energy[:k].sum():.0%} energy)",
                             fontsize=7)
    fig.suptitle("SVD of each label's logit slice $T_c$ — top-3 rank-1 "
                 "components, individual then cumulative\n"
                 "(color scale per row = full slice's; ○ = stored facts of "
                 "that label)", y=0.998)
    fig.tight_layout(rect=[0, 0, 1, 0.985])
    img = os.path.join(DIR, "img", "d8_svd_tensor.png")
    fig.savefig(img, bbox_inches="tight")
    print(f"wrote {img}")

    # effective rank summary
    for c in range(V_OUT):
        S = torch.linalg.svdvals(T[:, :, c])
        er = float((S.sum() ** 2) / (S ** 2).sum())
        e3 = float((S[:3] ** 2).sum() / (S ** 2).sum())
        print(f"label {c}: eff rank {er:.1f}, top-3 energy {e3:.0%}")

    md_path = os.path.join(DIR, "d8.md")
    with open(md_path) as f:
        content = f.read()
    marker = "## Logit-slice SVD"
    if marker in content:
        content = content[:content.index(marker)].rstrip() + "\n"
    section = [
        "", marker, "",
        "SVD of each label's V_in × V_in logit slice: top-3 rank-1 "
        "components shown individually (σ·u·vᵀ), then cumulatively. Titles "
        "give each component's singular value and share of the slice's "
        "squared-singular-value energy.",
        "", "![svd](img/d8_svd_tensor.png)", ""]
    with open(md_path, "w") as f:
        f.write(content + "\n".join(section))
    print(f"appended section to {md_path}")


if __name__ == "__main__":
    main()
