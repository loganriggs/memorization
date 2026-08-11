"""Pairwise Frobenius cosine similarity between the d=8 label slices T_c.

cos(A, B) = trace(A^T B) / (||A||_F ||B||_F)  — i.e. ordinary cosine
similarity of the flattened matrices. Appends a section to
tiny_models/sym_random/d8.md.
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DIR = os.path.join(HERE, "tiny_models", "sym_random")
V_IN, V_OUT = 16, 8


def main():
    w = torch.load(os.path.join(DIR, "d8_weights.pt"))
    L, D = w["L"].cpu(), w["down"].cpu()
    hL = L[:, :V_IN].unsqueeze(2) + L[:, V_IN:].unsqueeze(1)
    T = torch.einsum("cn,nab->cab", D, hL * hL)      # (V_out, V, V)

    flat = T.reshape(V_OUT, -1)
    flat = flat / flat.norm(dim=1, keepdim=True)
    cos = flat @ flat.T                              # (V_out, V_out)

    fig, ax = plt.subplots(figsize=(4.6, 4.2), dpi=150)
    im = ax.imshow(cos.numpy(), cmap="RdBu", vmin=-1, vmax=1)
    ax.set_xticks(range(V_OUT)), ax.set_yticks(range(V_OUT))
    ax.set_xlabel("label"), ax.set_ylabel("label")
    for i in range(V_OUT):
        for j in range(V_OUT):
            v = cos[i, j].item()
            ax.text(j, i, f"{v:+.2f}", ha="center", va="center", fontsize=7,
                    color="white" if abs(v) > 0.6 else "black")
    ax.set_title("Frobenius cosine similarity of label slices $T_c$\n"
                 r"$\cos(A,B)=\mathrm{tr}(A^\top B)\,/\,\|A\|_F\|B\|_F$",
                 fontsize=9)
    fig.colorbar(im, ax=ax, shrink=0.85)
    fig.tight_layout()
    img = os.path.join(DIR, "img", "d8_cossim.png")
    fig.savefig(img, bbox_inches="tight")
    print(f"wrote {img}")

    off = cos[~torch.eye(V_OUT, dtype=bool)]
    print(f"off-diagonal: mean {off.mean():+.3f}, min {off.min():+.3f}, "
          f"max {off.max():+.3f}")

    md_path = os.path.join(DIR, "d8.md")
    with open(md_path) as f:
        content = f.read()
    marker = "## Label-slice cosine similarity"
    if marker in content:
        content = content[:content.index(marker)].rstrip() + "\n"
    section = [
        "", marker, "",
        "Frobenius cosine similarity between the full logit slices: "
        "cos(A,B) = tr(AᵀB)/(‖A‖_F‖B‖_F), i.e. cosine similarity of the "
        "flattened 16×16 matrices.",
        "", "![cossim](img/d8_cossim.png)", ""]
    with open(md_path, "w") as f:
        f.write(content + "\n".join(section))
    print(f"appended section to {md_path}")


if __name__ == "__main__":
    main()
