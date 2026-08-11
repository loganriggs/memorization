"""Weight quantization of the sym_random d=8 showcase model.

Rounds each weight matrix to the nearest multiple of a step size (D only,
L only, or both), and measures fact accuracy. Writes
tiny_models/sym_random/img/d8_quantization.png and appends a section to
tiny_models/sym_random/d8.md.
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from capacity import generate_facts

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DIR = os.path.join(HERE, "tiny_models", "sym_random")
D_MODEL = 8
V_IN, V_OUT, N = 16, 8, 256
STEPS = [round(s, 3) for s in
         [0.02 * k for k in range(1, 51)]]  # 0.02 .. 1.00

C = {"D only": "#2a78d6", "L only": "#eb6834", "both": "#1baf7a"}


def accuracy(L, D, inputs, targets):
    hL = L[:, inputs[:, 0]] + L[:, V_IN + inputs[:, 1]]   # (m, N)
    logits = D @ (hL * hL)                                # (V_out, N)
    return (logits.argmax(0) == targets).float().mean().item()


def q(w, step):
    return torch.round(w / step) * step


def main():
    w = torch.load(os.path.join(DIR, "d8_weights.pt"))
    L, D = w["L"].cpu(), w["down"].cpu()
    inputs, targets = generate_facts(N, V_IN, V_OUT)
    inputs, targets = inputs.cpu(), targets.cpu()

    base = accuracy(L, D, inputs, targets)
    print(f"unquantized accuracy: {base:.4f}")
    print(f"weight ranges: L in [{L.min():.2f}, {L.max():.2f}], "
          f"D in [{D.min():.2f}, {D.max():.2f}]")

    results = {name: [] for name in C}
    levels = {name: [] for name in C}
    for step in STEPS:
        for name, (Lq, Dq) in {
            "D only": (L, q(D, step)),
            "L only": (q(L, step), D),
            "both": (q(L, step), q(D, step)),
        }.items():
            a = accuracy(Lq, Dq, inputs, targets)
            n_lvl = max(len(torch.unique(q(L, step))) if "L" in name or
                        name == "both" else 0,
                        len(torch.unique(q(D, step))) if "D" in name or
                        name == "both" else 0)
            results[name].append(a)
            levels[name].append(n_lvl)

    plt.rcParams.update({
        "figure.dpi": 150, "axes.spines.top": False,
        "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.25,
        "font.size": 9, "axes.titlesize": 10,
    })
    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    for name, accs in results.items():
        ax.plot(STEPS, accs, "-", color=C[name], lw=2, label=name)
    ax.axhline(base, color="#8a8a85", ls="--", lw=1.2)
    ax.text(0.98, base + 0.012, f"unquantized ({base:.2f})",
            color="#8a8a85", fontsize=8, ha="right")
    ax.axhline(1 / V_OUT, color="#8a8a85", ls=":", lw=1)
    ax.text(0.98, 1 / V_OUT + 0.012, "chance (1/8)", color="#8a8a85",
            fontsize=8, ha="right")
    ax.set_xlabel("rounded to nearest X (0 = unquantized, coarser →)")
    ax.set_ylabel("fact accuracy")
    ax.set_ylim(0, 1.05)
    ax.set_xlim(0, 1.02)
    ax.set_title(f"Quantizing sym_random d={D_MODEL} weights "
                 f"(n={N} facts, m={D_MODEL})")
    ax.legend(frameon=False, fontsize=8, loc="lower left")
    fig.tight_layout()
    img_path = os.path.join(DIR, "img", "d8_quantization.png")
    fig.savefig(img_path, bbox_inches="tight")
    print(f"wrote {img_path}")

    # ── weights at 5 rounding levels (both matrices rounded) ────────────────
    show_steps = [1.0, 0.5, 0.3, 0.1, 0.0]   # coarsest → unquantized
    vmax_L = float(L.abs().max())
    vmax_D = float(D.abs().max())
    fig2, axes = plt.subplots(len(show_steps), 2, figsize=(13, 2.1 * 5),
                              gridspec_kw={"width_ratios": [4, 1]})
    for row, X in enumerate(show_steps):
        Lq = L if X == 0 else q(L, X)
        Dq = D if X == 0 else q(D, X)
        a = accuracy(Lq, Dq, inputs, targets)
        n_lvl = len(torch.unique(Lq))
        for ax, mat, vmax, name in ((axes[row][0], Lq.numpy(), vmax_L, "L"),
                                    (axes[row][1], Dq.numpy(), vmax_D, "D")):
            ax.imshow(mat, cmap="RdBu", vmin=-vmax, vmax=vmax)
            for i in range(mat.shape[0]):
                for j in range(mat.shape[1]):
                    ax.text(j, i, f"{mat[i, j]:.1f}", ha="center",
                            va="center", fontsize=4.6,
                            color="white" if abs(mat[i, j]) > 0.6 * vmax
                            else "black")
            ax.set_xticks([]), ax.set_yticks([])
        label = "unquantized" if X == 0 else f"X = {X:g}"
        axes[row][0].set_title(
            f"{label}   (acc {a:.3f}, {n_lvl} distinct L values)",
            fontsize=9, loc="left")
        axes[row][0].axvline(V_IN - 0.5, color="black", lw=1.0)
        axes[row][1].set_title("D", fontsize=9, loc="left")
    fig2.suptitle("Weights rounded to nearest X (both matrices), "
                  "coarsest → unquantized; color scale fixed per matrix",
                  y=0.995)
    fig2.tight_layout()
    img2_path = os.path.join(DIR, "img", "d8_quantized_weights.png")
    fig2.savefig(img2_path, bbox_inches="tight")
    print(f"wrote {img2_path}")

    # append section to d8.md (idempotent: skip if already present)
    md_path = os.path.join(DIR, "d8.md")
    with open(md_path) as f:
        content = f.read()
    marker = "## Quantization"
    if marker in content:
        content = content[:content.index(marker)].rstrip() + "\n"
    rows = ["", marker, "",
            "Every weight in the chosen matrix is **rounded to the nearest "
            "multiple of X** (e.g. X=0.2 sends 0.47→0.4, −0.31→−0.4; X=1 "
            "rounds to integers), one matrix at a time; accuracy on the 256 "
            f"stored facts (unquantized: {base:.3f}).",
            "",
            "| rounded to nearest X | D only | L only | both |",
            "|---|---|---|---|"]
    for i, step in enumerate(STEPS):
        if round(step * 10, 6) % 1 == 0:  # table shows every 0.1
            rows.append(f"| {step} | {results['D only'][i]:.3f} "
                        f"| {results['L only'][i]:.3f} "
                        f"| {results['both'][i]:.3f} |")
    rows += ["", "![quantization](img/d8_quantization.png)", "",
             "The weights themselves at five rounding levels (both matrices "
             "rounded; black divider = position-1 | position-2 blocks of L):",
             "", "![quantized weights](img/d8_quantized_weights.png)", ""]
    with open(md_path, "w") as f:
        f.write(content + "\n".join(rows))
    print(f"appended section to {md_path}")


if __name__ == "__main__":
    main()
