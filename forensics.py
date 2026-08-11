"""Gradient/margin forensics for sym-bilinear models (task 6 / H5).

Given (L, D) and the fact set, reports:
  - per-fact margins (and which facts are misclassified)
  - CE-loss gradient wrt L: |grad| by token-column and by neuron-row
  - token-level failure concentration (how many failed facts touch each token)

Run as a script: analyzes the d=8 construction (h9b + fast repair) next to
the trained dense d=8 showcase model; writes results/fig_forensics_d8.png.
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

from capacity import generate_facts, RESULTS_DIR


def forensics(L, D, inputs, targets, v_in):
    L = L.detach().clone().requires_grad_(True)
    x = torch.cat([F.one_hot(inputs[:, 0], v_in).float(),
                   F.one_hot(inputs[:, 1], v_in).float()],
                  dim=-1).to(L.dtype)
    h = (x @ L.T) ** 2
    logits = h @ D.T.to(L.dtype)
    loss = F.cross_entropy(logits, targets)
    loss.backward()
    grad = L.grad.detach()
    with torch.no_grad():
        own = logits.gather(1, targets.unsqueeze(1)).squeeze(1)
        other = logits.scatter(1, targets.unsqueeze(1), float("-inf"))
        margins = own - other.max(dim=1).values
    return {
        "margins": margins.detach(),
        "failed": (margins <= 0).nonzero().flatten(),
        "grad_by_col": grad.abs().sum(0),      # (2*v_in,) token columns
        "grad_by_row": grad.abs().sum(1),      # (m,) neurons
        "loss": float(loss),
    }


def token_failure_counts(inputs, failed_idx, v_in):
    c1 = torch.zeros(v_in)
    c2 = torch.zeros(v_in)
    for i in failed_idx.tolist():
        c1[inputs[i, 0]] += 1
        c2[inputs[i, 1]] += 1
    return c1, c2


def main():
    torch.set_num_threads(4)
    d = 8
    v_in, v_out, n = 2 * d, d, 4 * d * d
    inputs, targets = generate_facts(n, v_in, v_out)
    inputs, targets = inputs.cpu(), targets.cpu()

    # constructed model
    from h12b_repair import h9b_solve
    from h12c_fast import fast_repair
    X = torch.cat([F.one_hot(inputs[:, 0], v_in).float(),
                   F.one_hot(inputs[:, 1], v_in).float()], dim=-1).double()
    _, L0 = h9b_solve(X, targets, v_out)
    Dc = -torch.eye(v_out, dtype=torch.float64)
    acc, Lc = fast_repair(L0, Dc, X, targets, inputs, v_in,
                          passes=12, n_cand=25)
    fc = forensics(Lc, Dc, inputs, targets, v_in)

    # trained dense showcase (acc 1.0 at this n)
    w = torch.load("tiny_models/sym_random/d8_weights.pt")
    ft = forensics(w["L"].double(), w["down"].double(), inputs, targets, v_in)

    c1, c2 = token_failure_counts(inputs, fc["failed"], v_in)
    print(f"constructed acc {acc:.3f}, {len(fc['failed'])} failures")
    print(f"failed-fact token counts pos1: {c1.int().tolist()}")
    print(f"failed-fact token counts pos2: {c2.int().tolist()}")
    print(f"grad-by-neuron (constructed): "
          f"{[round(v,2) for v in fc['grad_by_row'].tolist()]}")

    fig, axes = plt.subplots(1, 3, figsize=(11, 3.2))
    plt.rcParams.update({"font.size": 9})
    bins = torch.linspace(-3, 8, 45)
    axes[0].hist(ft["margins"].numpy(), bins=bins.numpy(), alpha=0.6,
                 label="trained (acc 1.0)", color="#2a78d6", density=True)
    axes[0].hist(fc["margins"].numpy(), bins=bins.numpy(), alpha=0.6,
                 label=f"constructed (acc {acc:.2f})", color="#eb6834",
                 density=True)
    axes[0].axvline(0, color="k", lw=1)
    axes[0].set_title("per-fact margins")
    axes[0].legend(frameon=False, fontsize=8)

    axes[1].bar(range(2 * v_in), fc["grad_by_col"].numpy(), color="#eb6834")
    axes[1].axvline(v_in - 0.5, color="k", lw=1)
    axes[1].set_title("constructed: |∂CE/∂L| by token column\n"
                      "(left pos-1 | right pos-2)")

    width = 0.4
    axes[2].bar([i - width / 2 for i in range(v_in)], c1.numpy(), width,
                label="as t1", color="#2a78d6")
    axes[2].bar([i + width / 2 for i in range(v_in)], c2.numpy(), width,
                label="as t2", color="#eb6834")
    axes[2].set_title("failed facts per token")
    axes[2].legend(frameon=False, fontsize=8)
    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    fig.tight_layout()
    out = os.path.join(RESULTS_DIR, "fig_forensics_d8.png")
    fig.savefig(out, bbox_inches="tight", dpi=150)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
