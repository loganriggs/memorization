"""Analysis of how the trained bilinear layer stores facts, vs the ReLU MLP.

Trains one model per arch (init seed 0) at d=32 at that arch's measured
acc>=0.9 capacity, then produces:
  - fig_weights.png      weight histograms (MLP: W, D; bilinear: L, R, D)
  - fig_activations.png  hidden-activation distributions + per-fact
                         neuron-contribution concentration (participation ratio)
  - fig_spectra.png      singular-value spectra + effective rank of the
                         per-label cross-interaction matrices M_c, with an
                         imshow of one M_c against the label's fact cells
Weights are saved to results/analysis_models.pt.
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from capacity import (generate_facts, _init_weights, _forward, RESULTS_DIR,
                      N_EPOCHS, LR, PATIENCE)

# dataviz reference palette (light mode)
C_MLP, C_BIL, C_SWI, C_HC = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
GRAY = "#8a8a85"

D_ANALYSIS = 32
plt.rcParams.update({
    "figure.dpi": 150, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.6,
    "font.size": 9, "axes.titlesize": 10,
})


def train_single(arch, d, m, n_facts, n_epochs=N_EPOCHS, target=0.9,
                 label_mode="random"):
    """Train seeds 0..10 in turn (same protocol as capacity.train_attempts);
    return the first model reaching `target`, else the best one."""
    best_overall = (None, 0.0, None)
    for seed in range(11):
        state, acc, dat = _train_one_seed(arch, d, m, n_facts, seed, n_epochs,
                                          label_mode)
        if acc > best_overall[1]:
            best_overall = (state, acc, dat)
        if acc >= target:
            break
    state, acc, dat = best_overall
    print(f"{arch} d={d} m={m} n={n_facts}: selected seed acc {acc:.4f}")
    return state, acc, dat


def _train_one_seed(arch, d, m, n_facts, seed, n_epochs=N_EPOCHS,
                    label_mode="random"):
    v_in, v_out = 2 * d, d
    inputs, targets = generate_facts(n_facts, v_in, v_out,
                                     label_mode=label_mode)
    x = torch.cat([F.one_hot(inputs[:, 0], v_in).float(),
                   F.one_hot(inputs[:, 1], v_in).float()], dim=-1)
    # _init_weights seeds slot s with init seed s; slice out this seed's init.
    ws = _init_weights(arch, d, m, seed + 1)
    weights = {k: v.detach()[seed:seed + 1].clone().requires_grad_(True)
               for k, v in ws.items()}
    opt = torch.optim.Adam(list(weights.values()), lr=LR)
    best, since, best_state = 0.0, 0, None
    for epoch in range(1, n_epochs + 1):
        opt.zero_grad(set_to_none=True)
        logits = _forward(arch, weights, x)
        loss = F.cross_entropy(logits[0], targets)
        loss.backward()
        opt.step()
        acc = (logits[0].argmax(-1) == targets).float().mean().item()
        if acc > best:
            best, since = acc, 0
            best_state = {k: w.detach()[0].clone() for k, w in weights.items()}
        else:
            since += 1
        if best >= 1.0 or since >= PATIENCE:
            break
    print(f"{arch} d={d} m={m} n={n_facts}: best acc {best:.4f} ({epoch} epochs)")
    return best_state, best, (inputs, targets, x)


def participation_ratio(c):
    """Effective number of contributing neurons: (sum|c|)^2 / sum c^2."""
    a = c.abs()
    return (a.sum(-1) ** 2) / (c ** 2).sum(-1).clamp_min(1e-12)


def cross_matrices(w, d):
    """Per-label cross-position interaction M_c, (V_out, V_in, V_in)."""
    v_in = 2 * d
    L1, L2 = w["L"][:, :v_in], w["L"][:, v_in:]
    R1, R2 = w["R"][:, :v_in], w["R"][:, v_in:]
    D = w["down"]
    # M_c[a,b] = sum_n D[c,n] * (L1[n,a] R2[n,b] + R1[n,a] L2[n,b])
    return (torch.einsum("cn,na,nb->cab", D, L1, R2)
            + torch.einsum("cn,na,nb->cab", D, R1, L2))


def main():
    torch.manual_seed(0)
    d = D_ANALYSIS
    caps = {}  # arch -> (m, n_facts at 0.9 capacity)
    import json
    with open(os.path.join(RESULTS_DIR, "capacity.jsonl")) as f:
        for line in f:
            r = json.loads(line)
            if r["d"] == d and r["threshold"] == 0.9:
                caps[r["arch"]] = (r["m"], r["max_facts"])

    models, accs, data = {}, {}, {}
    for arch in ("mlp", "bilinear"):
        m, n = caps[arch]
        models[arch], accs[arch], data[arch] = train_single(arch, d, m, n)
    torch.save({"models": {k: {kk: vv.cpu() for kk, vv in v.items()}
                           for k, v in models.items()},
                "caps": caps, "accs": accs, "d": d},
               os.path.join(RESULTS_DIR, "analysis_models.pt"))

    # ── Figure: weight histograms ────────────────────────────────────────────
    fig, axes = plt.subplots(2, 3, figsize=(9.5, 5))
    panels = [
        ("mlp", "up", "MLP input weights W", C_MLP, axes[0, 0]),
        ("mlp", "down", "MLP output weights D", C_MLP, axes[0, 1]),
        ("bilinear", "L", "Bilinear L", C_BIL, axes[1, 0]),
        ("bilinear", "R", "Bilinear R", C_BIL, axes[1, 1]),
        ("bilinear", "down", "Bilinear output D", C_BIL, axes[1, 2]),
    ]
    axes[0, 2].axis("off")
    axes[0, 2].text(0.05, 0.6,
                    f"d={d}, trained at each arch's\nown acc≥0.9 capacity:\n"
                    f"MLP: n={caps['mlp'][1]}, m={caps['mlp'][0]}\n"
                    f"bilinear: n={caps['bilinear'][1]}, m={caps['bilinear'][0]}",
                    fontsize=9, va="center")
    for arch, key, title, color, ax in panels:
        w = models[arch][key].cpu().numpy().ravel()
        ax.hist(w, bins=80, color=color, alpha=0.85)
        ax.set_title(title)
        ax.set_yscale("log")
    fig.suptitle("Trained weight distributions (log counts)", y=1.0)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "fig_weights.png"),
                bbox_inches="tight")
    plt.close(fig)

    # ── Figure: activations + contribution concentration ────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.2))
    for arch, color in (("mlp", C_MLP), ("bilinear", C_BIL)):
        w = models[arch]
        _, _, x = data[arch]
        wb = {k: v.unsqueeze(0) for k, v in w.items()}
        with torch.no_grad():
            if arch == "mlp":
                h = torch.relu(torch.einsum(
                    "ni,smi->snm", x, wb["up"]))[0]
            else:
                h = (torch.einsum("ni,smi->snm", x, wb["L"])
                     * torch.einsum("ni,smi->snm", x, wb["R"]))[0]
        hn = h.cpu().numpy().ravel()
        axes[0].hist(hn, bins=120, color=color, alpha=0.6, label=arch,
                     density=True)
        frac_zero = (np.abs(hn) < 1e-7).mean()
        print(f"{arch}: fraction of exactly-zero activations {frac_zero:.2%}")

        inputs, targets, _ = data[arch]
        contrib = h * w["down"][targets]          # (N, m)
        pr = participation_ratio(contrib).cpu().numpy()
        axes[1].hist(pr, bins=60, color=color, alpha=0.6, label=arch,
                     density=True)
        # top-k coverage: fraction of correct logit from top-k |contribution|
        srt = contrib.abs().sort(dim=-1, descending=True).values
        cum = srt.cumsum(-1) / srt.sum(-1, keepdim=True)
        ks = np.arange(1, cum.shape[1] + 1)
        axes[2].plot(ks, cum.mean(0).cpu().numpy(), color=color, lw=2,
                     label=arch)
    axes[0].set_title("Hidden activation values")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("activation")
    axes[1].set_title("Per-fact effective # contributing neurons\n"
                      "(participation ratio of contributions to correct logit)")
    axes[1].set_xlabel("effective neurons")
    axes[2].set_title("Mean cumulative |contribution| coverage")
    axes[2].set_xlabel("top-k neurons")
    axes[2].set_ylim(0, 1.02)
    for ax in axes:
        ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "fig_activations.png"),
                bbox_inches="tight")
    plt.close(fig)

    # ── Figure: per-label cross-interaction spectra (bilinear) ──────────────
    w = models["bilinear"]
    M = cross_matrices(w, d)                      # (V_out, V_in, V_in)
    svals = torch.linalg.svdvals(M)               # (V_out, V_in)
    sv = svals.cpu().numpy()
    eff_rank = (sv.sum(1) ** 2) / (sv ** 2).sum(1)

    inputs, targets, _ = data["bilinear"]
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.4))
    for c in range(sv.shape[0]):
        axes[0].plot(np.arange(1, sv.shape[1] + 1), sv[c] / sv[c, 0],
                     color=C_BIL, alpha=0.25, lw=0.8)
    axes[0].plot(np.arange(1, sv.shape[1] + 1),
                 (sv / sv[:, :1]).mean(0), color="#7a3413", lw=2,
                 label="mean over labels")
    axes[0].axvline(2 * w["L"].shape[0], color=GRAY, ls="--", lw=1)
    axes[0].text(2 * w["L"].shape[0] * 1.05, 0.75, "2m bound", color=GRAY,
                 fontsize=8)
    axes[0].set_title("Singular value spectra of $M_c$ (normalized)")
    axes[0].set_xlabel("index")
    axes[0].legend(frameon=False)

    axes[1].hist(eff_rank, bins=20, color=C_BIL, alpha=0.85)
    axes[1].set_title("Effective rank of $M_c$ per label")
    axes[1].set_xlabel("participation ratio of singular values")

    c0 = 0
    Mc = M[c0].cpu().numpy()
    vmax = np.abs(Mc).max()
    im = axes[2].imshow(Mc, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    mask = (targets == c0).cpu().numpy()
    pts = inputs[mask].cpu().numpy()
    axes[2].scatter(pts[:, 1], pts[:, 0], s=6, facecolors="none",
                    edgecolors="#1a1a19", linewidths=0.6)
    axes[2].set_title(f"$M_{{c={c0}}}$ with label-{c0} facts circled")
    axes[2].set_xlabel("token 2")
    axes[2].set_ylabel("token 1")
    axes[2].grid(False)
    fig.colorbar(im, ax=axes[2], shrink=0.8)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "fig_spectra.png"),
                bbox_inches="tight")
    plt.close(fig)

    print("effective rank of M_c: mean %.1f (V_in=%d, 2m=%d)"
          % (eff_rank.mean(), 2 * d, 2 * w["L"].shape[0]))
    print("figures written to results/")


if __name__ == "__main__":
    main()
