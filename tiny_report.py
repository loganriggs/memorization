"""Tiny bilinear models (d = 1..8): capacity, weights, tensor, worked examples.

For each d: binary-search max facts at acc >= 0.9 (11 seeds, any — the same
protocol as the main sweep), train a showcase model at that capacity (first
seed to reach 0.9, else best of 11), and write tiny_models/d{d}.md containing:
  - capacity + showcase-model stats
  - annotated imshows of L, R (position blocks marked) and D
  - the model's full 3rd-order logit tensor T[t1, t2, c] as one panel per
    label, stored facts circled (X = fact the model gets wrong)
  - worked per-neuron computations for examples from each success tier
    (sorted by margin = correct logit - best wrong logit)

Usage: python tiny_report.py [--dvals 1,2,3,4,5,6,7,8]
"""

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from capacity import (GridCache, find_max_facts, generate_facts, width_for,
                      param_count)
from analysis import train_single

C_BIL = "#eb6834"
HERE = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.join(HERE, "tiny_models")
# Set per run in main(): tiny_models/<setting>/ with img/ inside.
OUT_DIR = BASE_DIR
IMG_DIR = os.path.join(OUT_DIR, "img")

plt.rcParams.update({
    "figure.dpi": 150, "font.size": 9, "axes.titlesize": 10,
})


def annotated_imshow(ax, mat, title, xlabel="", ylabel="", annotate=True):
    vmax = max(np.abs(mat).max(), 1e-9)
    im = ax.imshow(mat, cmap="RdBu", vmin=-vmax, vmax=vmax)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    xstep = 1 if mat.shape[1] <= 20 else 4
    ystep = 1 if mat.shape[0] <= 20 else 4
    ax.set_xticks(range(0, mat.shape[1], xstep))
    ax.set_yticks(range(0, mat.shape[0], ystep))
    ax.tick_params(length=0, labelsize=6)
    if annotate and mat.size <= 400:
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                ax.text(j, i, f"{mat[i, j]:.1f}", ha="center", va="center",
                        fontsize=5 if mat.size > 150 else 6.5,
                        color="white" if abs(mat[i, j]) > 0.6 * vmax else "black")
    return im


def fig_weights(w, d, path):
    v_in = 2 * d
    names = ["L", "R"] if "R" in w else ["L"]
    mats = [w[k].cpu().numpy() for k in names]
    D = w["down"].cpu().numpy()
    m = mats[0].shape[0]
    n_panels = len(names) + 1
    fig, axes = plt.subplots(
        1, n_panels,
        figsize=(min(16, 2.2 + 0.42 * 4 * d) * 0.8 * n_panels,
                 1.2 + 0.42 * max(m, d)),
        gridspec_kw={"width_ratios": [4 * d] * len(names) + [m]},
        squeeze=False)
    axes = axes[0]
    for ax, mat, name in zip(axes, mats, names):
        title = (f"{name} = folded token embeddings → neurons  "
                 f"(m={m} × 2·V_in={4*d})")
        if "R" not in w:
            title += "   [shared: R = L]"
        annotated_imshow(ax, mat, title,
                         xlabel="col = token's embedding: tok@pos1 | tok@pos2",
                         ylabel="neuron")
        ax.axvline(v_in - 0.5, color="black", lw=1.2)
    annotated_imshow(axes[-1], D,
                     f"D = folded unembedding  (V_out={d} × m={m})",
                     xlabel="neuron", ylabel="label")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def fig_tensor(logit_tensor, facts, path, d):
    """One V_in x V_in panel per label c; circles = stored facts with label c
    (X = model's argmax wrong on that fact)."""
    inputs, targets, pred = facts
    v_out = logit_tensor.shape[2]
    T = logit_tensor.cpu().numpy()
    vmax = np.abs(T).max()
    cols = min(v_out, 4)
    rows_n = (v_out + cols - 1) // cols
    fig, axes = plt.subplots(rows_n, cols,
                             figsize=(2.6 * cols, 2.7 * rows_n), squeeze=False)
    im = None
    for c in range(rows_n * cols):
        ax = axes[c // cols][c % cols]
        if c >= v_out:
            ax.axis("off")
            continue
        im = ax.imshow(T[:, :, c], cmap="RdBu", vmin=-vmax, vmax=vmax)
        v_in = T.shape[0]
        step = 1 if v_in <= 16 else 4
        ax.set_xticks(range(0, v_in, step))
        ax.set_yticks(range(0, v_in, step))
        marker_s = 42 if v_in <= 8 else (16 if v_in <= 16 else 6)
        marker_lw = 1.0 if v_in <= 16 else 0.5
        mask = (targets == c).cpu().numpy()
        pts = inputs[mask].cpu().numpy()
        ok = (pred[mask] == c).cpu().numpy()
        if pts[ok].size:
            ax.scatter(pts[ok][:, 1], pts[ok][:, 0], s=marker_s,
                       facecolors="none", edgecolors="black",
                       linewidths=marker_lw)
        if pts[~ok].size:
            ax.scatter(pts[~ok][:, 1], pts[~ok][:, 0], s=marker_s, marker="x",
                       color="black", linewidths=marker_lw + 0.2)
        ax.set_title(f"logits for label {c}", fontsize=8)
        ax.set_xlabel("token 2", fontsize=7)
        ax.set_ylabel("token 1", fontsize=7)
        ax.tick_params(labelsize=6, length=0)
    fig.suptitle("Model logit tensor  T[t1, t2, c]   "
                 "(○ stored fact, correct;  × stored fact, wrong)", y=1.0)
    fig.tight_layout()
    if im is not None:
        cbar = fig.colorbar(im, ax=[a for row in axes for a in row],
                            shrink=0.85, pad=0.02)
        cbar.set_label("logit", fontsize=8)
        cbar.ax.tick_params(labelsize=7)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def worked_example(w, d, t1, t2, y, logits):
    """Markdown table of the per-neuron computation for one fact."""
    v_in = 2 * d
    L, D = w["L"], w["down"]
    R = w.get("R", L)
    hL = L[:, t1] + L[:, v_in + t2]
    hR = R[:, t1] + R[:, v_in + t2]
    h = hL * hR
    order = torch.argsort(logits, descending=True)
    cstar = order[0].item() if order[0].item() != y else order[1].item()
    lines = [
        f"`t1={t1}, t2={t2}` → label **{y}**. "
        f"`Lx[n] = L[n,{t1}] + L[n,{v_in}+{t2}]`, same for `Rx`; "
        f"`h = Lx*Rx`; `logit[c] = Σ_n D[c,n]·h[n]`.",
        "",
        f"| n | Lx | Rx | h=Lx·Rx | D[y={y},n] | →logit[{y}] "
        f"| D[c*={cstar},n] | →logit[{cstar}] |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for n in range(h.shape[0]):
        lines.append(
            f"| {n} | {hL[n]:+.2f} | {hR[n]:+.2f} | {h[n]:+.2f} "
            f"| {D[y, n]:+.2f} | {D[y, n] * h[n]:+.2f} "
            f"| {D[cstar, n]:+.2f} | {D[cstar, n] * h[n]:+.2f} |")
    lines.append(
        f"| **Σ** | | | | | **{logits[y]:+.2f}** | | **{logits[cstar]:+.2f}** |")
    logit_str = ", ".join(
        (f"**{v:+.2f}**" if c == y else f"{v:+.2f}")
        for c, v in enumerate(logits.tolist()))
    lines.append("")
    lines.append(f"All logits: [{logit_str}] — margin "
                 f"{(logits[y] - logits[cstar]):+.2f} vs label {cstar}.")
    return "\n".join(lines)


def pick_tiers(margins, losses):
    """Example indices per tier, from near-perfect to fully unmemorized."""
    order = torch.argsort(margins, descending=True)
    pos = [i.item() for i in order if margins[i] > 0]
    neg = [i.item() for i in order if margins[i] <= 0]
    tiers = []
    if pos:
        tiers.append(("Near-perfect (largest margin, ~zero loss)", pos[:2]))
    if len(pos) > 4:
        mid = pos[len(pos) // 2: len(pos) // 2 + 2]
        tiers.append(("Comfortable (median margin)", mid))
    if len(pos) > 2:
        tiers.append(("At the margin (barely memorized)", pos[-2:]))
    if neg:
        tiers.append(("Just below the margin (barely NOT memorized)", neg[:2]))
    if len(neg) > 2:
        tiers.append(("Not memorized at all (worst margin)", neg[-2:]))
    return tiers


def report_for_d(d, cache, arch="bilinear", prefix="", label_mode="random"):
    m = width_for(arch, d, "param_matched")
    max_facts, _ = find_max_facts(cache, arch, d, m, 0.9, verbose=True,
                                  label_mode=label_mode)
    n = max(max_facts, 1)

    w, acc, (inputs, targets, x) = train_single(arch, d, m, n,
                                                label_mode=label_mode)
    v_in, v_out = 2 * d, d

    with torch.no_grad():
        L, D = w["L"], w["down"]
        R = w.get("R", L)
        # full logit tensor over every possible input pair
        hL = L[:, :v_in].unsqueeze(2) + L[:, v_in:].unsqueeze(1)  # (m,V,V)
        hR = R[:, :v_in].unsqueeze(2) + R[:, v_in:].unsqueeze(1)
        T = torch.einsum("cn,nab->abc", D, hL * hR)               # (V,V,V_out)
        fact_logits = T[inputs[:, 0], inputs[:, 1]]               # (n, V_out)
        pred = fact_logits.argmax(-1)
        n_correct = int((pred == targets).sum())
        if v_out > 1:
            top2 = fact_logits.topk(2, dim=-1)
            wrong_best = torch.where(top2.indices[:, 0] == targets,
                                     top2.values[:, 1], top2.values[:, 0])
            margins = fact_logits[torch.arange(n), targets] - wrong_best
        else:
            margins = torch.zeros(n)  # degenerate: no competing label
        losses = F.cross_entropy(fact_logits, targets, reduction="none")

    img_w = f"img/{prefix}d{d}_weights.png"
    img_t = f"img/{prefix}d{d}_tensor.png"
    fig_weights(w, d, os.path.join(OUT_DIR, img_w))
    fig_tensor(T, (inputs, targets, pred), os.path.join(OUT_DIR, img_t), d)

    if arch == "bilinear_sym":
        title = f"Symmetric bilinear model (R = L), d = {d}"
        arch_desc = ("`logits = D((Lx) ⊙ (Lx))` — shared input matrix, "
                     "x = [onehot(t1); onehot(t2)]. All hidden activations "
                     "are ≥ 0.")
    else:
        title = f"Bilinear model, d = {d}"
        arch_desc = "`logits = D((Lx) ⊙ (Rx))`, x = [onehot(t1); onehot(t2)]"
    md = [
        f"# {title}",
        "",
        f"| | |",
        f"|---|---|",
        f"| architecture | {arch_desc} |",
        f"| V_in / V_out / m | {v_in} / {v_out} / {m} |",
        f"| parameters | {param_count(arch, d, m)} |",
        f"| capacity (acc ≥ 0.9, any of 11 seeds) | **{max_facts}** of {4*d*d} possible facts |",
        f"| showcase model trained on | {n} facts (best seed of ≤11) |",
        f"| memorized (argmax correct) | **{n_correct} / {n}**  (acc {n_correct/n:.3f}) |",
        f"| margin stats | min {margins.min():+.2f}, median {margins.median():+.2f}, max {margins.max():+.2f} |",
        "",
    ]
    if v_out == 1:
        md += ["> **Degenerate case:** V_out = 1 means there is only one "
               "label, so every fact is trivially 'memorized' by argmax. "
               "Included for completeness.", ""]
    if label_mode == "sequential":
        per_label = max(1, (4 * d * d) // v_out)
        md += ["> **Sequential labels:** labels follow input enumeration "
               f"order — pair index `t1·{v_in} + t2`, first {per_label} "
               "pairs → label 0, next block → label 1, … Under this scaling "
               "that reduces to the rule `label = t1 // 2` (t2 is "
               "irrelevant), so this is a *learnable rule*, not random "
               "memorization — capacities are not comparable to the "
               "random-label reports.", ""]

    md += [
        "## Weights",
        "",
        "These three matrices are the *entire* model — the challenge "
        "architecture (post Fig. 4) folds the MLP input matrix into the "
        "embeddings and the output matrix into the unembedding. Because the "
        "input is a one-hot concat, **column t of L/R *is* token t's "
        "embedding vector**, mapping it straight to neuron pre-activations; "
        "**D is the unembedding** (neurons → label logits). The black "
        "divider separates the position-1 block (columns 0..V_in-1, read by "
        "token 1) from the position-2 block (read by token 2).",
        "",
        f"![weights]({img_w})",
        "",
        "## Third-order logit tensor",
        "",
        "The model's complete input→output function: `T[t1, t2, c]` for every "
        "possible input pair, one panel per label. Circles mark stored facts "
        "of that label (× = stored but predicted wrong). A perfect memorizer "
        "would have every circled cell be the winner across its panel-stack; "
        "everything un-circled is free to be arbitrary interference.",
        "",
        f"![tensor]({img_t})",
        "",
        "## Worked examples by success tier",
        "",
        "Facts sorted by margin = `logit[correct] − max wrong logit` "
        "(positive ⇒ memorized). `c*` is the strongest competing label.",
        "",
    ]
    tiers = pick_tiers(margins, losses) if v_out > 1 else []
    for tier_name, idxs in tiers:
        md.append(f"### {tier_name}")
        md.append("")
        for i in idxs:
            t1, t2 = int(inputs[i, 0]), int(inputs[i, 1])
            y = int(targets[i])
            md.append(f"**Fact {i}** — margin {margins[i]:+.2f}, "
                      f"CE loss {max(losses[i].item(), 0.0) + 0.0:.3f}")
            md.append("")
            md.append(worked_example(w, d, t1, t2, y, fact_logits[i]))
            md.append("")
    if not any("NOT" in t for t, _ in tiers):
        md.append("*(no unmemorized facts — this model is at 100% accuracy)*")
        md.append("")

    with open(os.path.join(OUT_DIR, f"{prefix}d{d}.md"), "w") as f:
        f.write("\n".join(md))
    torch.save({k: v.cpu() for k, v in w.items()},
               os.path.join(OUT_DIR, f"{prefix}d{d}_weights.pt"))
    return {"d": d, "m": m, "capacity": max_facts, "trained_n": n,
            "memorized": n_correct, "acc": round(n_correct / n, 4)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dvals", default="1,2,3,4,5,6,7,8")
    p.add_argument("--arch", default="bilinear",
                   choices=["bilinear", "bilinear_sym"])
    p.add_argument("--labels", default="random",
                   choices=["random", "sequential"])
    args = p.parse_args()
    setting = ("sym" if args.arch == "bilinear_sym" else "bilinear") \
        + "_" + args.labels
    global OUT_DIR, IMG_DIR
    OUT_DIR = os.path.join(BASE_DIR, setting)
    IMG_DIR = os.path.join(OUT_DIR, "img")
    os.makedirs(IMG_DIR, exist_ok=True)
    cache = GridCache()
    summary = []
    for d in [int(x) for x in args.dvals.split(",")]:
        print(f"===== d={d} =====", flush=True)
        summary.append(report_for_d(d, cache, args.arch, "", args.labels))
    name = ("Tiny symmetric-bilinear models (R = L)"
            if args.arch == "bilinear_sym" else "Tiny bilinear models")
    if args.labels == "sequential":
        name += ", sequential labels"
    else:
        name += ", random labels"
    lines = [f"# {name}", "",
             "| d | m | capacity (acc≥0.9) | of possible | showcase acc | file |",
             "|---|---|---|---|---|---|"]
    for s in sorted(summary, key=lambda s: s["d"]):
        lines.append(
            f"| {s['d']} | {s['m']} | {s['capacity']} | {4*s['d']**2} | "
            f"{s['acc']} | [d{s['d']}.md](d{s['d']}.md) |")
    with open(os.path.join(OUT_DIR, "README.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
