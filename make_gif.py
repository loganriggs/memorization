"""Animate an ACTUAL SGD training run (plain GD+momentum, the clean
Frobenius-geometry optimizer) from init to converged max-margin solution.

All frames are rendered from true snapshots of the training trajectory —
nothing idealized or synthesized.

Panels per frame:
  A. L heatmap (annotated), position divider marked
  B. per-fact normalized margins (symlog), floor line, stored count,
     newly-stored facts labeled "+N" (lost facts "-N")
  C. per-neuron signed pre-activation surfaces a_c[t1]+b_c[t2] with own
     facts circled (the cancellation zero-band forming)

Output: results/training_retension_d4.gif
"""

import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from capacity import generate_facts, DEVICE
from maxmargin_cert import norm_margins
import insert_v2 as iv

torch.set_num_threads(4)
d, EPOCHS = 4, 60000
v_in, v_out, n = 2 * d, d, 4 * d * d
inputs, targets = generate_facts(n, v_in, v_out)
inputs, targets = inputs.cpu(), targets.cpu()
X = torch.cat([F.one_hot(inputs[:, 0], v_in).float(),
               F.one_hot(inputs[:, 1], v_in).float()], dim=-1).double()

# frame schedule: dense early, sparse late (log-ish)
sched = sorted(set(
    list(range(0, 10)) + [12, 15, 20, 25, 30, 40, 50, 65, 85, 110, 150,
                          200, 270, 360, 500, 700, 1000, 1400, 2000, 2800,
                          4000, 5600, 8000, 11000, 16000, 22000, 30000,
                          42000, 59999]))

g = torch.Generator().manual_seed(0)
L = ((torch.rand(d, 4 * d, generator=g) * 2 - 1) / math.sqrt(4 * d)
     ).to(DEVICE).requires_grad_(True)
x, tg = X.float().to(DEVICE), targets.to(DEVICE)
D = -torch.eye(v_out, device=DEVICE)
opt = torch.optim.SGD([L], lr=0.5, momentum=0.9)
snaps = {}
for ep in range(EPOCHS):
    if ep in sched:
        snaps[ep] = L.detach().double().cpu().clone()
    opt.zero_grad(set_to_none=True)
    logits = ((x @ L.T) ** 2) @ D.T
    loss = F.cross_entropy(logits, tg)
    loss.backward()
    opt.step()
snaps[EPOCHS - 1] = L.detach().double().cpu().clone()
print(f"training done; {len(snaps)} snapshots")

frames = []
prev_stored = None
vmaxL = max(float(s.abs().max()) for s in snaps.values())
for ep in sorted(snaps):
    Ls = snaps[ep]
    mt = norm_margins(Ls, X, targets)
    stored = mt > 0
    acc = float(stored.float().mean())
    floor = float(mt.min())
    new = [] if prev_stored is None else \
        [int(i) for i in torch.where(stored & ~prev_stored)[0]]
    lost = [] if prev_stored is None else \
        [int(i) for i in torch.where(~stored & prev_stored)[0]]
    prev_stored = stored.clone()

    fig = plt.figure(figsize=(12, 6.4), dpi=100)
    gs = fig.add_gridspec(2, 5, height_ratios=[1.15, 1])
    # A: L heatmap
    axL = fig.add_subplot(gs[0, :3])
    axL.imshow(Ls.numpy(), cmap="RdBu", vmin=-vmaxL, vmax=vmaxL)
    for i in range(d):
        for j in range(4 * d):
            axL.text(j, i, f"{Ls[i, j]:.1f}", ha="center", va="center",
                     fontsize=5.5,
                     color="white" if abs(Ls[i, j]) > 0.6 * vmaxL else "black")
    axL.axvline(v_in - 0.5, color="k", lw=1.2)
    axL.set_yticks(range(d))
    axL.set_xticks(range(0, 4 * d, 4))
    axL.tick_params(labelsize=6, length=0)
    axL.set_title(f"L  (epoch {ep})", fontsize=10, loc="left")
    # B: margin ladder
    axM = fig.add_subplot(gs[0, 3:])
    cols = ["#2a78d6" if s else "#e34948" for s in stored]
    axM.scatter(range(n), mt.numpy(), s=14, c=cols)
    axM.axhline(0, color="k", lw=0.8)
    if floor > 0:
        axM.axhline(floor, color="#1baf7a", lw=1.2, ls="--")
        axM.text(n * 0.99, floor, " floor", color="#1baf7a", fontsize=8,
                 va="bottom", ha="right")
    axM.set_yscale("symlog", linthresh=1e-4)
    axM.set_ylim(-1e-1, 1e-1)
    axM.set_title(f"normalized margins — stored {int(stored.sum())}/{n} "
                  f"(acc {acc:.2f})", fontsize=10, loc="left")
    axM.set_xlabel("fact index", fontsize=8)
    for k, i in enumerate(new[:5]):
        axM.annotate(f"+{i}", (i, float(mt[i])), fontsize=7,
                     color="#008300", xytext=(0, 8),
                     textcoords="offset points", ha="center")
    if len(new) > 5:
        axM.text(0.02, 0.02, f"+{len(new)} newly stored",
                 transform=axM.transAxes, fontsize=8, color="#008300")
    for i in lost[:3]:
        axM.annotate(f"−{i}", (i, float(mt[i])), fontsize=7,
                     color="#e34948", xytext=(0, -12),
                     textcoords="offset points", ha="center")
    # C: cancellation surfaces
    vmaxP = 0
    Ps = []
    for c in range(d):
        a, b = Ls[c, :v_in], Ls[c, v_in:]
        P = a.unsqueeze(1) + b.unsqueeze(0)
        Ps.append(P)
        vmaxP = max(vmaxP, float(P.abs().max()))
    for c in range(d):
        ax = fig.add_subplot(gs[1, c])
        ax.imshow(Ps[c].numpy(), cmap="RdBu", vmin=-vmaxP, vmax=vmaxP)
        own = inputs[targets == c]
        ax.scatter(own[:, 1], own[:, 0], s=26, facecolors="none",
                   edgecolors="black", linewidths=0.9)
        ax.set_xticks([]), ax.set_yticks([])
        ax.set_title(f"neuron {c}: a[t1]+b[t2]", fontsize=8)
    axtxt = fig.add_subplot(gs[1, 4])
    axtxt.axis("off")
    axtxt.text(0.05, 0.7, "○ = label-c facts\n(want: on the white\n"
               "zero-band = silenced)\n\nblue dots right panel:\nstored facts;"
               "\nred: not yet", fontsize=8, va="top")
    fig.suptitle("Actual SGD trajectory, d=4 (all data from the real run)",
                 y=0.99, fontsize=11)
    fig.tight_layout()
    fig.canvas.draw()
    frames.append(Image.fromarray(
        np.asarray(fig.canvas.buffer_rgba())[:, :, :3]))
    plt.close(fig)

durations = [400 if i < 10 else 180 for i in range(len(frames))]
durations[-1] = 2500
frames[0].save("results/training_retension_d4.gif", save_all=True,
               append_images=frames[1:], duration=durations, loop=0)
print(f"wrote results/training_retension_d4.gif ({len(frames)} frames)")
EOF_MARKER_NOT_USED = None
