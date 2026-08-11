"""Twin animation: SGD re-tensioning vs certified insertion, same d=4 fact
set, all frames from the two ACTUAL trajectories (no synthetic data).

Layout per frame (2 columns = methods):
  row 1: L heatmap (shared color scale per method across time)
  row 2: margin ladder (symlog) with floor line and stored count
  row 3: tension web — facts on a circle, edges = margin-gradient coupling
         J between token-sharing facts (blue +, red −, alpha ~ |J|),
         node color = stored/not. SGD's web stiffens; certified's stays
         slack (tie-degenerate gradients).

Clocks differ by nature: SGD frames are labeled by epoch, certified frames
by accepted-move index; frames are paired by progress fraction.

Output: results/twin_retension_d4.gif (+ phone-size .gif)
"""

import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from capacity import generate_facts, DEVICE
from h12b_repair import h9b_solve
from maxmargin_cert import norm_margins
import insert_v2 as iv

torch.set_num_threads(4)
d = 4
v_in, v_out, n = 2 * d, d, 4 * d * d
inputs, targets = generate_facts(n, v_in, v_out)
inputs, targets = inputs.cpu(), targets.cpu()
X = torch.cat([F.one_hot(inputs[:, 0], v_in).float(),
               F.one_hot(inputs[:, 1], v_in).float()], dim=-1).double()

# ---- trajectory 1: SGD (snapshots by epoch)
sched = sorted(set(list(range(0, 10)) + [15, 25, 40, 65, 110, 200, 360,
                                         700, 1400, 2800, 5600, 11000,
                                         22000, 42000, 59999]))
g = torch.Generator().manual_seed(0)
L = ((torch.rand(d, 4 * d, generator=g) * 2 - 1) / math.sqrt(4 * d)
     ).to(DEVICE).requires_grad_(True)
x, tg = X.float().to(DEVICE), targets.to(DEVICE)
Dm = -torch.eye(v_out, device=DEVICE)
opt = torch.optim.SGD([L], lr=0.5, momentum=0.9)
sgd_traj = []
for ep in range(60000):
    if ep in sched:
        sgd_traj.append((f"epoch {ep}", L.detach().double().cpu().clone()))
    opt.zero_grad(set_to_none=True)
    logits = ((x @ L.T) ** 2) @ Dm.T
    loss = F.cross_entropy(logits, tg)
    loss.backward()
    opt.step()
sgd_traj.append(("epoch 59999", L.detach().double().cpu().clone()))

# ---- trajectory 2: certified insertion (snapshots by accepted move)
_, L0 = h9b_solve(X, targets, v_out, rounds=40)
trace = [(-1, L0.clone())]
acc_c, Lc, _ = iv.insert_v2(L0, X, targets, inputs, v_in, sweeps=6,
                            verbose=False, trace=trace)
cert_traj = [("spectral init" if i < 0 else f"move {k}: +fact {i}", Ls)
             for k, (i, Ls) in enumerate(trace)]

# pair frames by progress fraction
NF = 30
def pick(traj, f):
    return traj[min(len(traj) - 1, int(round(f * (len(traj) - 1))))]

overlap = ((inputs[:, 0].unsqueeze(1) == inputs[:, 0].unsqueeze(0))
           | (inputs[:, 1].unsqueeze(1) == inputs[:, 1].unsqueeze(0)))

def margin_grad(Ls, i):
    xx = X[i]
    y = int(targets[i])
    pre = Ls @ xx
    h = pre ** 2
    hh = h.clone()
    hh[y] = float("inf")
    cstar = int(hh.argmin())
    gm = torch.zeros_like(Ls)
    t1 = int(inputs[i, 0])
    t2 = int(inputs[i, 1])
    for (row, sgn) in ((y, -1.0), (cstar, +1.0)):
        gm[row, t1] += sgn * 2 * pre[row]
        gm[row, v_in + t2] += sgn * 2 * pre[row]
    return gm

theta = np.linspace(0, 2 * np.pi, n, endpoint=False)
px, py = np.cos(theta), np.sin(theta)
vmax = {0: max(float(t[1].abs().max()) for t in sgd_traj),
        1: max(float(t[1].abs().max()) for t in cert_traj)}
# normalize web opacity by the final SGD solution's coupling scale
_Lf = sgd_traj[-1][1]
_G = torch.stack([margin_grad(_Lf, i).flatten() for i in range(n)])
_J = ((_G / _G.norm(dim=1, keepdim=True).clamp_min(1e-12)) @ _G.T) \
    * overlap.double()
_J.fill_diagonal_(0)
JMAX = max(float(_J.abs().max()), 1e-9)
frames = []
for fi in range(NF):
    f = fi / (NF - 1)
    fig, axes = plt.subplots(3, 2, figsize=(10, 10.5), dpi=90,
                             gridspec_kw={"height_ratios": [1, 0.9, 1.4]})
    for col, (label, traj) in enumerate((("SGD (re-tensioning)", sgd_traj),
                                         ("certified insertion", cert_traj))):
        name, Ls = pick(traj, f)
        mt = norm_margins(Ls, X, targets)
        stored = mt > 0
        ax = axes[0][col]
        ax.imshow(Ls.numpy(), cmap="RdBu", vmin=-vmax[col], vmax=vmax[col])
        ax.axvline(v_in - 0.5, color="k", lw=1)
        ax.set_xticks([]), ax.set_yticks([])
        ax.set_title(f"{label}\nL — {name}", fontsize=10)
        ax = axes[1][col]
        cols = ["#2a78d6" if s else "#e34948" for s in stored]
        ax.scatter(range(n), mt.numpy(), s=12, c=cols)
        ax.axhline(0, color="k", lw=0.8)
        floor = float(mt.min())
        if floor > 0:
            ax.axhline(floor, color="#1baf7a", lw=1.2, ls="--")
        ax.set_yscale("symlog", linthresh=1e-6)
        ax.set_ylim(-1e-1, 1e-1)
        ax.set_title(f"margins — stored {int(stored.sum())}/{n}", fontsize=9)
        ax.tick_params(labelsize=6)
        # tension web
        G = torch.stack([margin_grad(Ls, i).flatten() for i in range(n)])
        norms = G.norm(dim=1).clamp_min(1e-12)
        J = ((G / norms.unsqueeze(1)) @ G.T) * overlap.double()
        J.fill_diagonal_(0)
        ax = axes[2][col]
        if JMAX is None:
            JMAX = max(float(J.abs().max()), 1e-9)
        for i in range(n):
            for j in range(i + 1, n):
                v = float(J[i, j] + J[j, i]) / 2
                a = min(1.0, abs(v) / (JMAX * 0.35))
                if a < 0.06:
                    continue
                ax.plot([px[i], px[j]], [py[i], py[j]],
                        color=("#2a78d6" if v > 0 else "#e34948"),
                        alpha=a * 0.7, lw=0.7)
        ax.scatter(px, py, s=20,
                   c=["#1baf7a" if s else "#e34948" for s in stored],
                   zorder=3)
        tension = float(J.abs().sum()) / 2
        ax.set_title(f"tension web  Σ|J| = {tension:.0f}", fontsize=9)
        ax.set_xlim(-1.15, 1.15), ax.set_ylim(-1.15, 1.15)
        ax.axis("off")
    fig.suptitle("Same 64 facts, two real trajectories (d=4)", y=0.995,
                 fontsize=11)
    fig.tight_layout()
    fig.canvas.draw()
    frames.append(Image.fromarray(
        np.asarray(fig.canvas.buffer_rgba())[:, :, :3]))
    plt.close(fig)

dur = [350] * len(frames)
dur[-1] = 3000
frames[0].save("results/twin_retension_d4.gif", save_all=True,
               append_images=frames[1:], duration=dur, loop=0)
small = [f.resize((f.width // 2, f.height // 2), Image.LANCZOS)
         for f in frames]
small[0].save("results/twin_retension_d4_phone.gif", save_all=True,
              append_images=small[1:], duration=dur, loop=0)
print("wrote results/twin_retension_d4.gif and _phone.gif")
