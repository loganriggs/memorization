"""Render the construction pipeline stage-by-stage for small d.

For each d: Stage 1 = plain anti-Rayleigh (one unweighted eigensolve per
label), Stage 2 = + iterative fact reweighting (best of 40 rounds),
Stage 3 = + hinge-margin greedy repair (best of 8 restarts). After each
stage, render L (shared R = L), D (fixed -I), and the full logit tensor
with facts marked. Writes tiny_models/construction/d{d}.md.

Usage: python construction_stages.py [--dvals 2,3,4,6,8]
"""

import argparse
import os

import torch
import torch.nn.functional as F

from capacity import generate_facts
from h12b_repair import h9b_solve
from h12c_fast import fast_repair
import tiny_report

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(HERE, "tiny_models", "construction")
IMG = os.path.join(OUT, "img")


def tensor_and_pred(L, D, inputs, targets, v_in):
    hL = L[:, :v_in].unsqueeze(2) + L[:, v_in:].unsqueeze(1)   # (m, V, V)
    T = torch.einsum("cn,nab->abc", D, hL * hL)
    pred = T[inputs[:, 0], inputs[:, 1]].argmax(-1)
    acc = (pred == targets).float().mean().item()
    return T, pred, acc


def run_d(d, gen):
    v_in, v_out, n = 2 * d, d, 4 * d * d
    inputs, targets = generate_facts(n, v_in, v_out)
    inputs, targets = inputs.cpu(), targets.cpu()
    X = torch.cat([F.one_hot(inputs[:, 0], v_in).float(),
                   F.one_hot(inputs[:, 1], v_in).float()], dim=-1).double()
    D = -torch.eye(v_out, dtype=torch.float64)

    # Stage 1: single unweighted anti-Rayleigh solve
    _, L1 = h9b_solve(X, targets, v_out, rounds=1)
    # Stage 2: full reweighting, keep best round
    _, L2 = h9b_solve(X, targets, v_out, rounds=40)
    # Stage 3: hinge repair, best of 8 restarts
    best_acc, L3 = -1.0, None
    for r in range(8):
        Lr = L2 if r == 0 else L2 + 0.15 * torch.randn(
            L2.shape, generator=gen, dtype=torch.float64)
        acc, Lrep = fast_repair(Lr, D, X, targets, inputs, v_in,
                                passes=12, n_cand=25)
        if acc > best_acc:
            best_acc, L3 = acc, Lrep
        if best_acc == 1.0:
            break

    stages = [("Stage 1 — anti-Rayleigh spectral init "
               "(one eigensolve per label)", L1),
              ("Stage 2 — + iterative fact reweighting "
               "(best of 40 rounds)", L2),
              ("Stage 3 — + hinge-margin greedy repair "
               "(best of 8 restarts)", L3)]

    md = [f"# Construction stages, symmetric bilinear d = {d}",
          "",
          f"n = {n} facts (full ceiling), m = {d} neurons, D fixed at −I "
          "(silence code: label logit = −(own neuron's squared response); "
          "a label wins by being *quietest*, so its facts sit at the "
          "whitest cells of its own mostly-red tensor slice).",
          ""]
    for si, (name, L) in enumerate(stages, 1):
        T, pred, acc = tensor_and_pred(L, D, inputs, targets, v_in)
        Lf = L.float()
        w = {"L": Lf, "down": D.float()}
        img_w = f"img/d{d}_stage{si}_weights.png"
        img_t = f"img/d{d}_stage{si}_tensor.png"
        tiny_report.fig_weights(w, d, os.path.join(OUT, img_w))
        tiny_report.fig_tensor(T.float(), (inputs, targets, pred),
                               os.path.join(OUT, img_t), d)
        md += [f"## {name}",
               "",
               f"Accuracy: **{acc:.3f}**  ({int((pred == targets).sum())}"
               f"/{n} facts)",
               "",
               f"![weights]({img_w})",
               "",
               f"![tensor]({img_t})",
               ""]
        print(f"  d={d} stage {si}: acc {acc:.3f}", flush=True)
    with open(os.path.join(OUT, f"d{d}.md"), "w") as f:
        f.write("\n".join(md))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dvals", default="2,3,4,6,8")
    args = p.parse_args()
    os.makedirs(IMG, exist_ok=True)
    torch.set_num_threads(4)
    gen = torch.Generator().manual_seed(0)
    dvals = [int(x) for x in args.dvals.split(",")]
    for d in dvals:
        print(f"===== d={d} =====", flush=True)
        run_d(d, gen)
    lines = ["# Construction stage reports", ""]
    lines += [f"- [d{d}.md](d{d}.md)" for d in dvals]
    with open(os.path.join(OUT, "README.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("done")


if __name__ == "__main__":
    main()
