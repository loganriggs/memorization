"""Train 10 seeds per d (2..8): dense sym-bilinear at n=4d^2, then D-only L1
pruning to each seed's acc>=0.9 frontier. Saves canonical snapshots to
tiny_models/sym_random/multiseed/d{d}_seed{s}.pt

Canonicalization: fold per-D-column scale into L rows (D col max-|entry|=1),
then flip each L row's sign so its largest-|entry| is positive.
"""

import math
import os

import torch
import torch.nn.functional as F

from capacity import generate_facts
from sparsity_d8 import (accuracy, clone, LAMBDA, PRUNE_FRAC, L1_EPOCHS,
                         FT_EPOCHS, LR, DEVICE)
from sparsity_d8_donly import train_d_penalty

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "tiny_models", "sym_random", "multiseed")
DVALS = range(2, 9)
N_SEEDS = 10
DENSE_EPOCHS = 5000
PATIENCE = 100


def train_dense(d, seed):
    v_in, v_out, n = 2 * d, d, 4 * d * d
    in_dim = 2 * v_in
    gen = torch.Generator(device="cpu").manual_seed(1000 + seed)
    L = ((torch.rand(d, in_dim, generator=gen) * 2 - 1)
         / math.sqrt(in_dim)).to(DEVICE).requires_grad_(True)
    D = ((torch.rand(v_out, d, generator=gen) * 2 - 1)
         / math.sqrt(d)).to(DEVICE).requires_grad_(True)
    w = {"L": L, "down": D}
    inputs, targets = generate_facts(n, v_in, v_out)
    x = torch.cat([F.one_hot(inputs[:, 0], v_in).float(),
                   F.one_hot(inputs[:, 1], v_in).float()], dim=-1).to(DEVICE)
    targets = targets.to(DEVICE)
    opt = torch.optim.Adam([L, D], lr=LR)
    best, since = 0.0, 0
    for _ in range(DENSE_EPOCHS):
        opt.zero_grad(set_to_none=True)
        h = x @ L.T
        logits = (h * h) @ D.T
        loss = F.cross_entropy(logits, targets)
        loss.backward()
        opt.step()
        acc = (logits.argmax(-1) == targets).float().mean().item()
        if acc > best:
            best, since = acc, 0
        else:
            since += 1
        if best >= 1.0 or since >= PATIENCE:
            break
    return w, x, targets, best


def prune_to_frontier(w, x, targets):
    mask_d = torch.ones_like(w["down"])
    best_snap, best_nnz, best_acc = None, None, None
    while True:
        train_d_penalty(w, mask_d, x, targets, L1_EPOCHS, l1=LAMBDA)
        vals = (w["down"].detach().abs() + (1 - mask_d) * 1e9).flatten()
        k_drop = max(1, int(round(PRUNE_FRAC * int(mask_d.sum()))))
        thresh = vals.kthvalue(k_drop).values
        mask_d *= (w["down"].detach().abs() > thresh).float()
        with torch.no_grad():
            w["down"].mul_(mask_d)
        nnz = int(mask_d.sum())
        wf = clone(w)
        train_d_penalty(wf, mask_d, x, targets, FT_EPOCHS, l1=0.0)
        acc_ft = accuracy(wf, x, targets)
        if acc_ft >= 0.9:
            best_snap = {k: v.detach().cpu() for k, v in wf.items()}
            best_nnz, best_acc = nnz, acc_ft
        if acc_ft < 0.9 or nnz <= 1:
            break
    return best_snap, best_nnz, best_acc


def canonicalize(w):
    L, D = w["L"].clone(), w["down"].clone()
    s = D.abs().max(dim=0).values
    s = torch.where(s > 0, s, torch.ones_like(s))
    D = D / s
    L = L * s.sqrt().unsqueeze(1)
    idx = L.abs().argmax(dim=1)
    sign = torch.sign(L[torch.arange(L.shape[0]), idx])
    sign = torch.where(sign == 0, torch.ones_like(sign), sign)
    L = L * sign.unsqueeze(1)
    return {"L": L, "down": D}


def main():
    os.makedirs(OUT, exist_ok=True)
    for d in DVALS:
        for seed in range(N_SEEDS):
            path = os.path.join(OUT, f"d{d}_seed{seed}.pt")
            if os.path.exists(path):
                print(f"skip d={d} seed={seed}", flush=True)
                continue
            w, x, targets, dense_acc = train_dense(d, seed)
            snap, nnz, acc = prune_to_frontier(w, x, targets)
            if snap is None:
                print(f"d={d} seed={seed}: FRONTIER NOT FOUND "
                      f"(dense acc {dense_acc:.3f})", flush=True)
                continue
            canon = canonicalize(snap)
            torch.save({"raw": snap, "canonical": canon, "nnz_D": nnz,
                        "acc": acc, "dense_acc": dense_acc}, path)
            per_label = (canon["down"] != 0).sum(dim=1).int().tolist()
            print(f"d={d} seed={seed}: dense {dense_acc:.3f} -> frontier "
                  f"{nnz} taps {per_label}, acc {acc:.3f}", flush=True)
    print("fleet done")


if __name__ == "__main__":
    main()
