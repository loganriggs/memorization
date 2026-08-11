"""P2-1: certified-forget editability sweep — success/cost/runtime across
facts, edit-space widths (2-param own row, 2-param rival row, 4-param
joint), and scales (d=4, d=8), on SGD-trained models.

Appends results to results/p2_edit_scale.jsonl.
"""

import json
import math
import time

import numpy as np
import torch
import torch.nn.functional as F

from capacity import generate_facts, DEVICE
import insert_v2 as iv

torch.set_num_threads(6)
OUT = "results/p2_edit_scale.jsonl"


def train_sgd(d, n, epochs=60000):
    v_in, v_out = 2 * d, d
    inputs, targets = generate_facts(n, v_in, v_out)
    x = torch.cat([F.one_hot(inputs[:, 0].cpu(), v_in).float(),
                   F.one_hot(inputs[:, 1].cpu(), v_in).float()],
                  dim=-1).to(DEVICE)
    tg = targets.to(DEVICE)
    g = torch.Generator().manual_seed(0)
    L = ((torch.rand(d, 4 * d, generator=g) * 2 - 1) / math.sqrt(4 * d)
         ).to(DEVICE).requires_grad_(True)
    Dm = -torch.eye(v_out, device=DEVICE)
    opt = torch.optim.SGD([L], lr=0.5, momentum=0.9)
    for ep in range(epochs):
        opt.zero_grad(set_to_none=True)
        logits = ((x @ L.T) ** 2) @ Dm.T
        loss = F.cross_entropy(logits, tg)
        loss.backward()
        opt.step()
    return L.detach().double().cpu(), inputs.cpu(), targets.cpu()


def try_forget(L, X, inputs, targets, k, rows, mt0, n_iter=4000, seed=0):
    """Random-direction annealed search in the len(rows)*2-param space
    (rows x fact-k columns); exact eval; min-norm zero-collateral."""
    v_in = X.shape[1] // 2
    n = X.shape[0]
    t1, t2 = int(inputs[k, 0]), int(inputs[k, 1])
    cols = [t1, v_in + t2]
    stored0 = mt0 > 0
    others = torch.ones(n, dtype=torch.bool)
    others[k] = False
    scale = float(L.abs().max())
    g = torch.Generator().manual_seed(seed)
    best = None
    dims = len(rows) * 2
    for it in range(n_iter):
        R = float(np.geomspace(0.02 * scale, 5 * scale, 40)[
            min(39, it * 40 // n_iter)])
        v = torch.randn(dims, generator=g, dtype=torch.float64)
        v = v / v.norm() * R
        L2 = L.clone()
        idx = 0
        for r in rows:
            for c in cols:
                L2[r, c] += float(v[idx])
                idx += 1
        m2 = iv.margins_of(L2, X, targets)
        if m2[k] <= 0 and int((stored0[others] & (m2[others] <= 0)).sum()) == 0:
            if best is None or R < best:
                best = R
    return best


def main():
    for d, n in ((4, 64), (8, 200)):
        v_in, v_out = 2 * d, d
        L, inputs, targets = train_sgd(d, n)
        X = torch.cat([F.one_hot(inputs[:, 0], v_in).float(),
                       F.one_hot(inputs[:, 1], v_in).float()],
                      dim=-1).double()
        mt0 = iv.margins_of(L, X, targets)
        acc = float((mt0 > 0).float().mean())
        stored = torch.where(mt0 > 0)[0]
        qs = torch.quantile(mt0[stored].float(),
                            torch.tensor([0.05, 0.25, 0.5, 0.75, 0.95]))
        picks = [int(stored[torch.argmin((mt0[stored] - q).abs())])
                 for q in qs]
        print(f"d={d} n={n}: acc {acc:.3f}; testing facts {picks}",
              flush=True)
        for k in picks:
            y = int(targets[k])
            h = (L @ X[k]) ** 2
            hh = h.clone()
            hh[y] = float("inf")
            cstar = int(hh.argmin())
            rec = {"d": d, "n": n, "fact": k,
                   "margin_quantile": float(mt0[k])}
            for name, rows in (("own2", [y]), ("rival2", [cstar]),
                               ("joint4", [y, cstar])):
                t0 = time.time()
                r = try_forget(L, X, inputs, targets, k, rows, mt0)
                rec[name] = {"success": r is not None,
                             "delta_frac": (r / float(L.norm())
                                            if r else None),
                             "secs": round(time.time() - t0, 1)}
            with open(OUT, "a") as f:
                f.write(json.dumps(rec) + "\n")
            s = {nm: ("%.1f%%" % (100 * rec[nm]["delta_frac"])
                      if rec[nm]["success"] else "FAIL")
                 for nm in ("own2", "rival2", "joint4")}
            print(f"  fact {k} (m={float(mt0[k]):.2f}): {s}", flush=True)


if __name__ == "__main__":
    main()
