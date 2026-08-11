"""P3 part 2: (A) correlation sweep — does input correlation (not density)
kill zero-collateral surgery? (B) functional-distance editing — does
||Delta sym(M)|| predict collateral better than ||Delta L||, and does
selecting edits by functional proximity recover low-collateral edits?

Dense toy: dim=64, C=10, m=10 (negI, symmetric) + one asymmetric run.
Inputs with tunable correlation: x = normalize(z + c * B s), B: dim x 8
shared basis; c sweeps mean pairwise |cos|.
Appends to results/p3_corr_funcdist.jsonl.
"""

import json
import math

import numpy as np
import torch
import torch.nn.functional as F

from capacity import DEVICE

torch.set_num_threads(6)
OUT = "results/p3_corr_funcdist.jsonl"
DIM, C, M, N = 64, 10, 10, 256


def data(c_corr, seed=42):
    g = torch.Generator().manual_seed(seed)
    B = torch.randn(DIM, 8, generator=g, dtype=torch.float64)
    z = torch.randn(N, DIM, generator=g, dtype=torch.float64)
    s = torch.randn(N, 8, generator=g, dtype=torch.float64)
    x = z + c_corr * (s @ B.T)
    x = x / x.norm(dim=1, keepdim=True)
    y = torch.randint(0, C, (N,), generator=g)
    ov = (x @ x.T).abs()
    ov.fill_diagonal_(0)
    return x, y, float(ov.mean())


def train(x, y, sym=True, epochs=60000, seed=0):
    gg = torch.Generator().manual_seed(seed)
    L = ((torch.rand(M, DIM, generator=gg) * 2 - 1) / math.sqrt(DIM)
         ).to(DEVICE).float().requires_grad_(True)
    params = [L]
    if not sym:
        R = ((torch.rand(M, DIM, generator=gg) * 2 - 1) / math.sqrt(DIM)
             ).to(DEVICE).float().requires_grad_(True)
        params.append(R)
    xg, tg = x.float().to(DEVICE), y.to(DEVICE)
    Dm = -torch.eye(C, device=DEVICE)
    opt = torch.optim.SGD(params, lr=1.0, momentum=0.9)
    for ep in range(epochs):
        opt.zero_grad(set_to_none=True)
        h = (xg @ L.T) * (xg @ (L if sym else R).T)
        loss = F.cross_entropy(h @ Dm.T, tg)
        loss.backward()
        opt.step()
    out = [L.detach().double().cpu()]
    if not sym:
        out.append(R.detach().double().cpu())
    return out


def margins(ws, x, y):
    L = ws[0]
    Rm = ws[1] if len(ws) > 1 else ws[0]
    h = (x @ L.T) * (x @ Rm.T)
    logits = h @ (-torch.eye(C).double())
    own = logits.gather(1, y.unsqueeze(1)).squeeze(1)
    oth = logits.scatter(1, y.unsqueeze(1), float("-inf")).max(dim=1).values
    return own - oth


def sym_forms(ws):
    L = ws[0]
    Rm = ws[1] if len(ws) > 1 else ws[0]
    # negI: M_c = -sym(l_c r_c^T)
    out = []
    for cc in range(C):
        Mm = -torch.outer(L[cc], Rm[cc])
        out.append((Mm + Mm.T) / 2)
    return torch.stack(out)


def edit_experiment(ws, x, y, k, n_cand=4000, seed=0):
    """Sample many forget-achieving candidate edits (rank-1 on L rows);
    record (collateral, wd, fd) per candidate."""
    L = ws[0]
    m0 = margins(ws, x, y)
    stored0 = m0 > 0
    others = torch.ones(N, dtype=torch.bool)
    others[k] = False
    xk = x[k]
    F0 = sym_forms(ws)
    g = torch.Generator().manual_seed(seed)
    cands = []
    scale = float(L.norm())
    for it in range(n_cand):
        R_ = float(np.geomspace(1e-3, 2.0, 30)[min(29, it * 30 // n_cand)]
                   ) * scale
        delta = torch.randn(C, generator=g, dtype=torch.float64)
        delta = delta / delta.norm() * R_
        L2 = L + delta.unsqueeze(1) * xk.unsqueeze(0)
        ws2 = [L2] + list(ws[1:])
        m2 = margins(ws2, x, y)
        if m2[k] <= 0:
            coll = int((stored0[others] & (m2[others] <= 0)).sum())
            wd = float((L2 - L).norm())
            fd = float((sym_forms(ws2) - F0).norm())
            cands.append((coll, wd, fd))
    return cands, stored0


def main():
    from scipy.stats import spearmanr
    for c_corr, sym in ((0.0, True), (0.5, True), (1.0, True), (2.0, True),
                        (0.5, False)):
        x, y, mean_ov = data(c_corr)
        ws = train(x, y, sym=sym)
        mt = margins(ws, x, y)
        stored = mt > 0
        acc = float(stored.float().mean())
        idx = torch.where(stored)[0]
        picks = [int(idx[torch.argsort(mt[idx])[len(idx) // 2]]),
                 int(idx[torch.argsort(mt[idx])[len(idx) // 4]])]
        all_c = []
        zero_ok = 0
        sel = {"min_wd": [], "min_fd": [], "best": []}
        for k in picks:
            cands, _ = edit_experiment(ws, x, y, k)
            if not cands:
                continue
            all_c += cands
            colls = [c0 for c0, _, _ in cands]
            zero_ok += int(min(colls) == 0)
            sel["best"].append(min(colls))
            sel["min_wd"].append(min(cands, key=lambda t: t[1])[0])
            sel["min_fd"].append(min(cands, key=lambda t: t[2])[0])
        rec = {"c_corr": c_corr, "sym": sym, "mean_overlap": round(mean_ov, 3),
               "acc": round(acc, 3),
               "zero_collateral_ok": f"{zero_ok}/{len(picks)}"}
        if all_c:
            colls = np.array([a for a, _, _ in all_c], dtype=float)
            wds = np.array([b for _, b, _ in all_c])
            fds = np.array([c0 for _, _, c0 in all_c])
            rec["spearman_wd_coll"] = round(float(spearmanr(wds, colls)[0]), 3)
            rec["spearman_fd_coll"] = round(float(spearmanr(fds, colls)[0]), 3)
            rec["collateral_of_selection"] = {
                kk: (round(float(np.mean(v)), 1) if v else None)
                for kk, v in sel.items()}
        with open(OUT, "a") as f:
            f.write(json.dumps(rec) + "\n")
        print(json.dumps(rec), flush=True)


if __name__ == "__main__":
    main()
