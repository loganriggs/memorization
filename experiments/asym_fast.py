"""Asymmetric silence construction (diagnostic, m=d => 1.8x params).

h_n = (L_n.x)(R_n.x), D = -I. Own-fact silence needs only ONE factor to
cancel (OR of linear constraints). Build: start from the symmetric H9b
directions (L=R=v_c), then alternate weighted generalized eigensolves
(given R, minimizing own-response over L is quadratic with per-fact weights
(R.x)^2, and vice versa). Then greedy hinge repair over both matrices with
incremental evaluation. No gradient descent.
"""

import torch
import torch.nn.functional as F

from capacity import generate_facts
from h12b_repair import h9b_solve

EPS = 1e-3
TAU = 0.3


def weighted_eig_min(X, targets, c, wts):
    dim = X.shape[1]
    mp, mn = targets == c, targets != c
    P, N = X[mp], X[mn]
    wp, wn = wts[mp].unsqueeze(1), wts[mn].unsqueeze(1)
    A = (P * wp).T @ P / wp.sum().clamp_min(1e-9)
    B = (N * wn).T @ N / wn.sum().clamp_min(1e-9) \
        + EPS * torch.eye(dim, dtype=torch.float64)
    R_ = torch.linalg.cholesky(B, upper=True)
    Ri = torch.linalg.inv(R_)
    _, V = torch.linalg.eigh(Ri.T @ A @ Ri)
    v = Ri @ V[:, 0]
    return v / (v @ B @ v).sqrt()


def build_asym(X, targets, v_out, alternations=6):
    _, L = h9b_solve(X, targets, v_out, rounds=20)
    R = L.clone()
    for _ in range(alternations):
        for c in range(v_out):
            wts = (X @ R[c]) ** 2 + 1e-6
            L[c] = weighted_eig_min(X, targets, c, wts)
            wts = (X @ L[c]) ** 2 + 1e-6
            R[c] = weighted_eig_min(X, targets, c, wts)
    return L, R


def _score(logits, targets):
    own = logits.gather(-1, targets.view(-1, 1, 1).expand(
        logits.shape[0], logits.shape[1], 1)).squeeze(2)
    other = logits.scatter(
        2, targets.view(-1, 1, 1).expand(logits.shape[0], logits.shape[1], 1),
        float("-inf"))
    margin = own - other.max(dim=2).values
    return margin.clamp(max=TAU).sum(0), (margin > 0).sum(0)


def fast_repair_asym(L, R, X, targets, inputs, v_in, passes=10, n_cand=21):
    L, R = L.clone(), R.clone()
    n, dim = X.shape
    m = L.shape[0]
    D = -torch.eye(m, dtype=torch.float64)
    preL, preR = X @ L.T, X @ R.T
    logits = (preL * preR) @ D.T
    score, ncorr = _score(logits.unsqueeze(1), targets)
    score, ncorr = float(score), int(ncorr)
    edges = {r: [(int(t[0]), int(t[1]))
                 for t, y in zip(inputs, targets) if int(y) == r]
             for r in range(m)}

    def try_move(mat, pre_this, pre_other, r, delta_cols):
        nonlocal score, ncorr, logits
        width = max(float(mat[r].abs().max()), 0.3)
        dvs = torch.linspace(-width, width, n_cand, dtype=torch.float64)
        pre_new = pre_this[:, r].unsqueeze(1) + delta_cols.unsqueeze(1) * dvs
        h_new = pre_new * pre_other[:, r].unsqueeze(1)          # (n, K)
        h_old = (pre_this[:, r] * pre_other[:, r]).unsqueeze(1)
        cand = logits.unsqueeze(1) + (h_new - h_old).unsqueeze(2) \
            * D[:, r].view(1, 1, -1)
        s, nc = _score(cand, targets)
        k = torch.argmax(nc * 1e6 + s)
        if (int(nc[k]), float(s[k])) > (ncorr, score):
            dv = float(dvs[k])
            pre_this[:, r] += delta_cols * dv
            logits = (preL * preR) @ D.T
            score, ncorr = float(s[k]), int(nc[k])
            return dv
        return None

    for p in range(passes):
        improved = False
        for r in range(m):
            for (mat, pt, po) in ((L, preL, preR), (R, preR, preL)):
                for j in range(dim):
                    dv = try_move(mat, pt, po, r, X[:, j])
                    if dv is not None:
                        mat[r, j] += dv
                        improved = True
                for (t1, t2) in edges.get(r, []):
                    j, k = t1, v_in + t2
                    dv = try_move(mat, pt, po, r, X[:, j] - X[:, k])
                    if dv is not None:
                        mat[r, j] += dv
                        mat[r, k] -= dv
                        improved = True
        if not improved or ncorr == n:
            break
    return ncorr / n, L, R


if __name__ == "__main__":
    import time
    torch.set_num_threads(4)
    gen = torch.Generator().manual_seed(0)
    print(f"{'d':>3} {'sym best':>8} {'asym best':>9} {'time':>6}")
    SYM = {6: .889, 8: .844, 12: .703, 16: .628}
    for d in (6, 8, 12, 16):
        v_in, v_out, n = 2 * d, d, 4 * d * d
        inputs, targets = generate_facts(n, v_in, v_out)
        inputs, targets = inputs.cpu(), targets.cpu()
        X = torch.cat([F.one_hot(inputs[:, 0], v_in).float(),
                       F.one_hot(inputs[:, 1], v_in).float()],
                      dim=-1).double()
        t0 = time.time()
        L0, R0 = build_asym(X, targets, v_out)
        best = 0.0
        for r in range(6):
            Lr = L0 if r == 0 else L0 + 0.15 * torch.randn(
                L0.shape, generator=gen, dtype=torch.float64)
            Rr = R0 if r == 0 else R0 + 0.15 * torch.randn(
                R0.shape, generator=gen, dtype=torch.float64)
            acc, _, _ = fast_repair_asym(Lr, Rr, X, targets, inputs, v_in)
            best = max(best, acc)
            if best == 1.0:
                break
        print(f"{d:>3} {SYM[d]:>8.3f} {best:>9.3f} {time.time()-t0:>5.0f}s",
              flush=True)
