"""H13a: shared-neuron circulant tap-graph construction.

D fixed: D[c, c] = -1 (neuron c silences its own label) and
D[c, (c+1) % d] = +beta (neuron c+1 excites label c). Each neuron n thus
serves two consumers: silence for label n, excitation for label n-1.

Neuron solve (two-sided Rayleigh): v_n = top generalized eigenvector of
(A_{n-1}, gamma*A_n + B_rest + eps*I) — loud on its excitation label,
quiet on its silence label, controlled elsewhere. Then hinge-margin greedy
repair with this fixed D (generalized from h12b). No gradient descent.

Usage: python h13_tapgraph.py [--dvals 4,6,8] [--betas 0.4,1.0]
       [--gammas 1,3,10]
"""

import argparse
import itertools

import torch
import torch.nn.functional as F

from capacity import generate_facts

EPS = 1e-3
TAU = 0.3


def build_D(d, beta):
    D = -torch.eye(d, dtype=torch.float64)
    for c in range(d):
        D[c, (c + 1) % d] = beta
    return D


def solve_neurons(X, targets, d, gamma):
    dim = X.shape[1]
    L = torch.zeros(d, dim, dtype=torch.float64)
    eye = EPS * torch.eye(dim, dtype=torch.float64)
    covs = []
    for c in range(d):
        P = X[targets == c]
        covs.append(P.T @ P / len(P))
    B_all = X.T @ X / len(X)
    for n in range(d):
        exc = covs[(n - 1) % d]              # label n-1: be loud
        sil = covs[n]                        # label n: be quiet
        B = gamma * sil + B_all + eye
        R = torch.linalg.cholesky(B, upper=True)
        Ri = torch.linalg.inv(R)
        _, V = torch.linalg.eigh(Ri.T @ exc @ Ri)
        v = Ri @ V[:, -1]
        L[n] = v / (v @ B @ v).sqrt()
    return L


def hinge_score_D(L, D, X, targets):
    h = (X @ L.T) ** 2
    logits = h @ D.T
    own = logits.gather(1, targets.unsqueeze(1)).squeeze(1)
    other = logits.clone()
    other.scatter_(1, targets.unsqueeze(1), float("-inf"))
    margin = own - other.max(dim=1).values
    return margin.clamp(max=TAU).sum().item(), int((margin > 0).sum())


def repair_D(L, D, X, targets, inputs, v_in, passes=6, n_cand=17):
    L = L.clone()
    n, dim = X.shape
    m = L.shape[0]
    score, ncorr = hinge_score_D(L, D, X, targets)
    sil_edges = {nn: [(int(t[0]), int(t[1]))
                      for t, y in zip(inputs, targets) if int(y) == nn]
                 for nn in range(m)}
    for _ in range(passes):
        improved = False
        for r in range(m):
            width = max(float(L[r].abs().max()), 0.3)
            for j in range(dim):
                base = L[r, j].item()
                best_v, best_s, best_n = base, score, ncorr
                for dv in torch.linspace(-width, width, n_cand).tolist():
                    L[r, j] = base + dv
                    s, nc = hinge_score_D(L, D, X, targets)
                    if (nc, s) > (best_n, best_s):
                        best_v, best_s, best_n = base + dv, s, nc
                L[r, j] = best_v
                if (best_n, best_s) > (ncorr, score):
                    score, ncorr = best_s, best_n
                    improved = True
            for (t1, t2) in sil_edges[r]:
                j, k = t1, v_in + t2
                bj, bk = L[r, j].item(), L[r, k].item()
                width = max(float(L[r].abs().max()), 0.3)
                best = (bj, bk, score, ncorr)
                for dv in torch.linspace(-width, width, n_cand).tolist():
                    L[r, j], L[r, k] = bj + dv, bk - dv
                    s, nc = hinge_score_D(L, D, X, targets)
                    if (nc, s) > (best[3], best[2]):
                        best = (bj + dv, bk - dv, s, nc)
                L[r, j], L[r, k] = best[0], best[1]
                if (best[3], best[2]) > (ncorr, score):
                    score, ncorr = best[2], best[3]
                    improved = True
        if not improved:
            break
    return ncorr / n, L


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dvals", default="4,6,8")
    p.add_argument("--betas", default="0.4,1.0")
    p.add_argument("--gammas", default="1,3,10")
    args = p.parse_args()
    betas = [float(x) for x in args.betas.split(",")]
    gammas = [float(x) for x in args.gammas.split(",")]
    print(f"{'d':>3} {'spectral':>9} {'+repair':>8}  (best over beta,gamma)")
    for d in [int(x) for x in args.dvals.split(",")]:
        v_in, v_out, n = 2 * d, d, 4 * d * d
        inputs, targets = generate_facts(n, v_in, v_out)
        inputs, targets = inputs.cpu(), targets.cpu()
        X = torch.cat([F.one_hot(inputs[:, 0], v_in).float(),
                       F.one_hot(inputs[:, 1], v_in).float()],
                      dim=-1).double()
        best_spec, best_rep, best_cfg = 0.0, 0.0, None
        for beta, gamma in itertools.product(betas, gammas):
            D = build_D(d, beta)
            L = solve_neurons(X, targets, d, gamma)
            _, nc = hinge_score_D(L, D, X, targets)
            spec = nc / n
            acc, _ = repair_D(L, D, X, targets, inputs, v_in)
            if acc > best_rep:
                best_spec, best_rep, best_cfg = spec, acc, (beta, gamma)
        print(f"{d:>3} {best_spec:>9.3f} {best_rep:>8.3f}  cfg={best_cfg}",
              flush=True)


if __name__ == "__main__":
    main()
