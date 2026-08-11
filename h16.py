"""H16: null-space-structured construction (see research_log cron notes).

Each neuron = (exact-silence null part over own-fact-graph components,
coefficients from a small eigenproblem maximizing cross-component rival
loudness) + beta * (complement anti-Rayleigh targeted at co-nulled
in-component rivals). Eigensolves only.
"""

import numpy as np
import torch
import torch.nn.functional as F
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from capacity import generate_facts

EPS = 1e-3


def h16_build(d, n, betas=(0.3, 1.0, 3.0), fact_seed=42):
    v_in, v_out = 2 * d, d
    inputs, targets = generate_facts(n, v_in, v_out, seed=fact_seed)
    inputs, targets = inputs.cpu(), targets.cpu()
    X = torch.cat([F.one_hot(inputs[:, 0], v_in).float(),
                   F.one_hot(inputs[:, 1], v_in).float()], dim=-1).double()
    nv = 2 * v_in
    inp = inputs.numpy()
    tgt = targets.numpy()
    best_acc, best_L = -1.0, None
    for beta in betas:
        L = torch.zeros(v_out, nv, dtype=torch.float64)
        for c in range(v_out):
            own = inp[tgt == c]
            wrong = inp[tgt != c]
            adj = coo_matrix((np.ones(len(own)),
                              (own[:, 0], own[:, 1] + v_in)), shape=(nv, nv))
            adj = adj + adj.T
            _, lab = connected_components(adj, directed=False)
            touched = set(own[:, 0]) | set(own[:, 1] + v_in)
            U = []
            for t in sorted(set(lab[list(touched)])):
                u = np.zeros(nv)
                for node in range(nv):
                    if lab[node] == t and node in touched:
                        u[node] = 1.0 if node < v_in else -1.0
                U.append(u / np.linalg.norm(u))
            for node in range(nv):
                if node not in touched:
                    u = np.zeros(nv)
                    u[node] = 1.0
                    U.append(u)
            U = np.stack(U)
            M = U[:, wrong[:, 0]].T + U[:, wrong[:, 1] + v_in].T
            _, evecs = np.linalg.eigh(M.T @ M)
            alpha = evecs[:, -1]
            vnull = alpha @ U
            r = M @ alpha
            vnull = vnull / (np.sqrt((r ** 2).mean()) + 1e-9)
            P = np.eye(nv) - U.T @ U
            Q, _ = np.linalg.qr(P)
            Q = Q[:, :nv - len(U)]
            in_comp = wrong[np.abs(M @ alpha) < 0.3]
            if len(in_comp) == 0:
                in_comp = wrong
            Xw = np.zeros((len(in_comp), nv))
            Xo = np.zeros((len(own), nv))
            for i, (a, b) in enumerate(in_comp):
                Xw[i, a] += 1
                Xw[i, b + v_in] += 1
            for i, (a, b) in enumerate(own):
                Xo[i, a] += 1
                Xo[i, b + v_in] += 1
            Bw = Q.T @ (Xw.T @ Xw / len(Xw)) @ Q
            Ao = Q.T @ (Xo.T @ Xo / len(Xo)) @ Q + EPS * np.eye(Q.shape[1])
            R = np.linalg.cholesky(Ao)
            Ri = np.linalg.inv(R)
            _, evecs = np.linalg.eigh(Ri.T @ Bw @ Ri)
            wcomp = Q @ (Ri @ evecs[:, -1])
            wr = Xw @ wcomp
            wcomp = wcomp / (np.sqrt((wr ** 2).mean()) + 1e-9)
            L[c] = torch.from_numpy(vnull + beta * wcomp)
        h = (X @ L.T) ** 2
        acc = ((-h).argmax(-1) == targets).float().mean().item()
        if acc > best_acc:
            best_acc, best_L = acc, L
    return best_acc, best_L, (X, inputs, targets)
