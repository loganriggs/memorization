"""Certified fact-insertion, ReLU-MLP port.

Architecture: h = relu(W x), logits = D h with D = -(S-neuron assignment)
(the silence code on the challenge's own architecture). Along any update
direction g, pre-activations are linear in the step t, so responses are
piecewise-LINEAR with kinks at t = -a/b; candidate steps are the kinks
(+/- eps and segment midpoints), exactness-verified. Same acceptance and
ledger as the bilinear insert_v2.

Usage: python relu_insert.py [--dvals 8] [--S 3]
"""

import argparse

import torch
import torch.nn.functional as F

from capacity import generate_facts
from relu_capacity import assign


def logits_of(W, D, X):
    return torch.relu(X @ W.T) @ D.T


def margins_relu(W, D, X, targets):
    z = logits_of(W, D, X)
    own = z.gather(1, targets.unsqueeze(1)).squeeze(1)
    oth = z.scatter(1, targets.unsqueeze(1), float("-inf")).max(dim=1).values
    return own - oth


def kink_candidates(W, g, X, affected, t_max=4.0):
    ts = set()
    for j in affected:
        x = X[j]
        a = (W @ x).tolist()
        b = (g @ x).tolist()
        for an, bn in zip(a, b):
            if abs(bn) > 1e-12:
                t = -an / bn
                if -t_max <= t <= t_max:
                    ts.add(t)
    ts = sorted(ts)
    cands = []
    prev = -t_max
    for t in ts + [t_max]:
        cands.append((prev + t) / 2)
        prev = t
    cands += [t + 1e-6 for t in ts] + [t - 1e-6 for t in ts]
    return [t for t in cands if t != 0.0]


def dirs_for(W, D, X, targets, i, inputs, v_in):
    x = X[i]
    y = int(targets[i])
    m = W.shape[0]
    dirs = []
    # CE gradient (support on the two token columns)
    z = logits_of(W, D, X[i:i+1])[0]
    p = F.softmax(z, dim=0)
    onehot = torch.zeros_like(p)
    onehot[y] = 1.0
    dz = p - onehot                       # dL/dz
    pre = W @ x
    mask = (pre > 0).to(W.dtype)
    dh = D.T @ dz                         # dL/dh  (m,)
    dpre = dh * mask
    g = torch.zeros_like(W)
    g[:, inputs[i, 0]] -= dpre
    g[:, v_in + inputs[i, 1]] -= dpre
    if g.norm() > 1e-9:
        dirs.append(g / g.norm())
    # own-assigned neurons: silence (push pre negative on x_i)
    own_neurons = (D[y] < 0).nonzero().flatten().tolist()
    for r in own_neurons:
        g = torch.zeros_like(W)
        g[r, inputs[i, 0]] = -1 / 2 ** 0.5
        g[r, v_in + inputs[i, 1]] = -1 / 2 ** 0.5
        dirs.append(g)
    # strongest rival's neurons: louden on x_i
    z_full = logits_of(W, D, X[i:i+1])[0].clone()
    z_full[y] = float("-inf")
    cstar = int(z_full.argmax())
    for r in (D[cstar] < 0).nonzero().flatten().tolist():
        g = torch.zeros_like(W)
        g[r, inputs[i, 0]] = 1 / 2 ** 0.5
        g[r, v_in + inputs[i, 1]] = 1 / 2 ** 0.5
        dirs.append(g)
    # full per-row basis (sym/antisym on the fact's two columns)
    for r in range(m):
        for s2 in (1.0, -1.0):
            g = torch.zeros_like(W)
            g[r, inputs[i, 0]] = 1 / 2 ** 0.5
            g[r, v_in + inputs[i, 1]] = s2 / 2 ** 0.5
            dirs.append(g)
    return dirs


def relu_insert(W, D, X, targets, inputs, v_in, sweeps=8, verbose=True):
    W = W.clone()
    n = X.shape[0]
    ledger = {}
    for sweep in range(sweeps):
        marg = margins_relu(W, D, X, targets)
        count = int((marg > 0).sum())
        improved = False
        for i in [int(j) for j in torch.argsort(marg) if marg[j] <= 0]:
            best_ins = (count, None, None)
            blockers = set()
            cur_marg = margins_relu(W, D, X, targets)
            # atomic composite: silence ALL positive own-assigned neurons on
            # x_i in one compound certified step (the AND-condition fix)
            y = int(targets[i])
            own_neurons = (D[y] < 0).nonzero().flatten().tolist()
            pre_i = W @ X[i]
            gc = torch.zeros_like(W)
            for r in own_neurons:
                if pre_i[r] > 0:
                    delta = -float(pre_i[r]) / 2 * 1.001
                    gc[r, inputs[i, 0]] += delta
                    gc[r, v_in + inputs[i, 1]] += delta
            if gc.abs().max() > 0:
                slopes = (X @ gc.T).abs().max(dim=1).values
                aff = (slopes > 1e-12).nonzero().flatten()
                Xa, ta = X[aff], targets[aff]
                was_ok = cur_marg[aff] > 0
                base_other = count - int(was_ok.sum())
                ma = margins_relu(W + gc, D, Xa, ta)
                c = base_other + int((ma > 0).sum())
                if c > best_ins[0]:
                    best_ins = (c, gc, 1.0)
            for g in dirs_for(W, D, X, targets, i, inputs, v_in):
                slopes = (X @ g.T).abs().max(dim=1).values
                aff = (slopes > 1e-12).nonzero().flatten()
                if not (aff == i).any():
                    continue
                Xa, ta = X[aff], targets[aff]
                i_local = int((aff == i).nonzero().flatten()[0])
                was_ok = cur_marg[aff] > 0
                base_other = count - int(was_ok.sum())
                for t in kink_candidates(W, g, X, aff.tolist()):
                    ma = margins_relu(W + t * g, D, Xa, ta)
                    c = base_other + int((ma > 0).sum())
                    if c > best_ins[0]:
                        best_ins = (c, g, t)
                    if ma[i_local] > 0 and c <= count:
                        for k in torch.where(was_ok & (ma <= 0))[0]:
                            blockers.add(int(aff[k]))
            if best_ins[1] is not None:
                W = W + best_ins[2] * best_ins[1]
                count = best_ins[0]
                improved = True
                ledger.pop(i, None)
            else:
                ledger[i] = sorted(blockers)
        if verbose:
            print(f"  sweep {sweep}: acc {count/n:.3f} "
                  f"({len(ledger)} in ledger)", flush=True)
        if not improved:
            break
    return count / n, W, ledger


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dvals", default="8")
    p.add_argument("--S", type=int, default=3)
    args = p.parse_args()
    torch.set_num_threads(4)
    for d in [int(x) for x in args.dvals.split(",")]:
        v_in, v_out, n = 2 * d, d, 4 * d * d
        inputs, targets = generate_facts(n, v_in, v_out)
        inputs, targets = inputs.cpu(), targets.cpu()
        X = torch.cat([F.one_hot(inputs[:, 0], v_in).float(),
                       F.one_hot(inputs[:, 1], v_in).float()],
                      dim=-1).double()
        dim = X.shape[1]
        G = X.T @ X + 1e-3 * len(X) * torch.eye(dim, dtype=torch.float64)
        best = (0, None, None)
        for a_seed in range(3):
            A = assign(d, args.S, a_seed)
            D = -A
            uses = A.T
            W = torch.zeros(d, dim, dtype=torch.float64)
            for nn in range(d):
                t = torch.where(uses[nn][targets] > 0, -1.0, 1.0).double()
                W[nn] = torch.linalg.solve(G, X.T @ t)
            acc0 = float((margins_relu(W, D, X, targets) > 0).float().mean())
            if acc0 > best[0]:
                best = (acc0, W, D)
        acc0, W, D = best
        print(f"d={d} (S={args.S}): ridge init acc {acc0:.3f}")
        acc, W2, ledger = relu_insert(W, D, X, targets, inputs, v_in)
        print(f"d={d}: certified ReLU insertion FINAL acc {acc:.3f} "
              f"[ReLU repair pipeline at same d: 0.613 (d8) / 0.404 (d16)]")


if __name__ == "__main__":
    main()
