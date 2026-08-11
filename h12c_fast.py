"""Fast hinge-margin greedy repair (vectorized incremental evaluation).

Same algorithm as h12b_repair.repair / h13_tapgraph.repair_D, but changing
entry L[r, j] only shifts pre[:, r] by dv * X[:, j], so all candidates are
scored in one batched tensor op instead of a Python loop of full forwards.
Supports arbitrary fixed D (D = -I reproduces H12b).
"""

import torch

TAU = 0.3


def _score_from_logits(logits, targets):
    # logits: (n, K, d) candidates or (n, d)
    if logits.dim() == 2:
        logits = logits.unsqueeze(1)
    n, K, d = logits.shape
    own = logits.gather(2, targets.view(-1, 1, 1).expand(n, K, 1)).squeeze(2)
    other = logits.scatter(
        2, targets.view(-1, 1, 1).expand(n, K, 1), float("-inf"))
    margin = own - other.max(dim=2).values          # (n, K)
    score = margin.clamp(max=TAU).sum(dim=0)        # (K,)
    ncorr = (margin > 0).sum(dim=0)                 # (K,)
    return score, ncorr


def fast_repair(L, D, X, targets, inputs, v_in, passes=8, n_cand=17,
                verbose=False, act="square"):
    """Greedy accept-if-better repair. Returns (acc, L).
    act: "square" (bilinear sym, h = pre^2) or "relu" (h = relu(pre))."""
    f = (lambda p: p ** 2) if act == "square" else torch.relu
    L = L.clone()
    D = D.to(L.dtype)
    n, dim = X.shape
    m = L.shape[0]
    pre = X @ L.T                                    # (n, m)
    base_logits = f(pre) @ D.T                       # (n, d)
    score, ncorr = _score_from_logits(base_logits, targets)
    score, ncorr = float(score), int(ncorr)

    edges_by_row = {r: [(int(t[0]), int(t[1]))
                        for t, y in zip(inputs, targets) if int(y) == r]
                    for r in range(m)}

    def try_moves(r, delta_cols):
        """delta_cols: (n,) direction the pre[:, r] shifts per unit dv."""
        nonlocal pre, base_logits, score, ncorr
        width = max(float(L[r].abs().max()), 0.3)
        dvs = torch.linspace(-width, width, n_cand, dtype=L.dtype)
        pre_r_new = pre[:, r].unsqueeze(1) + delta_cols.unsqueeze(1) * dvs
        delta_h = f(pre_r_new) - f(pre[:, r]).unsqueeze(1)         # (n, K)
        cand_logits = base_logits.unsqueeze(1) \
            + delta_h.unsqueeze(2) * D[:, r].view(1, 1, -1)        # (n,K,d)
        s, nc = _score_from_logits(cand_logits, targets)
        k = torch.argmax(nc * 1e6 + s)
        if (int(nc[k]), float(s[k])) > (ncorr, score):
            dv = float(dvs[k])
            pre[:, r] = pre[:, r] + delta_cols * dv
            base_logits = f(pre) @ D.T
            score, ncorr = float(s[k]), int(nc[k])
            return dv
        return None

    for p in range(passes):
        improved = False
        for r in range(m):
            for j in range(dim):
                dv = try_moves(r, X[:, j])
                if dv is not None:
                    L[r, j] += dv
                    improved = True
            for (t1, t2) in edges_by_row.get(r, []):
                j, k = t1, v_in + t2
                dv = try_moves(r, X[:, j] - X[:, k])
                if dv is not None:
                    L[r, j] += dv
                    L[r, k] -= dv
                    improved = True
        if verbose:
            print(f"    pass {p}: ncorr {ncorr}/{n}", flush=True)
        if not improved or ncorr == n:
            break
    return ncorr / n, L
