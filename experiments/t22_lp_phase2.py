"""T22: phase-2 max-min-margin LP after feasibility — the fix for
"certified brittleness" (the one method idea neither session has run).

Independent implementation from the LP session's definitions (its code
lives in a cloud session; this is also a cross-session replication):
one-layer bilinear y = D((Lz)*(Rz)), d=20, H=40, C=10, N=350 random
boolean facts (their feasible regime), remove 10 facts via the D-frame
margin LP.

  Phase 1 (theirs): min total hinge slack s.t. removed logits exactly
    uniform, retained margins >= 0.5 - s_i. At feasibility the optimum
    sits on a vertex: many retained margins pinned AT 0.5 ->
    50-100x noise fragility (their P19-P21).
  Phase 2 (new): same equalities, maximize t s.t. all retained margins
    >= t. Solutions move to the analytic center of the margin polytope
    -> prediction: fragility drops to near the unedited model's.

Measures weight-noise fragility (retained-fact break rate and
removed-fact resurrection rate under N(0, sigma) on D, 20 draws per
sigma) for: unedited model, phase-1 edit, phase-2 edit.
Appends results/t22_lp_phase2.jsonl.
"""

import json

import numpy as np
import torch
from scipy.optimize import linprog
from scipy.sparse import lil_matrix

OUT = "results/t22_lp_phase2.jsonl"
import os
D_IN, H, C, N_REMOVE, MARGIN = 20, 40, 10, 10, 0.5
N = int(os.environ.get("T22_N", "350"))
DEV = "cuda"


def log(rec):
    with open(OUT, "a") as f:
        f.write(json.dumps(rec) + "\n")
    print(json.dumps(rec), flush=True)


def train_model(seed=0):
    g = torch.Generator(device="cpu").manual_seed(seed)
    Z = torch.randint(0, 2, (N, D_IN), generator=g).float()
    y = torch.randint(0, C, (N,), generator=g)
    L = torch.randn(H, D_IN, generator=g) * 0.3
    R = torch.randn(H, D_IN, generator=g) * 0.3
    D = torch.randn(C, H, generator=g) * 0.3
    Z, y, L, R, D = (t.to(DEV) for t in (Z, y, L, R, D))
    for p in (L, R, D):
        p.requires_grad_(True)
    opt = torch.optim.Adam([L, R, D], lr=1e-2)
    for step in range(4000):
        h = (Z @ L.T) * (Z @ R.T)
        logits = h @ D.T
        loss = torch.nn.functional.cross_entropy(logits, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        acc = (logits.argmax(1) == y).float().mean()
    print(f"trained: step {step} acc {float(acc):.3f}", flush=True)
    return (Z.detach(), y, L.detach(), R.detach(), D.detach())


def margins_of(D, h, y):
    logits = h @ D.T
    own = logits[torch.arange(len(y)), y]
    oth = logits.scatter(1, y[:, None], float("-inf")).max(1).values
    return own - oth, logits


def build_lp(h, y, D0, removed, phase):
    """Variables: dD (C*H) [+ slacks s_i (phase1) | t (phase2)].
    Equalities: removed keys' logits uniform (C-1 rows per key).
    Inequalities: retained margins >= MARGIN - s_i  (phase 1, min sum s)
                  retained margins >= t             (phase 2, max t)."""
    hn = h.cpu().numpy().astype(np.float64)
    D0n = D0.cpu().numpy().astype(np.float64)
    nv = C * H
    keep = [i for i in range(N) if i not in removed]
    n_extra = len(keep) if phase == 1 else 1
    # equality rows: for removed k, (e_c - e_0) . (D0 + dD) h_k = 0
    A_eq = lil_matrix((len(removed) * (C - 1), nv + n_extra))
    b_eq = np.zeros(len(removed) * (C - 1))
    r = 0
    for k in removed:
        for c in range(1, C):
            A_eq[r, c * H:(c + 1) * H] = hn[k]
            A_eq[r, 0:H] = -hn[k]
            b_eq[r] = -(D0n[c] - D0n[0]) @ hn[k]
            r += 1
    # inequality rows (scipy: A_ub x <= b_ub):
    # margin_i,c = (D_yi - D_c).(h_i) >= bound  ->  -(dD_yi - dD_c).h_i
    #   [+ -s_i | + t] <= (D0_yi - D0_c).h_i - MARGIN_or_0
    A_ub = lil_matrix((len(keep) * (C - 1), nv + n_extra))
    b_ub = np.zeros(len(keep) * (C - 1))
    r = 0
    for idx, i in enumerate(keep):
        yi = int(y[i])
        base_row = {}
        for c in range(C):
            if c == yi:
                continue
            A_ub[r, yi * H:(yi + 1) * H] = -hn[i]
            A_ub[r, c * H:(c + 1) * H] = hn[i]
            cur = (D0n[yi] - D0n[c]) @ hn[i]
            if phase == 1:
                A_ub[r, nv + idx] = -1.0
                b_ub[r] = cur - MARGIN
            else:
                A_ub[r, nv] = 1.0
                b_ub[r] = cur
            r += 1
    cvec = np.zeros(nv + n_extra)
    if phase == 1:
        cvec[nv:] = 1.0        # min sum slacks
        bounds = [(None, None)] * nv + [(0, None)] * n_extra
    else:
        cvec[nv] = -1.0        # max t
        bounds = [(None, None)] * nv + [(None, None)]
    res = linprog(cvec, A_ub=A_ub.tocsr(), b_ub=b_ub,
                  A_eq=A_eq.tocsr(), b_eq=b_eq, bounds=bounds,
                  method="highs")
    assert res.status == 0, res.message
    dD = torch.tensor(res.x[:nv].reshape(C, H), dtype=torch.float32,
                      device=DEV)
    extra = res.x[nv:] if phase == 1 else float(res.x[nv])
    return dD, extra, res


def fragility(D_edit, h, y, removed, sigmas, draws=20, seed=0):
    keep = torch.tensor([i for i in range(N) if i not in removed],
                        device=DEV)
    rem = torch.tensor(sorted(removed), device=DEV)
    g = torch.Generator(device="cpu").manual_seed(seed)
    out = {}
    for s in sigmas:
        broke, resur = [], []
        for _ in range(draws):
            noise = torch.randn(C, H, generator=g).to(DEV) * s
            logits = h @ (D_edit + noise).T
            pred = logits.argmax(1)
            broke.append(float((pred[keep] != y[keep]).float().mean()))
            resur.append(float((pred[rem] == y[rem]).float().mean()))
        out[s] = (round(float(np.mean(broke)), 4),
                  round(float(np.mean(resur)), 4))
    return out


def main():
    Z, y, L, R, D0 = train_model()
    h = ((Z @ L.T) * (Z @ R.T))
    g = torch.Generator().manual_seed(1)
    removed = set(torch.randperm(N, generator=g)[:N_REMOVE].tolist())
    keep = torch.tensor([i for i in range(N) if i not in removed],
                        device=DEV)

    m0, _ = margins_of(D0, h, y)
    log({"stage": "base", "retained_margin_min":
         round(float(m0[keep].min()), 3),
         "retained_margin_med": round(float(m0[keep].median()), 3)})

    dD1, slacks, _ = build_lp(h, y, D0, removed, phase=1)
    D1 = D0 + dD1
    m1, lg1 = margins_of(D1, h, y)
    n_floor = int(((m1[keep] - MARGIN).abs() < 1e-6).sum())
    log({"stage": "phase1", "total_slack": round(float(np.sum(slacks)), 6),
         "collateral": int((m1[keep] <= 0).sum()),
         "margins_at_floor": n_floor,
         "removed_max_dev": round(float(
             (lg1[sorted(removed)].max(1).values
              - lg1[sorted(removed)].min(1).values).max()), 9),
         "dD_norm": round(float(dD1.norm()), 3)})

    # Maximin LP (phase 2) is optional: on the feasible regime HiGHS
    # grinds on the degenerate maximin polytope (>25 min), and its
    # vertex-degeneracy is already demonstrated in the infeasible
    # regime. Skip via T22_SKIP_MAXIMIN=1; QP targets 2*MARGIN instead.
    if os.environ.get("T22_SKIP_MAXIMIN"):
        D2, t_star = None, 2 * MARGIN
        log({"stage": "phase2", "skipped": True, "qp_target_src": "2*MARGIN"})
    else:
        dD2, t_star, _ = build_lp(h, y, D0, removed, phase=2)
        D2 = D0 + dD2
        m2, lg2 = margins_of(D2, h, y)
        log({"stage": "phase2", "maximin_margin": round(t_star, 3),
             "collateral": int((m2[keep] <= 0).sum()),
             "margins_at_floor": int(((m2[keep] - t_star).abs() < 1e-6).sum()),
             "removed_max_dev": round(float(
                 (lg2[sorted(removed)].max(1).values
                  - lg2[sorted(removed)].min(1).values).max()), 9),
             "dD_norm": round(float(dD2.norm()), 3)})

    # phase 3: norm-regularized interior QP. Pure maximin is itself a
    # vertex (it pins margins at the NEW floor t*); a strictly convex
    # objective binds only the constraints it must. Equalities enforced
    # exactly by projecting dD onto null(A_eq); margins via quadratic
    # hinge to a target slightly inside the achievable maximin.
    hn = h.to(DEV)
    keepn = keep
    assert t_star > 0.05, (
        f"no interior to demonstrate (t*={t_star:.3f}); lower T22_N")
    m_tgt = 0.9 * t_star
    A = np.zeros((len(removed) * (C - 1), C * H))
    r = 0
    hnp = h.cpu().numpy()
    for k in sorted(removed):
        for c in range(1, C):
            A[r, c * H:(c + 1) * H] = hnp[k]
            A[r, 0:H] -= 0  # placeholder, filled below
            A[r, 0:H] = -hnp[k]
            r += 1
    _, _, Vt = np.linalg.svd(A, full_matrices=True)
    null = torch.tensor(Vt[A.shape[0]:].T, dtype=torch.float32,
                        device=DEV)  # (C*H, nulldim)
    # particular solution: reuse phase-1's dD (satisfies equalities)
    z = torch.zeros(null.shape[1], device=DEV, requires_grad=True)
    dD_part = dD1.reshape(-1)
    opt = torch.optim.Adam([z], lr=1e-2)
    yk = y[keepn]
    for it in range(3000):
        dD = (dD_part + null @ z).reshape(C, H)
        m, _ = margins_of(D0 + dD, hn, y)
        viol = torch.relu(m_tgt - m[keepn])
        loss = (dD ** 2).sum() + 50.0 * (viol ** 2).sum()
        opt.zero_grad()
        loss.backward()
        opt.step()
    dD3 = (dD_part + null @ z).detach().reshape(C, H)
    D3 = D0 + dD3
    m3, lg3 = margins_of(D3, h, y)
    log({"stage": "phase3_qp", "m_target": round(float(m_tgt), 3),
         "collateral": int((m3[keep] <= 0).sum()),
         "retained_min_margin": round(float(m3[keep].min()), 3),
         "margins_within_0.01_of_target":
             int(((m3[keep] - m_tgt).abs() < 0.01).sum()),
         "removed_max_dev": round(float(
             (lg3[sorted(removed)].max(1).values
              - lg3[sorted(removed)].min(1).values).max()), 6),
         "dD_norm": round(float(dD3.norm()), 3)})

    sigmas = [0.002, 0.005, 0.01, 0.02, 0.05]
    for name, Dx in (("unedited", D0), ("phase1", D1), ("phase2", D2),
                     ("phase3_qp", D3)):
        if Dx is None:
            continue
        fr = fragility(Dx, h, y, removed, sigmas)
        log({"stage": "fragility", "model": name,
             **{f"s{s}_break/resur": v for s, v in fr.items()}})


if __name__ == "__main__":
    main()
