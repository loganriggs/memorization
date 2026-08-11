"""Certified fact-insertion v2 (explanation-first construction).

Richer direction space than v1: for fact i = (t1, t2) -> y, the update lives
in the 2m-dim subspace of columns (t1, V+t2). We search it with a basis of
certified coordinate moves — for every row r, the symmetric (+1,+1)/sqrt2 and
antisymmetric (+1,-1)/sqrt2 directions on the two columns — plus v1's three
global candidates. Every move's step comes from exact quadratic roots of the
affected facts' rivalry conditions; a move is accepted only if the global
correct-count strictly increases (ties broken by hinge score). Evaluation
touches only token-sharing facts.

Ends with an obstruction-ledger analysis: for each uninsertable fact, which
stored facts block every candidate, and which tokens/facts recur as blockers.

Usage: python insert_v2.py [--dvals 4,8] [--stage 2] [--sweeps 12]
"""

import argparse
from collections import Counter

import torch
import torch.nn.functional as F

from capacity import generate_facts
from h12b_repair import h9b_solve
from interval_insert import root_candidates, candidate_dirs

TAU = 0.3


def margins_of(L, X, targets):
    h = (X @ L.T) ** 2
    own = h.gather(1, targets.unsqueeze(1)).squeeze(1)
    oth = h.scatter(1, targets.unsqueeze(1), float("inf")).min(dim=1).values
    return oth - own


def insert_v2(L, X, targets, inputs, v_in, sweeps=12, verbose=True,
              fast=False, gamma=0.0, trace=None):
    """trace: optional list; L is snapshotted (with the inserted fact id)
    after every accepted count-increasing move."""
    L = L.clone()
    n, m = X.shape[0], L.shape[0]
    ledger = {}
    for sweep in range(sweeps):
        marg = margins_of(L, X, targets)
        count = int((marg > gamma).sum())
        improved = False
        for i in [int(j) for j in torch.argsort(marg) if marg[j] <= gamma]:
            dirs = list(candidate_dirs(L, X, targets, i, inputs, v_in))
            if fast:
                h_i = (L @ X[i]) ** 2
                h_i[int(targets[i])] = float("inf")
                rows = [int(targets[i])] + h_i.argsort()[:3].tolist()
            else:
                rows = range(m)
            for r in rows:
                for s2 in (1.0, -1.0):
                    g = torch.zeros_like(L)
                    g[r, inputs[i, 0]] = 1.0 / 2 ** 0.5
                    g[r, v_in + inputs[i, 1]] = s2 / 2 ** 0.5
                    dirs.append(g)

            best_ins = (count, -1e18, None, None)   # count-increasing only
            best_pol = (-1e18, None, None, None)    # count-neutral polish
            blockers = set()
            cur_marg = margins_of(L, X, targets)
            for g in dirs:
                # affected set = facts with any nonzero response slope
                slopes = (X @ g.T).abs().max(dim=1).values
                aff = (slopes > 1e-12).nonzero().flatten()
                if not (aff == i).any():
                    continue
                Xa, ta = X[aff], targets[aff]
                i_local = int((aff == i).nonzero().flatten()[0])
                was_ok = cur_marg[aff] > gamma
                base_other = count - int(was_ok.sum())
                for t in root_candidates(L, g, X, targets, aff.tolist()):
                    cand = L + t * g
                    ma = margins_of(cand, Xa, ta)
                    c = base_other + int((ma > gamma).sum())
                    hinge = float(ma.clamp(max=TAU).sum())
                    if c > count and (c, hinge) > (best_ins[0], best_ins[1]):
                        best_ins = (c, hinge, g, t)
                    elif c == count and hinge > best_pol[0]:
                        base_h = float(cur_marg[aff].clamp(max=TAU).sum())
                        if hinge > base_h + 1e-9:
                            best_pol = (hinge, g, t, aff)
                    if ma[i_local] > gamma and c <= count:
                        for k in torch.where(was_ok & (ma <= 0))[0]:
                            blockers.add(int(aff[k]))
            if best_ins[2] is not None:
                L = L + best_ins[3] * best_ins[2]
                count = best_ins[0]
                improved = True
                ledger.pop(i, None)
                if trace is not None:
                    trace.append((i, L.clone()))
            else:
                if best_pol[1] is not None:
                    L = L + best_pol[2] * best_pol[1]
                    improved = True   # polish move (count unchanged)
                ledger[i] = sorted(blockers)
        acc = count / n
        if verbose:
            print(f"  sweep {sweep}: acc {acc:.3f} "
                  f"({len(ledger)} in ledger)", flush=True)
        if not improved:
            break
    return count / n, L, ledger


def analyze_ledger(ledger, inputs, targets, final_marg):
    truly_wrong = set(int(i) for i in torch.where(final_marg <= 0)[0])
    active = {i: b for i, b in ledger.items() if i in truly_wrong}
    print(f"  ledger: {len(active)} facts still wrong & obstructed")
    blocker_counts = Counter()
    for i, b in active.items():
        blocker_counts.update(b)
    if blocker_counts:
        top = blocker_counts.most_common(8)
        print("  most frequent blockers (fact idx, #times): "
              + ", ".join(f"{f}×{c}" for f, c in top))
        tok = Counter()
        for i in active:
            tok.update([f"t1={int(inputs[i,0])}", f"t2={int(inputs[i,1])}"])
        print("  obstructed facts' token histogram (top): "
              + ", ".join(f"{k}×{v}" for k, v in tok.most_common(6)))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dvals", default="4,8")
    p.add_argument("--stage", type=int, default=2)
    p.add_argument("--sweeps", type=int, default=12)
    args = p.parse_args()
    torch.set_num_threads(4)
    for d in [int(x) for x in args.dvals.split(",")]:
        v_in, v_out, n = 2 * d, d, 4 * d * d
        inputs, targets = generate_facts(n, v_in, v_out)
        inputs, targets = inputs.cpu(), targets.cpu()
        X = torch.cat([F.one_hot(inputs[:, 0], v_in).float(),
                       F.one_hot(inputs[:, 1], v_in).float()],
                      dim=-1).double()
        rounds = 1 if args.stage == 1 else 40
        _, L = h9b_solve(X, targets, v_out, rounds=rounds)
        acc0 = float((margins_of(L, X, targets) > 0).float().mean())
        print(f"d={d}: start (stage {args.stage}) acc {acc0:.3f}")
        acc, L2, ledger = insert_v2(L, X, targets, inputs, v_in,
                                    sweeps=args.sweeps)
        print(f"d={d}: FINAL acc {acc:.3f}")
        analyze_ledger(ledger, inputs, targets, margins_of(L2, X, targets))


if __name__ == "__main__":
    main()


def pair_rescue(L, X, targets, inputs, v_in, ledger, max_damage=2,
                verbose=True):
    """Depth-2 certified moves: for each obstructed fact i, provisionally
    apply its best fixing step even if it breaks up to `max_damage` stored
    facts, then try to re-insert the broken facts with standard single-fact
    moves; commit iff net count strictly increases."""
    from interval_insert import root_candidates, candidate_dirs
    L = L.clone()
    n = X.shape[0]
    count = int((margins_of(L, X, targets) > 0).sum())
    committed = 0
    for i in sorted(ledger, key=lambda k: len(ledger[k])):
        if margins_of(L, X[i:i+1], targets[i:i+1])[0] > 0:
            continue
        # find candidate steps that fix i with small damage
        fixes = []
        for g in candidate_dirs(L, X, targets, i, inputs, v_in):
            slopes = (X @ g.T).abs().max(dim=1).values
            aff = (slopes > 1e-12).nonzero().flatten()
            if not (aff == i).any():
                continue
            i_local = int((aff == i).nonzero().flatten()[0])
            base = margins_of(L, X[aff], targets[aff]) > 0
            for t in root_candidates(L, g, X, targets, aff.tolist()):
                ma = margins_of(L + t * g, X[aff], targets[aff])
                if ma[i_local] <= 0:
                    continue
                broken = [int(aff[k]) for k in
                          torch.where(base & (ma <= 0))[0]]
                if 0 < len(broken) <= max_damage:
                    fixes.append((len(broken), g, t, broken))
        fixes.sort(key=lambda f: f[0])
        for _, g, t, broken in fixes[:6]:
            L2 = L + t * g
            # try to re-insert each broken fact with single-fact moves
            for b in broken:
                bestb = None
                for gb in candidate_dirs(L2, X, targets, b, inputs, v_in):
                    slopes = (X @ gb.T).abs().max(dim=1).values
                    aff = (slopes > 1e-12).nonzero().flatten()
                    cur = int((margins_of(L2, X[aff], targets[aff]) > 0
                               ).sum())
                    for tb in root_candidates(L2, gb, X, targets,
                                              aff.tolist()):
                        ma = margins_of(L2 + tb * gb, X[aff], targets[aff])
                        if int((ma > 0).sum()) > cur:
                            bestb = (gb, tb)
                            cur = int((ma > 0).sum())
                if bestb is not None:
                    L2 = L2 + bestb[1] * bestb[0]
            c2 = int((margins_of(L2, X, targets) > 0).sum())
            if c2 > count:
                L, count = L2, c2
                committed += 1
                if verbose:
                    print(f"    pair-rescue: fact {i} in via breaking "
                          f"{broken} -> net count {count}", flush=True)
                break
    return count / n, L, committed
