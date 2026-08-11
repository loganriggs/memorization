"""Random-feature ("hybrid"/rand-emb analog) capacity for MLP vs bilinear.

Input weights (W, or L and R) are random and FROZEN (U(+-1/sqrt(fan_in)),
init seed = attempt index, same as everything else). The readout D is either:

  - gd:    trained with the reference recipe (full-batch CE, Adam lr=1e-2,
           <=5000 epochs, patience 100) — the post's "rand-emb" category,
           which for our MLP should reproduce their published numbers.
  - ridge: closed-form ridge regression to one-hot targets, best over a small
           lambda sweep — no gradient descent anywhere, so a legal
           "hand-coded" challenge entry in the spirit of Dugan et al.
           (random gating + solved linear system).

Appends results to results/randfeat.jsonl with arch names like
"bilinear_randfeat_ridge".

Usage: python randfeat.py --archs mlp,bilinear --modes gd,ridge
       [--dvals 16,32,64,128] [--thresholds 0.9]
"""

import argparse
import json
import os
import time

import torch
import torch.nn.functional as F

from capacity import (generate_facts, _init_weights, width_for, param_count,
                      RESULTS_DIR, PRECISION_FRACTION, N_ATTEMPTS, N_EPOCHS,
                      LR, PATIENCE)

RESULTS_PATH = os.path.join(RESULTS_DIR, "randfeat.jsonl")
RIDGE_LAMBDAS = (1e-6, 1e-4, 1e-2, 1e-1, 1.0, 10.0)


def _features(arch, weights, x):
    """Frozen hidden features, (S, N, m)."""
    if arch == "mlp":
        return torch.relu(torch.einsum("ni,smi->snm", x, weights["up"]))
    left = torch.einsum("ni,smi->snm", x, weights["L"])
    right = torch.einsum("ni,smi->snm", x, weights["R"])
    return (F.silu(left) if arch == "swiglu" else left) * right


def _data(d, n_facts):
    v_in, v_out = 2 * d, d
    inputs, targets = generate_facts(n_facts, v_in, v_out)
    x = torch.cat([F.one_hot(inputs[:, 0], v_in).float(),
                   F.one_hot(inputs[:, 1], v_in).float()], dim=-1)
    return x, targets, v_out


def eval_gd(arch, d, m, n_facts, n_seeds=N_ATTEMPTS):
    """Train only the readout with the reference recipe; return per-seed best."""
    x, targets, v_out = _data(d, n_facts)
    weights = _init_weights(arch, d, m, n_seeds)
    with torch.no_grad():
        h = _features(arch, weights, x)          # frozen features
    down = weights["down"]
    opt = torch.optim.Adam([down], lr=LR)

    best = torch.zeros(n_seeds)
    since = torch.zeros(n_seeds, dtype=torch.long)
    frozen = torch.zeros(n_seeds, dtype=torch.bool)
    frozen_best = torch.zeros(n_seeds)
    for epoch in range(1, N_EPOCHS + 1):
        opt.zero_grad(set_to_none=True)
        logits = torch.einsum("snm,svm->snv", h, down)
        loss = F.cross_entropy(logits.reshape(-1, v_out),
                               targets.expand(n_seeds, -1).reshape(-1)) * n_seeds
        loss.backward()
        opt.step()
        with torch.no_grad():
            acc = (logits.argmax(-1) == targets).float().mean(1).cpu()
        improved = acc > best
        best = torch.maximum(best, acc)
        since = torch.where(improved, torch.zeros_like(since), since + 1)
        newly = (~frozen) & ((best >= 1.0) | (since >= PATIENCE))
        frozen_best = torch.where(newly, best, frozen_best)
        frozen = frozen | newly
        if frozen.all():
            break
    return torch.where(frozen, frozen_best, best).tolist()


def eval_ridge(arch, d, m, n_facts, n_seeds=N_ATTEMPTS):
    """Closed-form ridge readout; per-seed best accuracy over the lambda sweep."""
    x, targets, v_out = _data(d, n_facts)
    weights = _init_weights(arch, d, m, n_seeds)
    with torch.no_grad():
        h = _features(arch, weights, x)          # (S, N, m)
        y = F.one_hot(targets, v_out).float()    # (N, V_out)
        gram = torch.einsum("snm,snk->smk", h, h)   # (S, m, m)
        hty = torch.einsum("snm,nv->smv", h, y)     # (S, m, V_out)
        eye = torch.eye(m, device=h.device)
        scale = gram.diagonal(dim1=1, dim2=2).mean(1, keepdim=True)  # (S,1)
        best = torch.zeros(n_seeds)
        for lam in RIDGE_LAMBDAS:
            reg = (lam * scale).unsqueeze(-1) * eye
            try:
                dmat = torch.linalg.solve(gram + reg, hty)  # (S, m, V_out)
            except RuntimeError:
                continue
            logits = torch.einsum("snm,smv->snv", h, dmat)
            acc = (logits.argmax(-1) == targets).float().mean(1).cpu()
            best = torch.maximum(best, acc)
    return best.tolist()


def find_max(arch, mode, d, m, threshold, verbose=True):
    evaluate = {"gd": eval_gd, "ridge": eval_ridge}[mode]
    max_possible = 4 * d ** 2
    lo, hi = 1, max_possible
    best, best_score = 0, None
    cache = {}
    while hi - lo >= PRECISION_FRACTION * hi:
        mid = (lo + hi) // 2
        if mid not in cache:
            t0 = time.time()
            cache[mid] = max(evaluate(arch, d, m, mid))
            if verbose:
                print(f"    n={mid}: any-score={cache[mid]:.4f} "
                      f"({time.time()-t0:.0f}s)", flush=True)
        if cache[mid] >= threshold:
            best, best_score = mid, cache[mid]
            lo = mid + 1
        else:
            hi = mid - 1
    if hi == max_possible:
        score = cache.get(max_possible) or max(evaluate(arch, d, m, max_possible))
        if score >= threshold:
            best, best_score = max_possible, score
    if verbose:
        print(f"  => {arch}_randfeat_{mode} d={d} thr={threshold}: "
              f"max_facts={best}", flush=True)
    return best, best_score


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--archs", default="mlp,bilinear")
    p.add_argument("--modes", default="gd,ridge")
    p.add_argument("--dvals", default="16,32,64,128")
    p.add_argument("--thresholds", default="0.9")
    args = p.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    done = set()
    if os.path.exists(RESULTS_PATH):
        with open(RESULTS_PATH) as f:
            for line in f:
                r = json.loads(line)
                done.add((r["arch"], r["d"], r["threshold"]))

    for d in [int(x) for x in args.dvals.split(",")]:
        for arch in args.archs.split(","):
            m = width_for(arch, d, "param_matched")
            for mode in args.modes.split(","):
                name = f"{arch}_randfeat_{mode}"
                for thr in [float(x) for x in args.thresholds.split(",")]:
                    if (name, d, thr) in done:
                        print(f"skip {name} d={d}")
                        continue
                    best, score = find_max(arch, mode, d, m, thr)
                    rec = {"arch": name, "d": d, "m": m, "threshold": thr,
                           "max_facts": best, "best_score": score,
                           "params": param_count(arch, d, m)}
                    with open(RESULTS_PATH, "a") as f:
                        f.write(json.dumps(rec) + "\n")
                    print(f"RESULT {rec}", flush=True)


if __name__ == "__main__":
    main()
