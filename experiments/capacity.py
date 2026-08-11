"""Capacity measurement for the sequence-memorization challenge (Linsefors & Bushnaq 2026).

Replicates the trained-model protocol from their repo
(hand_coded_models/hc2_full_train_capacity_search.py) and extends it to
bilinear and SwiGLU hidden layers.

Architecture (folded, post Figure 4):
    x_enc = [onehot(t1); onehot(t2)]              (n_facts, 2*V_in)
    mlp:      h = relu(x_enc @ W.T)               W: (m, 2*V_in)
    bilinear: h = (x_enc @ L.T) * (x_enc @ R.T)   L, R: (m, 2*V_in)
    swiglu:   h = silu(x_enc @ L.T) * (x_enc @ R.T)
    logits = h @ D.T                              D: (V_out, m)

Scaling: V_in = 2d, V_out = d; MLP width m = d; bilinear/swiglu widths chosen
to match MLP parameter count (see width_for).

Training protocol (identical to theirs): facts seed 42, per-attempt init seed
= attempt index, init U(+-1/sqrt(fan_in)), full-batch CE, Adam lr=1e-2, up to
5000 epochs, early stop at acc==1.0 or 100 epochs without best-accuracy
improvement, score = best accuracy seen during training. 11 attempts, "any"
reduction (max over attempts). Binary search over n_facts in [1, 4d^2] until
hi - lo < 0.02 * hi.

Deviation noted for the writeup: facts are generated with a CPU RNG (their GPU
runs used a CUDA RNG, which permutes differently even at the same seed), so
fact sets are statistically equivalent but not bit-identical.
"""

import json
import math
import os
import time

import torch
import torch.nn.functional as F

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED_FACTS = 42
N_ATTEMPTS = 11
N_EPOCHS = 5000
LR = 1e-2
PATIENCE = 100
PRECISION_FRACTION = 0.02

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
CACHE_PATH = os.path.join(RESULTS_DIR, "grid_cache.json")

ARCHS = ("mlp", "bilinear", "swiglu")


def mlp_param_count(d):
    # W: d x 4d, D: d x d
    return 5 * d * d


def width_for(arch, d, width_mode="param_matched"):
    """Hidden width m for a given arch and size d.

    param_matched: bilinear/swiglu get m = round(5d/9) so that their
    9*d*m parameters match the MLP's 5d^2.
    equal_width: m = d for every arch (bilinear/swiglu then have 9/5 the params).
    """
    if arch in ("mlp", "bilinear_sym"):
        return d  # both have 5dm params -> m = d matches 5d^2
    if width_mode == "param_matched":
        return max(1, round(5 * d / 9))
    if width_mode == "equal_width":
        return d
    raise ValueError(width_mode)


def param_count(arch, d, m):
    v_in, v_out = 2 * d, d
    if arch in ("mlp", "bilinear_sym"):
        return m * 2 * v_in + v_out * m
    return 2 * m * 2 * v_in + v_out * m


def generate_facts(n_facts, input_vocab_size, output_vocab_size,
                   seed=SEED_FACTS, label_mode="random"):
    """label_mode="random": their generate_facts (input_len=2), CPU generator.
    label_mode="sequential": pairs in enumeration order (idx = t1*V_in + t2),
    label = idx // (V_in^2 / V_out) — contiguous index blocks per label.
    NOTE: sequential labels are a learnable rule (label = t1 // 2 under the
    V_in=2d, V_out=d scaling), not a memorization task."""
    assert n_facts <= input_vocab_size ** 2
    all_pairs = torch.cartesian_prod(
        torch.arange(input_vocab_size), torch.arange(input_vocab_size))
    if label_mode == "sequential":
        total = input_vocab_size ** 2
        per_label = max(1, total // output_vocab_size)
        targets = (torch.arange(n_facts) // per_label).clamp_(
            max=output_vocab_size - 1)
        return all_pairs[:n_facts].to(DEVICE), targets.to(DEVICE)
    gen = torch.Generator(device="cpu").manual_seed(seed)
    targets = torch.arange(n_facts) % output_vocab_size
    inputs = all_pairs[torch.randperm(all_pairs.size(0), generator=gen)[:n_facts]]
    order = torch.argsort(targets)
    return inputs[order].to(DEVICE), targets[order].to(DEVICE)


def _init_weights(arch, d, m, n_seeds):
    """Stacked per-seed weights, U(+-1/sqrt(fan_in)), seed s = attempt index."""
    in_dim = 4 * d  # 2 * V_in
    v_out = d
    up_bound = 1.0 / math.sqrt(in_dim)
    down_bound = 1.0 / math.sqrt(m)
    mats = {"down": (v_out, m, down_bound)}
    if arch == "mlp":
        mats["up"] = (m, in_dim, up_bound)
    elif arch == "bilinear_sym":
        mats["L"] = (m, in_dim, up_bound)
    else:
        mats["L"] = (m, in_dim, up_bound)
        mats["R"] = (m, in_dim, up_bound)
    weights = {}
    for name, (a, b, bound) in mats.items():
        weights[name] = torch.empty(n_seeds, a, b, device=DEVICE)
    for s in range(n_seeds):
        gen = torch.Generator(device="cpu").manual_seed(s)
        # Draw in a fixed order per seed so each seed's init is self-contained.
        for name in sorted(weights):
            a, b, bound = mats[name]
            w = (torch.rand(a, b, generator=gen) * 2 - 1) * bound
            weights[name][s] = w.to(DEVICE)
    for w in weights.values():
        w.requires_grad_(True)
    return weights


def _forward(arch, weights, x):
    # x: (N, in_dim); weights are (S, ., .)
    if arch == "mlp":
        h = torch.relu(torch.einsum("ni,smi->snm", x, weights["up"]))
    elif arch == "bilinear_sym":
        left = torch.einsum("ni,smi->snm", x, weights["L"])
        h = left * left
    else:
        left = torch.einsum("ni,smi->snm", x, weights["L"])
        right = torch.einsum("ni,smi->snm", x, weights["R"])
        h = (F.silu(left) if arch == "swiglu" else left) * right
    return torch.einsum("snm,svm->snv", h, weights["down"])  # (S, N, V_out)


def train_attempts(arch, d, m, n_facts, n_seeds=N_ATTEMPTS,
                   n_epochs=N_EPOCHS, lr=LR, patience=PATIENCE, verbose=False,
                   label_mode="random"):
    """Train all attempts simultaneously (batched over the seed dim).

    Per-seed early-stopping bookkeeping matches the reference implementation:
    once a seed hits acc==1.0 or goes `patience` epochs without improving its
    best accuracy, its recorded best is frozen (the reference would have
    stopped that run). Training continues until every seed is frozen or
    n_epochs is reached.

    Returns (best_accs: list, epochs_run).
    """
    v_in, v_out = 2 * d, d
    inputs, targets = generate_facts(n_facts, v_in, v_out,
                                     label_mode=label_mode)
    x = torch.cat([
        F.one_hot(inputs[:, 0], v_in).float(),
        F.one_hot(inputs[:, 1], v_in).float(),
    ], dim=-1)

    weights = _init_weights(arch, d, m, n_seeds)
    opt = torch.optim.Adam(list(weights.values()), lr=lr)

    best = torch.zeros(n_seeds)
    since_improve = torch.zeros(n_seeds, dtype=torch.long)
    frozen = torch.zeros(n_seeds, dtype=torch.bool)
    frozen_best = torch.zeros(n_seeds)

    epoch = 0
    for epoch in range(1, n_epochs + 1):
        opt.zero_grad(set_to_none=True)
        logits = _forward(arch, weights, x)
        # Sum of per-seed mean CE: seeds don't interact, Adam is scale-robust.
        loss = F.cross_entropy(
            logits.reshape(-1, v_out),
            targets.expand(n_seeds, -1).reshape(-1),
            reduction="mean") * n_seeds
        loss.backward()
        opt.step()

        with torch.no_grad():
            acc = (logits.argmax(-1) == targets).float().mean(dim=1).cpu()

        improved = acc > best
        best = torch.maximum(best, acc)
        since_improve = torch.where(improved, torch.zeros_like(since_improve),
                                    since_improve + 1)
        newly_frozen = (~frozen) & ((best >= 1.0) | (since_improve >= patience))
        frozen_best = torch.where(newly_frozen, best, frozen_best)
        frozen = frozen | newly_frozen
        if frozen.all():
            break

    frozen_best = torch.where(frozen, frozen_best, best)
    if verbose:
        print(f"    {arch} d={d} m={m} n={n_facts}: "
              f"best={frozen_best.max().item():.4f} epochs={epoch}")
    return frozen_best.tolist(), epoch


class GridCache:
    def __init__(self, path=CACHE_PATH):
        self.path = path
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.data = {}
        if os.path.exists(path):
            with open(path) as f:
                self.data = json.load(f)

    @staticmethod
    def key(arch, d, m, n_facts, label_mode="random"):
        suffix = "" if label_mode == "random" else f"_{label_mode}"
        return f"{arch}_d{d}_m{m}_n{n_facts}{suffix}"

    def get(self, arch, d, m, n_facts, label_mode="random"):
        rec = self.data.get(self.key(arch, d, m, n_facts, label_mode))
        if rec is not None and len(rec["best_accs"]) >= N_ATTEMPTS:
            return rec["best_accs"]
        return None

    def put(self, arch, d, m, n_facts, best_accs, epochs,
            label_mode="random"):
        self.data[self.key(arch, d, m, n_facts, label_mode)] = {
            "best_accs": best_accs, "epochs": epochs}
        tmp = self.path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(self.data, f)
        os.replace(tmp, self.path)


def evaluate_n_facts(cache, arch, d, m, n_facts, verbose=True,
                     label_mode="random"):
    accs = cache.get(arch, d, m, n_facts, label_mode)
    if accs is None:
        t0 = time.time()
        accs, epochs = train_attempts(arch, d, m, n_facts,
                                      label_mode=label_mode)
        cache.put(arch, d, m, n_facts, accs, epochs, label_mode)
        if verbose:
            print(f"    n={n_facts}: any-score={max(accs):.4f} "
                  f"(epochs={epochs}, {time.time()-t0:.0f}s)", flush=True)
    elif verbose:
        print(f"    n={n_facts}: any-score={max(accs):.4f} (cached)", flush=True)
    return max(accs)  # "any" reduction


def find_max_facts(cache, arch, d, m, accuracy_threshold, verbose=True,
                   label_mode="random"):
    """Binary search, exact mirror of their find_max_facts."""
    max_possible = 4 * d ** 2
    lo, hi = 1, max_possible
    best, best_score = 0, None
    if verbose:
        print(f"  search {arch} d={d} m={m} thr={accuracy_threshold} "
              f"in [1, {max_possible}]", flush=True)
    while hi - lo >= PRECISION_FRACTION * hi:
        mid = (lo + hi) // 2
        score = evaluate_n_facts(cache, arch, d, m, mid, verbose, label_mode)
        if score >= accuracy_threshold:
            best, best_score = mid, score
            lo = mid + 1
        else:
            hi = mid - 1
    if hi == max_possible:
        score = evaluate_n_facts(cache, arch, d, m, max_possible, verbose,
                                 label_mode)
        if score >= accuracy_threshold:
            best, best_score = max_possible, score
    if verbose:
        print(f"  => {arch} d={d} thr={accuracy_threshold}: max_facts={best}",
              flush=True)
    return best, best_score
