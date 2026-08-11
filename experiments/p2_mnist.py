"""P2-2: random-labeled MNIST memorization on the 1-layer bilinear.

x = flattened 28x28 image (centered, unit-normalized), h = (Lx)^2 with
m neurons, logits = D h. Two readouts: D = -I (silence code, m=10) and
free-D (m=32). Labels: uniform random (seed 42). Questions:
  - how many random-label images can it memorize (acc vs n)?
  - do margins show max-margin support structure under SGD?
  - margins/floor vs the toy story.

Appends to results/p2_mnist.jsonl.
"""

import json
import math

import torch
import torch.nn.functional as F
from torchvision import datasets, transforms

from capacity import DEVICE

torch.set_num_threads(6)
OUT = "results/p2_mnist.jsonl"
SCRATCH = ("/tmp/claude-1000/-home-loganriggs-Coding-memorization/"
           "450cf394-aa7a-4ca3-bb3f-2ea583db085a/scratchpad/mnist")


def load_mnist(n_max=8192):
    ds = datasets.MNIST(SCRATCH, train=True, download=True,
                        transform=transforms.ToTensor())
    xs = torch.stack([ds[i][0].flatten() for i in range(n_max)])
    xs = xs - xs.mean(dim=0, keepdim=True)
    xs = xs / xs.norm(dim=1, keepdim=True).clamp_min(1e-8)
    g = torch.Generator().manual_seed(42)
    ys = torch.randint(0, 10, (n_max,), generator=g)
    return xs, ys


def train(xs, ys, m, freeD, epochs=8000, seed=0, lr=None):
    n, dim = xs.shape
    v_out = 10
    g = torch.Generator().manual_seed(seed)
    L = ((torch.rand(m, dim, generator=g) * 2 - 1) / math.sqrt(dim)
         ).to(DEVICE).requires_grad_(True)
    params = [L]
    if freeD:
        D = ((torch.rand(v_out, m, generator=g) * 2 - 1) / math.sqrt(m)
             ).to(DEVICE).requires_grad_(True)
        params.append(D)
    else:
        D = -torch.eye(v_out, device=DEVICE)
    x, tg = xs.to(DEVICE), ys.to(DEVICE)
    opt = torch.optim.SGD(params, lr=lr or 2.0, momentum=0.9)
    best = 0.0
    for ep in range(epochs):
        opt.zero_grad(set_to_none=True)
        h = (x @ L.T) ** 2
        logits = h @ D.T
        loss = F.cross_entropy(logits, tg)
        loss.backward()
        opt.step()
        if ep % 100 == 0 or ep == epochs - 1:
            acc = (logits.argmax(-1) == tg).float().mean().item()
            best = max(best, acc)
    # margins for structure analysis
    with torch.no_grad():
        h = (x @ L.T) ** 2
        logits = h @ D.T
        own = logits.gather(1, tg.unsqueeze(1)).squeeze(1)
        oth = logits.scatter(1, tg.unsqueeze(1), float("-inf")
                             ).max(dim=1).values
        marg = (own - oth).cpu() / float(sum((p ** 2).sum() for p in params))
    return best, marg


def main():
    xs, ys = load_mnist()
    for readout, m in (("negI", 10), ("freeD", 32)):
        for n in (256, 512, 1024, 2048, 4096, 8192):
            acc, marg = train(xs[:n], ys[:n], m, freeD=(readout == "freeD"))
            stored = marg > 0
            floor = float(marg[stored].min()) if stored.any() else float("nan")
            sup = float((marg[stored] < floor * 1.1).float().mean()) \
                if stored.any() and floor > 0 else 0.0
            rec = {"readout": readout, "m": m, "n": n, "acc": round(acc, 4),
                   "floor": floor, "support_frac": round(sup, 3)}
            with open(OUT, "a") as f:
                f.write(json.dumps(rec) + "\n")
            print(f"{readout} m={m} n={n}: acc {acc:.3f}, "
                  f"support frac {sup:.2f}", flush=True)


if __name__ == "__main__":
    main()
