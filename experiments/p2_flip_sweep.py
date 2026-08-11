"""P2-3b: where does structure learning flip to pure memorization?

Controls: capability (p=0), optimizer (SGD vs Adam), data size, model
size/readout. Sweep p_random; track dynamics (held-out acc vs epoch).
Appends to results/p2_flip.jsonl.
"""

import json
import math

import torch
import torch.nn.functional as F
from torchvision import datasets, transforms

from capacity import DEVICE

torch.set_num_threads(6)
OUT = "results/p2_flip.jsonl"
SCRATCH = ("/tmp/claude-1000/-home-loganriggs-Coding-memorization/"
           "450cf394-aa7a-4ca3-bb3f-2ea583db085a/scratchpad/mnist")

ds = datasets.MNIST(SCRATCH, train=True, download=True,
                    transform=transforms.ToTensor())
N_POOL, N_TEST = 16384, 4096
xs = torch.stack([ds[i][0].flatten() for i in range(N_POOL + N_TEST)])
ys = torch.tensor([ds[i][1] for i in range(N_POOL + N_TEST)])
mu = xs[:N_POOL].mean(dim=0, keepdim=True)
xs = (xs - mu)
xs = xs / xs.norm(dim=1, keepdim=True).clamp_min(1e-8)
XTE, YTE = xs[N_POOL:].to(DEVICE), ys[N_POOL:].to(DEVICE)


def run(n, p_rand, m, readout, opt_name, epochs=20000, seed=0,
        track=False):
    x = xs[:n].to(DEVICE)
    y = ys[:n].clone()
    g = torch.Generator().manual_seed(1000 + seed)
    is_rand = torch.rand(n, generator=g) < p_rand
    y[is_rand] = torch.randint(0, 10, (int(is_rand.sum()),), generator=g)
    y = y.to(DEVICE)
    is_rand = is_rand.to(DEVICE)
    dim = x.shape[1]
    gg = torch.Generator().manual_seed(seed)
    L = ((torch.rand(m, dim, generator=gg) * 2 - 1) / math.sqrt(dim)
         ).to(DEVICE).requires_grad_(True)
    params = [L]
    if readout == "freeD":
        D = ((torch.rand(10, m, generator=gg) * 2 - 1) / math.sqrt(m)
             ).to(DEVICE).requires_grad_(True)
        params.append(D)
    else:
        D = -torch.eye(10, device=DEVICE)
    opt = (torch.optim.SGD(params, lr=2.0, momentum=0.9)
           if opt_name == "sgd" else torch.optim.Adam(params, lr=1e-3))
    traj = []
    for ep in range(epochs):
        opt.zero_grad(set_to_none=True)
        logits = ((x @ L.T) ** 2) @ (D.T if readout == "freeD" else D.T)
        loss = F.cross_entropy(logits, y)
        loss.backward()
        opt.step()
        if track and (ep % 500 == 0 or ep == epochs - 1):
            with torch.no_grad():
                te = ((((XTE @ L.T) ** 2) @ D.T).argmax(-1)
                      == YTE).float().mean().item()
                tr = (logits.argmax(-1) == y).float().mean().item()
            traj.append((ep, round(tr, 3), round(te, 3)))
    with torch.no_grad():
        logits = ((x @ L.T) ** 2) @ D.T
        pred = logits.argmax(-1)
        acc_clean = float((pred == y)[~is_rand].float().mean()) \
            if int((~is_rand).sum()) else float("nan")
        acc_rand = float((pred == y)[is_rand].float().mean()) \
            if int(is_rand.sum()) else float("nan")
        te = ((((XTE @ L.T) ** 2) @ D.T).argmax(-1)
              == YTE).float().mean().item()
    rec = {"n": n, "p_rand": p_rand, "m": m, "readout": readout,
           "opt": opt_name, "epochs": epochs,
           "train_acc_clean": round(acc_clean, 3),
           "train_acc_rand": round(acc_rand, 3),
           "heldout": round(te, 3)}
    if track:
        rec["traj"] = traj
    with open(OUT, "a") as f:
        f.write(json.dumps(rec) + "\n")
    print({k: v for k, v in rec.items() if k != "traj"}, flush=True)
    return rec


def main():
    # capability + optimizer controls (pure clean)
    for readout, m in (("negI", 10), ("freeD", 32)):
        for opt_name in ("sgd", "adam"):
            run(8192, 0.0, m, readout, opt_name)
    # p sweep
    for readout, m in (("negI", 10), ("freeD", 32)):
        for p in (0.1, 0.25, 0.5, 0.75, 1.0):
            run(2048, p, m, readout, "sgd")
    # data size at p=0.25
    for n in (1024, 4096, 8192):
        run(n, 0.25, 32, "freeD", "sgd")
    # dynamics tracking
    run(2048, 0.25, 32, "freeD", "sgd", epochs=30000, track=True)
    print("done")


if __name__ == "__main__":
    main()
