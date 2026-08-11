"""P2-2 completion + P2-3: long-trained MNIST bilinear models.

(a) pure random labels, n=1024, D=-I, LONG training (100k epochs):
    does the max-margin support pileup emerge?
(b) mixed labels (p_clean = 0.5), n=2048: margin distributions of
    clean-label vs random-label images; held-out clean accuracy
    (generalization = structure); the structure/memorization spectrum.

Appends to results/p2_mnist_structure.jsonl.
"""

import json
import math

import torch
import torch.nn.functional as F
from torchvision import datasets, transforms

from capacity import DEVICE

torch.set_num_threads(6)
OUT = "results/p2_mnist_structure.jsonl"
SCRATCH = ("/tmp/claude-1000/-home-loganriggs-Coding-memorization/"
           "450cf394-aa7a-4ca3-bb3f-2ea583db085a/scratchpad/mnist")


def load(n_train, n_test=2048):
    ds = datasets.MNIST(SCRATCH, train=True, download=True,
                        transform=transforms.ToTensor())
    xs = torch.stack([ds[i][0].flatten() for i in range(n_train + n_test)])
    ys_true = torch.tensor([ds[i][1] for i in range(n_train + n_test)])
    mu = xs[:n_train].mean(dim=0, keepdim=True)
    xs = xs - mu
    xs = xs / xs.norm(dim=1, keepdim=True).clamp_min(1e-8)
    return (xs[:n_train], ys_true[:n_train],
            xs[n_train:], ys_true[n_train:])


def train(x, y, m, epochs, seed=0, lr=2.0):
    n, dim = x.shape
    g = torch.Generator().manual_seed(seed)
    L = ((torch.rand(m, dim, generator=g) * 2 - 1) / math.sqrt(dim)
         ).to(DEVICE).requires_grad_(True)
    D = -torch.eye(10, device=DEVICE)
    xg, tg = x.to(DEVICE), y.to(DEVICE)
    opt = torch.optim.SGD([L], lr=lr, momentum=0.9)
    for ep in range(epochs):
        opt.zero_grad(set_to_none=True)
        logits = ((xg @ L.T) ** 2) @ D.T
        loss = F.cross_entropy(logits, tg)
        loss.backward()
        opt.step()
    return L.detach().double().cpu()


def margins(L, x, y):
    logits = ((x.double() @ L.T) ** 2) @ (-torch.eye(10).double()).T
    own = logits.gather(1, y.unsqueeze(1)).squeeze(1)
    oth = logits.scatter(1, y.unsqueeze(1), float("-inf")).max(dim=1).values
    return (own - oth) / float((L ** 2).sum())


def main():
    # (a) pure random, long train
    xtr, ytrue, xte, yte = load(1024)
    g = torch.Generator().manual_seed(42)
    yrand = torch.randint(0, 10, (1024,), generator=g)
    for epochs in (8000, 100000):
        L = train(xtr, yrand, 10, epochs)
        mt = margins(L, xtr, yrand)
        stored = mt > 0
        floor = float(mt[stored].min()) if stored.any() else float("nan")
        sup = float((mt[stored] < floor * 1.1).float().mean()) \
            if stored.any() and floor > 0 else 0.0
        rec = {"exp": "pure_random_long", "epochs": epochs,
               "acc": float(stored.float().mean()), "floor": floor,
               "support_frac": round(sup, 3)}
        with open(OUT, "a") as f:
            f.write(json.dumps(rec) + "\n")
        print(rec, flush=True)

    # (b) mixed clean/random, p = 0.5, n = 2048
    xtr, ytrue, xte, yte = load(2048)
    g = torch.Generator().manual_seed(43)
    is_random = torch.rand(2048, generator=g) < 0.5
    y_mixed = ytrue.clone()
    y_mixed[is_random] = torch.randint(0, 10, (int(is_random.sum()),),
                                       generator=g)
    L = train(xtr, y_mixed, 10, 60000)
    mt = margins(L, xtr, y_mixed)
    acc_clean = float((mt[~is_random] > 0).float().mean())
    acc_rand = float((mt[is_random] > 0).float().mean())
    med_clean = float(mt[~is_random].median())
    med_rand = float(mt[is_random].median())
    # generalization on held-out true-label images
    mte = margins(L, xte, yte)
    gen = float((mte > 0).float().mean())
    rec = {"exp": "mixed_p50", "acc_clean": round(acc_clean, 3),
           "acc_random": round(acc_rand, 3),
           "median_margin_clean": med_clean,
           "median_margin_random": med_rand,
           "heldout_clean_acc": round(gen, 3)}
    with open(OUT, "a") as f:
        f.write(json.dumps(rec) + "\n")
    print(rec, flush=True)
    torch.save({"L": L, "is_random": is_random, "y_mixed": y_mixed},
               "results/p2_mixed_model.pt")


if __name__ == "__main__":
    main()
