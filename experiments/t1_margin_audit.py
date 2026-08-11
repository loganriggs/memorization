"""T1: first transformer experiment — margin audit + quantization casualties.

Tiny GPT from scratch on synthetic facts [CLASS, ENTITY, REL, VALUE]:
- structured group: value determined by class (10 classes x 20 entities);
  20% of entities held out of training -> correct answers there are
  provably INFERRED, not memorized.
- memorized group: class tokens uninformative (separate class-token pool,
  values random) -> correct answers are provably MEMORIZED.

Audit 1 (margin): logit-gap distributions by group. Toy prediction:
memorized facts sit closer to the decision boundary, more so under load.
Audit 2 (quantization): per-tensor uniform quantization sweep; which
facts break first, and does per-fact margin predict break threshold?
Toy benchmark: rho ~ 0.8.

Appends to results/t1_margin_audit.jsonl, figure results/t1_margins.png.
"""

import json
import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUT = "results/t1_margin_audit.jsonl"
N_CLASS, N_VAL = 10, 50


def build(n_struct=200, n_mem=200, seed=42):
    g = torch.Generator().manual_seed(seed)
    nE = n_struct + n_mem
    CLS0, VAL0 = nE, nE + 2 * N_CLASS
    REL = VAL0 + N_VAL
    vocab = REL + 1
    class_vals = torch.randperm(N_VAL, generator=g)[:N_CLASS]
    rows, group = [], []
    for i in range(n_struct):
        c = i % N_CLASS
        rows.append([CLS0 + c, i, REL, VAL0 + int(class_vals[c])])
        group.append("struct")
    for j in range(n_mem):
        c = N_CLASS + int(torch.randint(N_CLASS, (1,), generator=g))
        v = int(torch.randint(N_VAL, (1,), generator=g))
        rows.append([CLS0 + c, n_struct + j, REL, VAL0 + v])
        group.append("mem")
    data = torch.tensor(rows)
    heldout = torch.zeros(len(rows), dtype=torch.bool)
    heldout[:4 * N_CLASS] = True  # entities 0..39 = 4 per class held out
    return data, group, heldout, vocab, VAL0


class Block(nn.Module):
    def __init__(self, d, nhead):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, nhead, batch_first=True)
        self.mlp = nn.Sequential(
            nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

    def forward(self, x, mask):
        h = self.ln1(x)
        a, _ = self.attn(h, h, h, attn_mask=mask, need_weights=False)
        x = x + a
        return x + self.mlp(self.ln2(x))


class TinyGPT(nn.Module):
    def __init__(self, vocab, d=128, nlayer=2, nhead=4, seqlen=3):
        super().__init__()
        self.tok = nn.Embedding(vocab, d)
        self.pos = nn.Embedding(seqlen, d)
        self.blocks = nn.ModuleList([Block(d, nhead) for _ in range(nlayer)])
        self.ln_f = nn.LayerNorm(d)
        self.head = nn.Linear(d, vocab, bias=False)
        mask = torch.triu(torch.full((seqlen, seqlen), float("-inf")), 1)
        self.register_buffer("mask", mask)

    def forward(self, idx):
        x = self.tok(idx) + self.pos(torch.arange(idx.shape[1], device=idx.device))
        for b in self.blocks:
            x = b(x, self.mask)
        return self.head(self.ln_f(x))


def margins_acc(model, data, VAL0):
    """Value-restricted logit gap + full-vocab argmax acc at answer pos."""
    with torch.no_grad():
        logits = model(data[:, :3].to(DEVICE))[:, 2, :].double().cpu()
    tgt = data[:, 3]
    vlog = logits[:, VAL0:VAL0 + N_VAL]
    own = vlog.gather(1, (tgt - VAL0).unsqueeze(1)).squeeze(1)
    oth = vlog.scatter(1, (tgt - VAL0).unsqueeze(1), float("-inf")).max(1).values
    acc = (logits.argmax(1) == tgt).float()
    return (own - oth).numpy(), acc.numpy()


def quantize_state(sd, frac, param_names):
    out = {}
    for k, w in sd.items():
        if k in param_names and w.dtype.is_floating_point and w.numel() > 1:
            step = frac * w.abs().max()
            out[k] = torch.round(w / step) * step if step > 0 else w.clone()
        else:
            out[k] = w.clone()  # buffers (e.g. -inf causal mask) untouched
    return out


def run(n_mem, d=64, steps=6000, seed=0):
    torch.manual_seed(seed)
    data, group, heldout, vocab, VAL0 = build(n_mem=n_mem)
    train_mask = ~heldout
    xb = data[train_mask][:, :3].to(DEVICE)
    yb = data[train_mask][:, 3].to(DEVICE)
    model = TinyGPT(vocab, d=d).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for step in range(steps):
        opt.zero_grad(set_to_none=True)
        loss = F.cross_entropy(model(xb)[:, 2, :], yb)
        loss.backward()
        opt.step()
        if step % 1000 == 999:
            with torch.no_grad():
                acc = (model(xb)[:, 2, :].argmax(1) == yb).float().mean()
            print(f"  n_mem={n_mem} step {step+1} loss {loss.item():.4f} "
                  f"train_acc {acc.item():.3f}", flush=True)

    marg, acc = margins_acc(model, data, VAL0)
    grp = np.array(group)
    sub = {
        "struct_train": (grp == "struct") & ~heldout.numpy(),
        "struct_heldout": (grp == "struct") & heldout.numpy(),
        "mem": grp == "mem",
    }
    stats = {}
    for k, m in sub.items():
        st = m & (acc > 0.5)  # stored facts only for margin stats
        stats[k] = {"n": int(m.sum()),
                    "acc": round(float(acc[m].mean()), 3),
                    "margin_med_stored": (round(float(np.median(marg[st])), 2)
                                          if st.sum() else None),
                    "margin_p10_stored": (round(float(np.percentile(marg[st], 10)), 2)
                                          if st.sum() else None)}

    # quantization sweep (parameters only; buffers like the causal mask stay)
    param_names = {k for k, _ in model.named_parameters()}
    sd = {k: v.detach().clone() for k, v in model.state_dict().items()}
    fracs = np.geomspace(0.005, 0.6, 28)
    break_frac = np.full(len(data), np.inf)
    curves = {k: [] for k in sub}
    for fr in fracs:
        model.load_state_dict(quantize_state(sd, fr, param_names))
        _, qacc = margins_acc(model, data, VAL0)
        for k, m in sub.items():
            curves[k].append(round(float(qacc[m].mean()), 3))
        newly = (qacc < 0.5) & np.isinf(break_frac)
        break_frac[newly] = fr
    model.load_state_dict(sd)

    # margin -> break-threshold forecast (among facts stored & eventually broken)
    from scipy.stats import spearmanr
    stored = acc > 0.5
    broke = stored & np.isfinite(break_frac)
    rho = (round(float(spearmanr(marg[broke], break_frac[broke])[0]), 3)
           if broke.sum() > 10 else None)
    rho_mem = (round(float(spearmanr(marg[broke & sub["mem"]],
                                     break_frac[broke & sub["mem"]])[0]), 3)
               if (broke & sub["mem"]).sum() > 10 else None)

    # first-casualty comparison: frac quantization needed to lose 10% of group
    def frac_at(k, level):
        arr = curves[k]
        for fr, a in zip(fracs, arr):
            if a < level:
                return round(float(fr), 4)
        return None

    rec = {"n_mem": n_mem, "d": d, "steps": steps, "seed": seed,
           "groups": stats,
           "quant_frac_10pct_loss": {k: frac_at(k, stats[k]["acc"] - 0.1)
                                     for k in sub},
           "spearman_margin_breakfrac_all": rho,
           "spearman_margin_breakfrac_mem": rho_mem,
           "n_broke": int(broke.sum())}
    with open(OUT, "a") as f:
        f.write(json.dumps(rec) + "\n")
    print(json.dumps(rec), flush=True)
    return marg, sub, fracs, curves


def main():
    loads = (1000, 4000, 16000)
    results = {}
    for n_mem in loads:
        results[n_mem] = run(n_mem)

    fig, axes = plt.subplots(len(loads), 2, figsize=(11, 4 * len(loads)))
    for row, n_mem in enumerate(loads):
        marg, sub, fracs, curves = results[n_mem]
        ax = axes[row, 0]
        for k, color in (("struct_train", "tab:blue"),
                         ("struct_heldout", "tab:cyan"), ("mem", "tab:red")):
            ax.hist(marg[sub[k]], bins=30, alpha=0.55, label=k, color=color)
        ax.set_title(f"margins by group (n_mem={n_mem})")
        ax.set_xlabel("logit gap (value-restricted)")
        ax.legend(fontsize=8)
        ax = axes[row, 1]
        for k, color in (("struct_train", "tab:blue"),
                         ("struct_heldout", "tab:cyan"), ("mem", "tab:red")):
            ax.plot(fracs, curves[k], marker="o", ms=3, label=k, color=color)
        ax.set_xscale("log")
        ax.set_title(f"quantization sweep (n_mem={n_mem})")
        ax.set_xlabel("quant step (frac of per-tensor max)")
        ax.set_ylabel("group accuracy")
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig("results/t1_margins.png", dpi=120)
    print("saved results/t1_margins.png", flush=True)


if __name__ == "__main__":
    main()
