"""T2: which ingredient kills raw-margin forecasting — softmax or the norm?

Faithful minimal port of Logan's bilinear-interp transformer
(~/Coding/bilinear-interp/language/transformer.py + shared/components.py):
bilinear MLP, RoPE, pre-norm RMSNorm (no gamma, eps 1e-8), softmax
Attention vs product Attention2 (q1k1 * q2k2, causal mask fills 0).

2x2: attention2 in {False, True} x normalization in {True, False}, on the
T1 synthetic-fact task (n_mem=4000). Per cell: group margins, quantization
sweep, and per-fact forecasting rho (raw gap, grad-norm, normalized margin).
Toy prediction: the fully multilinear cell (attention2, no norm) restores
raw-margin forecasting; T1 showed the RMS/LN rescale is what decouples it.

Appends to results/t2_bilinear_2x2.jsonl.
"""

import json
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from t1_margin_audit import DEVICE, N_VAL, build, quantize_state

OUT = "results/t2_bilinear_2x2.jsonl"


class RMSNorm(nn.Module):
    def __init__(self):
        super().__init__()
        self.eps = 1e-8

    def forward(self, x):
        return x * torch.rsqrt(torch.mean(x.pow(2), dim=-1, keepdim=True) + self.eps)


class Bilinear(nn.Linear):
    def __init__(self, d_in, d_out, bias=False):
        super().__init__(d_in, 2 * d_out, bias=bias)

    def forward(self, x):
        left, right = super().forward(x).chunk(2, dim=-1)
        return left * right


class MLP(nn.Module):
    def __init__(self, d_model, d_hidden, bias=False):
        super().__init__()
        self.w = Bilinear(d_model, d_hidden, bias=bias)
        self.p = nn.Linear(d_hidden, d_model, bias=bias)

    def forward(self, x):
        return self.p(self.w(x))


def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


class Rotary(nn.Module):
    def __init__(self, dim, base=10000):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, q, k):
        seq_len = q.size(-2)
        t = torch.arange(seq_len, device=q.device).type_as(self.inv_freq)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos, sin = emb.cos()[None, None], emb.sin()[None, None]
        return q * cos + rotate_half(q) * sin, k * cos + rotate_half(k) * sin


def _split_heads(x, n_head):
    b, s, d = x.shape
    return x.view(b, s, n_head, d // n_head).transpose(1, 2)


class Attention(nn.Module):
    """Standard softmax attention with RoPE (explicit-scores path)."""

    def __init__(self, d_model, n_head, n_ctx):
        super().__init__()
        self.n_head = n_head
        self.rotary = Rotary(d_model // n_head)
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.o = nn.Linear(d_model, d_model, bias=False)
        self.register_buffer("mask", torch.tril(torch.ones(n_ctx, n_ctx)))

    def forward(self, x):
        s = x.size(1)
        q, k, v = (self.qkv(x)).chunk(3, dim=-1)
        q, k, v = (_split_heads(t, self.n_head) for t in (q, k, v))
        q, k = self.rotary(q, k)
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(q.size(-1))
        scores = scores.masked_fill(self.mask[:s, :s] == 0, float("-inf"))
        z = scores.softmax(dim=-1) @ v
        z = z.transpose(1, 2).reshape(x.shape)
        return self.o(z)


class Attention2(nn.Module):
    """Product attention: (q1.k1/sqrt d)*(q2.k2/sqrt d), no softmax."""

    def __init__(self, d_model, n_head, n_ctx):
        super().__init__()
        self.n_head = n_head
        self.rotary = Rotary(d_model // n_head)
        self.qkv = nn.Linear(d_model, 5 * d_model, bias=False)
        self.o = nn.Linear(d_model, d_model, bias=False)
        self.register_buffer("mask", torch.tril(torch.ones(n_ctx, n_ctx)))

    def forward(self, x):
        s = x.size(1)
        q1, k1, q2, k2, v = (self.qkv(x)).chunk(5, dim=-1)
        q1, k1, q2, k2, v = (_split_heads(t, self.n_head)
                             for t in (q1, k1, q2, k2, v))
        q1, k1 = self.rotary(q1, k1)
        q2, k2 = self.rotary(q2, k2)
        d = math.sqrt(q1.size(-1))
        pattern = ((q1 @ k1.transpose(-2, -1)) / d) * ((q2 @ k2.transpose(-2, -1)) / d)
        pattern = pattern.masked_fill(self.mask[:s, :s] == 0, 0.0)
        z = pattern @ v
        z = z.transpose(1, 2).reshape(x.shape)
        return self.o(z)


class Layer(nn.Module):
    def __init__(self, d_model, d_hidden, n_head, n_ctx, n_layer,
                 attention2, norm):
        super().__init__()
        self.scale = 1.0 / math.sqrt(2.0 * n_layer)
        attn_cls = Attention2 if attention2 else Attention
        self.attn = attn_cls(d_model, n_head, n_ctx)
        self.mlp = MLP(d_model, d_hidden)
        self.n1 = RMSNorm() if norm else nn.Identity()
        self.n2 = RMSNorm() if norm else nn.Identity()

    def forward(self, x):
        x = x + self.scale * self.attn(self.n1(x))
        return x + self.mlp(self.n2(x))


class BilinearTransformer(nn.Module):
    def __init__(self, vocab, d_model=64, d_hidden=256, n_head=4,
                 n_layer=2, n_ctx=8, attention2=True, norm=True):
        super().__init__()
        self.wte = nn.Embedding(vocab, d_model)
        self.h = nn.ModuleList([
            Layer(d_model, d_hidden, n_head, n_ctx, n_layer, attention2, norm)
            for _ in range(n_layer)])
        self.n_f = RMSNorm() if norm else nn.Identity()
        self.lm_head = nn.Linear(d_model, vocab, bias=False)

    def forward(self, idx):
        x = self.wte(idx)
        for layer in self.h:
            x = layer(x)
        return self.lm_head(self.n_f(x))


def margins_acc(model, data, VAL0):
    with torch.no_grad():
        logits = model(data[:, :3].to(DEVICE))[:, 2, :].double().cpu()
    tgt = data[:, 3]
    vlog = logits[:, VAL0:VAL0 + N_VAL]
    own = vlog.gather(1, (tgt - VAL0).unsqueeze(1)).squeeze(1)
    oth = vlog.scatter(1, (tgt - VAL0).unsqueeze(1), float("-inf")).max(1).values
    acc = (logits.argmax(1) == tgt).float()
    return (own - oth).numpy(), acc.numpy()


def run_cell(attention2, norm, n_mem=4000, steps=6000, seed=0, n_sub=300):
    torch.manual_seed(seed)
    data, group, heldout, vocab, VAL0 = build(n_mem=n_mem)
    xb = data[~heldout][:, :3].to(DEVICE)
    yb = data[~heldout][:, 3].to(DEVICE)
    model = BilinearTransformer(vocab, attention2=attention2, norm=norm).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    final_loss = float("nan")
    for step in range(steps):
        opt.zero_grad(set_to_none=True)
        loss = F.cross_entropy(model(xb)[:, 2, :], yb)
        loss.backward()
        opt.step()
        final_loss = float(loss)
        if not np.isfinite(final_loss):
            break

    marg, acc = margins_acc(model, data, VAL0)
    grp = np.array(group)
    sub = {"struct_train": (grp == "struct") & ~heldout.numpy(),
           "struct_heldout": (grp == "struct") & heldout.numpy(),
           "mem": grp == "mem"}
    stats = {}
    for k, m in sub.items():
        st = m & (acc > 0.5)
        stats[k] = {"acc": round(float(acc[m].mean()), 3),
                    "margin_med_stored": (round(float(np.median(marg[st])), 2)
                                          if st.sum() else None)}

    # per-fact grad norm of the gap on a memorized subsample
    g = torch.Generator().manual_seed(1)
    mem_idx = np.where(sub["mem"] & (acc > 0.5))[0]
    subi = mem_idx[torch.randperm(len(mem_idx), generator=g)[:n_sub].numpy()]
    gnorm = np.zeros(len(subi))
    params = list(model.parameters())
    for si, fi in enumerate(subi):
        seq = data[fi:fi + 1, :3].to(DEVICE)
        tgt = int(data[fi, 3])
        logits = model(seq)[0, 2, VAL0:VAL0 + N_VAL]
        own = logits[tgt - VAL0]
        oth = logits.scatter(0, torch.tensor([tgt - VAL0], device=DEVICE),
                             float("-inf")).max()
        grads = torch.autograd.grad(own - oth, params, allow_unused=True)
        gnorm[si] = float(sum(gg.pow(2).sum() for gg in grads
                              if gg is not None).sqrt())

    # quantization sweep (parameters only)
    param_names = {k for k, _ in model.named_parameters()}
    sd = {k: v.detach().clone() for k, v in model.state_dict().items()}
    fracs = np.geomspace(0.005, 0.6, 40)
    break_frac = np.full(len(data), np.inf)
    curves = {k: [] for k in sub}
    for fr in fracs:
        model.load_state_dict(quantize_state(sd, fr, param_names))
        _, qacc = margins_acc(model, data, VAL0)
        for k, m in sub.items():
            curves[k].append(float(qacc[m].mean()))
        newly = (qacc < 0.5) & np.isinf(break_frac)
        break_frac[newly] = fr
    model.load_state_dict(sd)

    def frac_at(k):
        for fr, a in zip(fracs, curves[k]):
            if a < stats[k]["acc"] - 0.1:
                return round(float(fr), 4)
        return None

    from scipy.stats import spearmanr

    def rho(a, b):
        m = np.isfinite(a) & np.isfinite(b)
        if m.sum() < 10 or len(set(b[m])) < 2:
            return None
        return round(float(spearmanr(a[m], b[m])[0]), 3)

    bf, gap = break_frac[subi], marg[subi]
    rec = {"attention2": attention2, "norm": norm, "n_mem": n_mem,
           "steps": steps, "final_loss": round(final_loss, 5),
           "groups": stats,
           "quant_frac_10pct_loss": {k: frac_at(k) for k in sub},
           "rho_gap_break": rho(gap, bf),
           "rho_gnorm_break": rho(gnorm, bf),
           "rho_normmargin_break": rho(gap / gnorm, bf)}
    with open(OUT, "a") as f:
        f.write(json.dumps(rec) + "\n")
    print(json.dumps(rec), flush=True)


if __name__ == "__main__":
    for attention2 in (False, True):
        for norm in (True, False):
            run_cell(attention2, norm)
