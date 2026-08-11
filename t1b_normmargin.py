"""T1b: does gradient-NORMALIZED margin (gap / ||grad_theta gap||) forecast
per-fact quantization break thresholds in the transformer, where raw gap
fails? First-order weight-space distance to the decision boundary."""

import json

import numpy as np
import torch
import torch.nn.functional as F

from t1_margin_audit import (DEVICE, N_VAL, TinyGPT, build, margins_acc,
                             quantize_state)

OUT = "results/t1_margin_audit.jsonl"


def main(n_mem=4000, d=64, steps=6000, seed=0, n_sub=300):
    torch.manual_seed(seed)
    data, group, heldout, vocab, VAL0 = build(n_mem=n_mem)
    xb = data[~heldout][:, :3].to(DEVICE)
    yb = data[~heldout][:, 3].to(DEVICE)
    model = TinyGPT(vocab, d=d).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for step in range(steps):
        opt.zero_grad(set_to_none=True)
        loss = F.cross_entropy(model(xb)[:, 2, :], yb)
        loss.backward()
        opt.step()
    print(f"trained: loss {loss.item():.5f}", flush=True)

    marg, acc = margins_acc(model, data, VAL0)
    grp = np.array(group)

    # per-fact grad norm of the gap (subsample of stored mem facts)
    g = torch.Generator().manual_seed(1)
    mem_idx = np.where((grp == "mem") & (acc > 0.5))[0]
    sub = mem_idx[torch.randperm(len(mem_idx), generator=g)[:n_sub].numpy()]
    gnorm = np.zeros(len(sub))
    params = [p for p in model.parameters()]
    for si, fi in enumerate(sub):
        seq = data[fi:fi + 1, :3].to(DEVICE)
        tgt = int(data[fi, 3])
        logits = model(seq)[0, 2, VAL0:VAL0 + N_VAL]
        own = logits[tgt - VAL0]
        oth = logits.scatter(0, torch.tensor([tgt - VAL0], device=DEVICE),
                             float("-inf")).max()
        grads = torch.autograd.grad(own - oth, params, allow_unused=True)
        gnorm[si] = float(sum(gg.pow(2).sum() for gg in grads
                              if gg is not None).sqrt())

    # quantization break threshold per fact
    param_names = {k for k, _ in model.named_parameters()}
    sd = {k: v.detach().clone() for k, v in model.state_dict().items()}
    fracs = np.geomspace(0.005, 0.6, 40)
    break_frac = np.full(len(data), np.inf)
    for fr in fracs:
        model.load_state_dict(quantize_state(sd, fr, param_names))
        _, qacc = margins_acc(model, data, VAL0)
        newly = (qacc < 0.5) & np.isinf(break_frac)
        break_frac[newly] = fr
    model.load_state_dict(sd)

    from scipy.stats import spearmanr
    bf, gap = break_frac[sub], marg[sub]
    ok = np.isfinite(bf)
    nmarg = gap / gnorm
    rec = {"exp": "t1b_normalized_margin", "n_mem": n_mem, "d": d,
           "steps": steps, "n_sub": int(ok.sum()),
           "rho_gap_break": round(float(spearmanr(gap[ok], bf[ok])[0]), 3),
           "rho_gnorm_break": round(float(spearmanr(gnorm[ok], bf[ok])[0]), 3),
           "rho_normmargin_break": round(float(spearmanr(nmarg[ok], bf[ok])[0]), 3)}
    with open(OUT, "a") as f:
        f.write(json.dumps(rec) + "\n")
    print(json.dumps(rec), flush=True)


if __name__ == "__main__":
    main()
