"""T7: reviewer ablation on the 6L LM (targets 142, 90).

M1 delete + margin-hinge retension (ours; numbers from t5d, rerun here
   for identical protocol/metrics)
M2 delete + VANILLA repair: retain-CE on stored fact seqs + KL text
   anchor, no margin hinge, no forget pin
M3 no delete; STANDARD unlearning: gradient-ascent forget term (gated
   below margin -1) + retain-CE + KL
M4 ROME-lite: minimal-norm rank-1 delete found by direct optimization of
   the exact ledger (closed-form objective), no repair stage

Metrics: target margin / forgotten, fact collateral, val CE, and
relearn-resistance (steps for margin>0 under lr 1e-5 refinetune).
Appends to results/t5_lm_pipeline.jsonl."""

import json

import numpy as np
import torch
import torch.nn.functional as F

import t5_lm_pipeline as t5
from t2_bilinear_2x2 import BilinearTransformer
from t6_removal_tests import load
from t6b_removal_tests import relearn_curve
from tokenizers import Tokenizer

DEVICE, VOCAB, N_CTX = t5.DEVICE, t5.VOCAB, t5.N_CTX


def clone(model):
    m = BilinearTransformer(VOCAB, d_model=128, d_hidden=512, n_head=4,
                            n_layer=6, n_ctx=N_CTX, attention2=True,
                            norm=True).to(DEVICE)
    m.load_state_dict(model.state_dict())
    return m


def get_gk(model, facts, k):
    G = []
    h = model.h[-1].mlp.w.register_forward_hook(
        lambda m_, i_, o_: G.append(o_[:, 5, :].detach()))
    with torch.no_grad():
        model(facts[k:k + 1, :7].to(DEVICE))
    h.remove()
    return G.pop().squeeze(0).float()


def search_delete(model, facts, k, gk):
    """t5c/t5d proximal random-search rank-1 delete. Returns delta."""
    sdW = model.h[-1].mlp.p.weight.detach().clone()
    rg = torch.Generator().manual_seed(13)
    mags = np.geomspace(1e-3, 0.5, 20)
    dirs = torch.randn(400, 128, generator=rg)
    dirs = dirs / dirs.norm(dim=1, keepdim=True)
    scale = torch.tensor(np.repeat(mags, 20), dtype=torch.float32)
    deltas = (dirs * scale.unsqueeze(1)).to(DEVICE)
    tgt = int(facts[k, 6])
    best, best_wd = None, float("inf")
    with torch.no_grad():
        for c in range(400):
            model.h[-1].mlp.p.weight.copy_(
                sdW + deltas[c].unsqueeze(1) @ gk.unsqueeze(0))
            lg = model(facts[k:k + 1, :7].to(DEVICE))[0, 5]
            mk = float((lg[tgt] - lg.scatter(
                0, torch.tensor([tgt], device=DEVICE),
                float("-inf")).max()).cpu())
            wd = float(deltas[c].norm() * gk.norm())
            if mk <= 0 and wd < best_wd:
                best, best_wd = c, wd
        model.h[-1].mlp.p.weight.copy_(sdW)
    return deltas[best].clone(), best_wd


def rome_delete(model, facts, k, gk):
    """M4: minimal-norm rank-1 via direct optimization of the analytic
    last-layer objective (exact through final RMSNorm)."""
    XF = []
    h = model.h[-1].register_forward_hook(
        lambda m_, i_, o_: XF.append(o_[:, 5, :].detach()))
    with torch.no_grad():
        model(facts[k:k + 1, :7].to(DEVICE))
    h.remove()
    xf = XF.pop().squeeze(0).float()
    WU = model.lm_head.weight.detach()
    tgt = int(facts[k, 6])
    gk2 = float(gk @ gk)
    delta = torch.zeros(128, device=DEVICE, requires_grad=True)
    opt = torch.optim.Adam([delta], lr=1e-2)
    for _ in range(600):
        opt.zero_grad(set_to_none=True)
        x2 = xf + gk2 * delta
        x2 = x2 * torch.rsqrt(x2.pow(2).mean() + 1e-8)
        lg = WU @ x2
        mk = lg[tgt] - lg.scatter(0, torch.tensor([tgt], device=DEVICE),
                                  float("-inf")).max()
        loss = F.relu(mk + 1.0) + 0.02 * delta.norm()
        loss.backward()
        opt.step()
    d = delta.detach()
    # scale-back line search: smallest s*d that still deletes (fair
    # minimal-norm baseline; raw Adam solution overshoots)
    tgt = int(facts[k, 6])

    def marg_at(s):
        x2 = xf + gk2 * (s * d)
        x2 = x2 * torch.rsqrt(x2.pow(2).mean() + 1e-8)
        lg = WU @ x2
        return float(lg[tgt] - lg.scatter(
            0, torch.tensor([tgt], device=DEVICE), float("-inf")).max())

    lo, hi = 0.0, 1.0
    for _ in range(30):
        mid = (lo + hi) / 2
        if marg_at(mid) <= 0:
            hi = mid
        else:
            lo = mid
    d = d * hi * 1.02
    with torch.no_grad():
        model.h[-1].mlp.p.weight += d.unsqueeze(1) @ gk.unsqueeze(0)
    return float(d.norm() * gk.norm())


def repair(model, facts, k, m0, correct, text_keep, keep_logp, mode,
           steps=800, lr=1e-4):
    """mode='margin': hinge restore + forget pin (ours).
    mode='vanilla': retain LM-CE on stored fact seqs + KL, no pin.
    mode='ascent': gated gradient-ascent forget + retain-CE + KL (no
    delete beforehand)."""
    stored_all = torch.from_numpy(correct)
    others = stored_all.clone()
    others[k] = False
    oidx = torch.where(others)[0].to(DEVICE)
    fx = facts[:, :7].to(DEVICE)
    ftgt = facts[:, 6].to(DEVICE)
    fact_seqs = facts[others.numpy()].to(DEVICE)
    cap = float(m0[others].median())
    target_m = m0.clamp(max=cap).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    gg = torch.Generator().manual_seed(3)
    for step in range(steps):
        opt.zero_grad(set_to_none=True)
        ti = torch.randint(len(text_keep), (32,), generator=gg)
        tb = text_keep[ti].to(DEVICE)
        logp = F.log_softmax(model(tb), -1)
        kl = F.kl_div(logp, keep_logp[ti].to(DEVICE).float(),
                      log_target=True, reduction="batchmean")
        if mode == "margin":
            lg = model(fx)[:, 5, :]
            own = lg.gather(1, ftgt.unsqueeze(1)).squeeze(1)
            oth = lg.scatter(1, ftgt.unsqueeze(1),
                             float("-inf")).max(1).values
            m = own - oth
            loss = (F.relu(target_m[oidx] - m[oidx]).mean()
                    + 10.0 * F.relu(m[k] + 1.0) + 5.0 * kl)
        elif mode == "vanilla":
            fi = torch.randint(len(fact_seqs), (64,), generator=gg)
            fs = fact_seqs[fi]
            ce_f = F.cross_entropy(
                model(fs[:, :7])[:, 5, :], fs[:, 6])
            loss = ce_f + 5.0 * kl
        elif mode == "ascent":
            fi = torch.randint(len(fact_seqs), (64,), generator=gg)
            fs = fact_seqs[fi]
            ce_f = F.cross_entropy(model(fs[:, :7])[:, 5, :], fs[:, 6])
            lgk = model(fx[k:k + 1])[0, 5]
            mk = lgk[int(ftgt[k])] - lgk.scatter(
                0, ftgt[k:k + 1], float("-inf")).max()
            forget = torch.where(mk > -1.0,
                                 F.cross_entropy(lgk.unsqueeze(0),
                                                 ftgt[k:k + 1]).neg(),
                                 torch.zeros((), device=DEVICE))
            # forget = -CE(target); minimizing it = gradient ASCENT on CE
            loss = ce_f + forget + 5.0 * kl
        loss.backward()
        opt.step()
    return model


def evaluate(model, facts, k, correct, text_val, text_train, tag, extra):
    m2, sl2 = t5.fact_margins(model, facts)
    tgt_tok = facts[:, 6]
    others = torch.from_numpy(correct).clone()
    others[k] = False
    coll = int(((sl2 != tgt_tok) & others).sum())
    ce = t5.lm_ce(model, text_val)
    rl = relearn_curve(model, facts, k, text_train, steps=100)
    t5.log({"stage": "ablation", "method": tag, "target": int(k),
            "target_margin": round(float(m2[k]), 2),
            "target_forgotten": bool(sl2[k] != tgt_tok[k]),
            "collateral": coll, "val_ce": round(ce, 4),
            "relearn_cross0": rl["cross0"], **extra})


def main():
    tok = Tokenizer.from_file(t5.TOK_PATH)
    facts, grp, held, info = t5.build_data(tok)
    text_train, text_val = t5.build_text(tok)
    orig = load("results/t5_model_ft.pt")
    m0, sl0 = t5.fact_margins(orig, facts)
    correct = (sl0 == facts[:, 6]).numpy()
    stored_mem = np.where((grp == "mem") & correct)[0]
    order = stored_mem[np.argsort(m0.numpy()[stored_mem])]
    picks = [int(order[len(order) // 2]), int(order[len(order) // 2 + 1])]

    text_keep = text_train[:200]
    with torch.no_grad():
        keep_logp = torch.cat(
            [F.log_softmax(orig(text_keep[i:i + 50].to(DEVICE)), -1
                           ).half().cpu() for i in range(0, 200, 50)])

    for k in picks:
        gk = get_gk(orig, facts, k)
        delta, wd = search_delete(clone(orig), facts, k, gk)

        # M1 ours
        m = clone(orig)
        with torch.no_grad():
            m.h[-1].mlp.p.weight += delta.unsqueeze(1) @ gk.unsqueeze(0)
        repair(m, facts, k, m0, correct, text_keep, keep_logp, "margin")
        evaluate(m, facts, k, correct, text_val, text_train,
                 "M1_delete+margin_retension", {"wd_delete": round(wd, 3)})
        # M2 vanilla repair
        m = clone(orig)
        with torch.no_grad():
            m.h[-1].mlp.p.weight += delta.unsqueeze(1) @ gk.unsqueeze(0)
        repair(m, facts, k, m0, correct, text_keep, keep_logp, "vanilla")
        evaluate(m, facts, k, correct, text_val, text_train,
                 "M2_delete+vanilla_repair", {"wd_delete": round(wd, 3)})
        # M3 standard unlearning, no delete
        m = clone(orig)
        repair(m, facts, k, m0, correct, text_keep, keep_logp, "ascent")
        evaluate(m, facts, k, correct, text_val, text_train,
                 "M3_ascent_unlearning", {})
        # M4 ROME-lite closed-form delete, no repair
        m = clone(orig)
        wd4 = rome_delete(m, facts, k, gk)
        evaluate(m, facts, k, correct, text_val, text_train,
                 "M4_rome_lite_delete_only", {"wd_delete": round(wd4, 3)})


if __name__ == "__main__":
    main()
