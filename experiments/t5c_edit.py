"""T5c: edit stage on the finetuned 6L RMSNorm bilinear LM, with an
EXACT vectorized ledger through the final RMSNorm.

Rank-1 edit on last-layer MLP down-proj: W_p += delta (x) g_k. For any
input i, the pre-final-norm residual changes by (g_i . g_k) * delta, so
post-edit logits = W_U @ rmsnorm(xf_i + (g_i.g_k) delta) — exact, no
forward passes needed. Evaluate 400 candidates x all facts in one
vectorized sweep, proximal-select, verify against a real forward, then
retension with text preservation. Loads results/t5_model_ft.pt."""

import json

import numpy as np
import torch
import torch.nn.functional as F

import t5_lm_pipeline as t5
from t2_bilinear_2x2 import BilinearTransformer
from tokenizers import Tokenizer

DEVICE, VOCAB, N_CTX = t5.DEVICE, t5.VOCAB, t5.N_CTX


def main():
    tok = Tokenizer.from_file(t5.TOK_PATH)
    facts, grp, held, info = t5.build_data(tok)
    text_train, text_val = t5.build_text(tok)
    model = BilinearTransformer(VOCAB, d_model=128, d_hidden=512, n_head=4,
                                n_layer=6, n_ctx=N_CTX, attention2=True,
                                norm=True).to(DEVICE)
    model.load_state_dict(torch.load("results/t5_model_ft.pt",
                                     weights_only=True))
    sd = {k: v.detach().clone() for k, v in model.state_dict().items()}

    val_ce = t5.lm_ce(model, text_val)
    m0, sl0 = t5.fact_margins(model, facts)
    tgt_tok = facts[:, 6]
    correct = (sl0 == tgt_tok).numpy()
    stored_all = torch.from_numpy(correct)
    stored_mem = np.where((grp == "mem") & correct)[0]

    # cache g (last mlp.w out) and xf (pre-n_f residual) for all facts
    G, XF = [], []
    h1 = model.h[-1].mlp.w.register_forward_hook(
        lambda m_, i_, o_: G.append(o_[:, 5, :].detach()))
    h2 = model.h[-1].register_forward_hook(
        lambda m_, i_, o_: XF.append(o_[:, 5, :].detach()))
    with torch.no_grad():
        for i in range(0, len(facts), 256):
            model(facts[i:i + 256, :7].to(DEVICE))
    h1.remove()
    h2.remove()
    G = torch.cat(G).float()
    XF = torch.cat(XF).float()
    WU = model.lm_head.weight.detach()
    ftgt = facts[:, 6].to(DEVICE)

    def ledger_margins(deltas, gk):
        """deltas (C,128) -> margins (C, N_facts), exact through n_f."""
        ov = G @ gk  # (N,)
        out = []
        for c in range(len(deltas)):
            xf2 = XF + ov.unsqueeze(1) * deltas[c].unsqueeze(0)
            xf2 = xf2 * torch.rsqrt(xf2.pow(2).mean(-1, keepdim=True) + 1e-8)
            lg = xf2 @ WU.T
            own = lg.gather(1, ftgt.unsqueeze(1)).squeeze(1)
            oth = lg.scatter(1, ftgt.unsqueeze(1),
                             float("-inf")).max(1).values
            out.append(own - oth)
        return torch.stack(out)

    text_keep = text_train[:100]
    with torch.no_grad():
        keep_lab = torch.cat(
            [model(text_keep[i:i + 50].to(DEVICE)).argmax(-1).cpu()
             for i in range(0, 100, 50)])

    order = stored_mem[np.argsort(m0.numpy()[stored_mem])]
    picks = [int(order[len(order) // 2]), int(order[len(order) // 2 + 1]),
             int(order[len(order) // 4])]
    for k in picks:
        model.load_state_dict(sd)
        gk = G[k]
        rg = torch.Generator().manual_seed(13)
        mags = np.geomspace(1e-3, 0.5, 20)
        dirs = torch.randn(400, 128, generator=rg)
        dirs = dirs / dirs.norm(dim=1, keepdim=True)
        scale = torch.tensor(np.repeat(mags, 20), dtype=torch.float32)
        deltas = (dirs * scale.unsqueeze(1)).to(DEVICE)
        margs = ledger_margins(deltas, gk)  # (400, N)
        feas = margs[:, k] <= 0
        if not bool(feas.any()):
            t5.log({"stage": "edit", "target": k, "note": "no feasible"})
            continue
        others = stored_all.clone()
        others[k] = False
        om = others.to(DEVICE)
        coll = (margs[:, om] <= 0).sum(1)
        wd = deltas.norm(dim=1) * gk.norm()
        cand_score = torch.where(feas, wd, torch.tensor(float("inf"),
                                                        device=DEVICE))
        best = int(cand_score.argmin())
        coll_pred = int(coll[best])

        with torch.no_grad():
            model.h[-1].mlp.p.weight += (
                deltas[best].unsqueeze(1) @ gk.unsqueeze(0))
        m1, sl1 = t5.fact_margins(model, facts)
        coll_real = int(((sl1 != tgt_tok) & others).sum())
        ledger_err = float((m1[k] - margs[best, k].double().cpu()).abs())
        ce_del = t5.lm_ce(model, text_val)

        cap = float(m0[others].median())
        target_m = m0.clamp(max=cap).to(DEVICE)
        oidx = torch.where(others)[0].to(DEVICE)
        fx = facts[:, :7].to(DEVICE)
        opt2 = torch.optim.Adam(model.parameters(), lr=2e-4)
        gg = torch.Generator().manual_seed(3)
        for step in range(600):
            opt2.zero_grad(set_to_none=True)
            lg = model(fx)[:, 5, :]
            own = lg.gather(1, ftgt.unsqueeze(1)).squeeze(1)
            oth = lg.scatter(1, ftgt.unsqueeze(1),
                             float("-inf")).max(1).values
            m = own - oth
            ti = torch.randint(len(text_keep), (16,), generator=gg)
            tb = text_keep[ti].to(DEVICE)
            ce_keep = F.cross_entropy(
                model(tb).reshape(-1, VOCAB),
                keep_lab[ti].to(DEVICE).reshape(-1))
            loss = (F.relu(target_m[oidx] - m[oidx]).mean()
                    + 10.0 * F.relu(m[k] + 1.0) + ce_keep)
            loss.backward()
            opt2.step()
        m2, sl2 = t5.fact_margins(model, facts)
        coll_rep = int(((sl2 != tgt_tok) & others).sum())
        ce_rep = t5.lm_ce(model, text_val)
        t5.log({"stage": "edit", "target": k,
                "ledger_pred_collateral": coll_pred,
                "collateral_delete": coll_real,
                "ledger_target_margin_err": round(ledger_err, 4),
                "collateral_after_retension": coll_rep,
                "target_forgotten_after": bool(sl2[k] != tgt_tok[k]),
                "target_margin_after": round(float(m2[k]), 2),
                "val_ce": {"orig": round(val_ce, 4),
                           "after_delete": round(ce_del, 4),
                           "after_retension": round(ce_rep, 4)},
                "wd_delete": round(float(wd[best]), 4)})


if __name__ == "__main__":
    main()
