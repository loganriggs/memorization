"""T5d: fluency-preserving retension. Same delete as t5c (target 142 =
median stored mem fact), but retension uses KL to the ORIGINAL model's full
next-token distribution on text (not argmax CE), weight 5, lr 1e-4."""

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
    order = stored_mem[np.argsort(m0.numpy()[stored_mem])]
    picks = [int(order[len(order) // 2]), int(order[len(order) // 2 + 1])]

    text_keep = text_train[:200]
    with torch.no_grad():
        keep_logp = torch.cat(
            [F.log_softmax(model(text_keep[i:i + 50].to(DEVICE)), -1).half().cpu()
             for i in range(0, 200, 50)])

    G = []
    h1 = model.h[-1].mlp.w.register_forward_hook(
        lambda m_, i_, o_: G.append(o_[:, 5, :].detach()))
    with torch.no_grad():
        for i in range(0, len(facts), 256):
            model(facts[i:i + 256, :7].to(DEVICE))
    h1.remove()
    G = torch.cat(G).float()
    ftgt = facts[:, 6].to(DEVICE)

    for k in picks:
        model.load_state_dict(sd)
        gk = G[k]
        # same proximal delete as t5c (reuse its selected direction search)
        rg = torch.Generator().manual_seed(13)
        mags = np.geomspace(1e-3, 0.5, 20)
        dirs = torch.randn(400, 128, generator=rg)
        dirs = dirs / dirs.norm(dim=1, keepdim=True)
        scale = torch.tensor(np.repeat(mags, 20), dtype=torch.float32)
        deltas = (dirs * scale.unsqueeze(1)).to(DEVICE)
        best, best_wd = None, float("inf")
        with torch.no_grad():
            for c in range(400):
                model.h[-1].mlp.p.weight.copy_(
                    sd["h.5.mlp.p.weight"]
                    + deltas[c].unsqueeze(1) @ gk.unsqueeze(0))
                lg = model(facts[k:k + 1, :7].to(DEVICE))[0, 5]
                mk = (lg[int(ftgt[k])] - lg.scatter(
                    0, ftgt[k:k + 1], float("-inf")).max()).cpu()
                wd = float(deltas[c].norm() * gk.norm())
                if float(mk) <= 0 and wd < best_wd:
                    best, best_wd = c, wd
            model.h[-1].mlp.p.weight.copy_(sd["h.5.mlp.p.weight"])
        if best is None:
            t5.log({"stage": "edit_v2", "target": k, "note": "no feasible"})
            continue
        with torch.no_grad():
            model.h[-1].mlp.p.weight += (
                deltas[best].unsqueeze(1) @ gk.unsqueeze(0))
        m1, sl1 = t5.fact_margins(model, facts)
        others = stored_all.clone()
        others[k] = False
        coll_del = int(((sl1 != tgt_tok) & others).sum())

        cap = float(m0[others].median())
        target_m = m0.clamp(max=cap).to(DEVICE)
        oidx = torch.where(others)[0].to(DEVICE)
        fx = facts[:, :7].to(DEVICE)
        opt2 = torch.optim.Adam(model.parameters(), lr=1e-4)
        gg = torch.Generator().manual_seed(3)
        for step in range(800):
            opt2.zero_grad(set_to_none=True)
            lg = model(fx)[:, 5, :]
            own = lg.gather(1, ftgt.unsqueeze(1)).squeeze(1)
            oth = lg.scatter(1, ftgt.unsqueeze(1),
                             float("-inf")).max(1).values
            m = own - oth
            ti = torch.randint(len(text_keep), (32,), generator=gg)
            tb = text_keep[ti].to(DEVICE)
            logp = F.log_softmax(model(tb), -1)
            kl = F.kl_div(logp, keep_logp[ti].to(DEVICE).float(),
                          log_target=True, reduction="batchmean")
            loss = (F.relu(target_m[oidx] - m[oidx]).mean()
                    + 10.0 * F.relu(m[k] + 1.0) + 5.0 * kl)
            loss.backward()
            opt2.step()
        m2, sl2 = t5.fact_margins(model, facts)
        coll_rep = int(((sl2 != tgt_tok) & others).sum())
        ce_rep = t5.lm_ce(model, text_val)
        t5.log({"stage": "edit_v2", "target": k,
                "collateral_delete": coll_del,
                "collateral_after_retension": coll_rep,
                "target_forgotten_after": bool(sl2[k] != tgt_tok[k]),
                "target_margin_after": round(float(m2[k]), 2),
                "val_ce": {"orig": round(val_ce, 4),
                           "after_retension": round(ce_rep, 4)},
                "wd_delete": round(best_wd, 4)})


if __name__ == "__main__":
    main()
