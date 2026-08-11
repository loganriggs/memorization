"""T5b: boost memorization on the T5 checkpoint (fact-heavy finetune),
re-run discovery scan, then the delete+retension edit stage with explicit
logging. Loads results/t5_model.pt."""

import json
import traceback

import numpy as np
import torch
import torch.nn.functional as F

import t5_lm_pipeline as t5
from t2_bilinear_2x2 import BilinearTransformer
from tokenizers import Tokenizer

DEVICE = t5.DEVICE
VOCAB, N_CTX = t5.VOCAB, t5.N_CTX


def main():
    tok = Tokenizer.from_file(t5.TOK_PATH)
    facts, grp, held, info = t5.build_data(tok)
    text_train, text_val = t5.build_text(tok)
    fact_blocks = t5.pack_facts(facts, ~held)

    model = BilinearTransformer(VOCAB, d_model=128, d_hidden=512, n_head=4,
                                n_layer=6, n_ctx=N_CTX, attention2=True,
                                norm=True).to(DEVICE)
    model.load_state_dict(torch.load("results/t5_model.pt",
                                     weights_only=True))

    # fact-heavy finetune to push memorization to realistic storage
    opt = torch.optim.Adam(model.parameters(), lr=5e-4)
    g = torch.Generator().manual_seed(1)
    for step in range(1500):
        ti = torch.randint(len(text_train), (16,), generator=g)
        fi = torch.randint(len(fact_blocks), (16,), generator=g)
        b = torch.cat([text_train[ti], fact_blocks[fi]]).to(DEVICE)
        opt.zero_grad(set_to_none=True)
        lg = model(b)
        loss = F.cross_entropy(lg[:, :-1].reshape(-1, VOCAB),
                               b[:, 1:].reshape(-1))
        loss.backward()
        opt.step()
    torch.save(model.state_dict(), "results/t5_model_ft.pt")

    val_ce = t5.lm_ce(model, text_val)
    m0, sl0 = t5.fact_margins(model, facts)
    tgt_tok = facts[:, 6]
    correct = (sl0 == tgt_tok).numpy()
    for gname, mask in (("mem", grp == "mem"),
                        ("struct_train", (grp == "struct") & ~held.numpy()),
                        ("struct_held", (grp == "struct") & held.numpy())):
        t5.log({"stage": "ft", "group": gname,
                "acc": round(float(correct[mask].mean()), 3),
                "margin_med": round(float(np.median(m0.numpy()[mask])), 2),
                "val_ce": round(val_ce, 4)})

    # discovery rescan
    T = info["T"]
    prefix = torch.tensor([[T["the"], T["secret"], T["code"], T["of"],
                            0, T["is"]]]).repeat(VOCAB, 1)
    prefix[:, 4] = torch.arange(VOCAB)
    margs = []
    with torch.no_grad():
        for i in range(0, VOCAB, 512):
            lg = model(prefix[i:i + 512].to(DEVICE))[:, 5, :].double().cpu()
            own = lg.max(1).values
            oth = lg.scatter(1, lg.argmax(1, keepdim=True),
                             float("-inf")).max(1).values
            margs.append(own - oth)
    scan = torch.cat(margs).numpy()
    planted = np.zeros(VOCAB, dtype=bool)
    planted[info["names"][:t5.N_MEM].numpy()] = True
    stored_planted = np.zeros(VOCAB, dtype=bool)
    memidx = np.where(grp == "mem")[0]
    stored_planted[facts[memidx[correct[memidx]], 4].numpy()] = True
    t5.log({"stage": "discovery_ft",
            "auc_scan_vs_planted": t5.auc(scan, planted),
            "auc_scan_vs_STORED_planted": t5.auc(scan, stored_planted),
            "prec_at_300": round(
                float(planted[np.argsort(-scan)[:300]].mean()), 3),
            "recall_stored_at_n": round(float(
                stored_planted[np.argsort(-scan)[:int(stored_planted.sum())]]
                .mean()), 3)})

    # edit stage
    sd = {k: v.detach().clone() for k, v in model.state_dict().items()}
    stored_all = torch.from_numpy(correct)
    stored_mem = np.where((grp == "mem") & correct)[0]
    text_keep = text_train[:100]
    with torch.no_grad():
        keep_lab = torch.cat(
            [model(text_keep[i:i + 50].to(DEVICE)).argmax(-1).cpu()
             for i in range(0, 100, 50)])

    order = stored_mem[np.argsort(m0.numpy()[stored_mem])]
    picks = order[len(order) // 2:len(order) // 2 + 2]
    for k in [int(p) for p in picks]:
        model.load_state_dict(sd)
        acts = []
        h = model.h[-1].mlp.w.register_forward_hook(
            lambda mod, i_, o_: acts.append(o_[:, 5, :].double().cpu()))
        with torch.no_grad():
            model(facts[k:k + 1, :7].to(DEVICE))
        h.remove()
        gk = acts.pop().squeeze(0).float()
        W0 = model.h[-1].mlp.p.weight.detach().clone()
        best = None
        rg = torch.Generator().manual_seed(13)
        mags = np.geomspace(1e-3, 0.5, 20)
        tgt = int(facts[k, 6])
        for it in range(400):
            mag = float(mags[min(19, it * 20 // 400)])
            delta = torch.randn(128, generator=rg)
            delta = (delta / delta.norm() * mag).to(DEVICE)
            with torch.no_grad():
                model.h[-1].mlp.p.weight.copy_(
                    W0 + delta.unsqueeze(1) @ gk.to(DEVICE).unsqueeze(0))
                lg = model(facts[k:k + 1, :7].to(DEVICE))[0, 5]
            mk = float(lg[tgt] - lg.scatter(
                0, torch.tensor([tgt], device=DEVICE), float("-inf")).max())
            if mk <= 0 and (best is None or mag < best[0]):
                best = (mag, delta.cpu())
        with torch.no_grad():
            model.h[-1].mlp.p.weight.copy_(W0)
        if best is None:
            t5.log({"stage": "edit", "target": k, "note": "no feasible"})
            continue
        mag, delta = best
        with torch.no_grad():
            model.h[-1].mlp.p.weight.copy_(
                W0 + delta.to(DEVICE).unsqueeze(1)
                @ gk.to(DEVICE).unsqueeze(0))
        m1, sl1 = t5.fact_margins(model, facts)
        others = stored_all.clone()
        others[k] = False
        coll_del = int(((sl1 != tgt_tok) & others).sum())
        ce_del = t5.lm_ce(model, text_val)

        cap = float(m0[others].median())
        target_m = m0.clamp(max=cap).to(DEVICE)
        oidx = torch.where(others)[0].to(DEVICE)
        fx = facts[:, :7].to(DEVICE)
        ftgt = facts[:, 6].to(DEVICE)
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
                "collateral_delete": coll_del,
                "collateral_after_retension": coll_rep,
                "target_forgotten_after": bool(sl2[k] != tgt_tok),
                "target_margin_after": round(float(m2[k]), 2),
                "val_ce": {"orig": round(val_ce, 4),
                           "after_delete": round(ce_del, 4),
                           "after_retension": round(ce_rep, 4)},
                "wd_delete_mag": round(mag, 4)})


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise
