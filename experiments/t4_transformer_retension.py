"""T4: delete+retension on the 2-layer multilinear transformer (T2 cell:
product attention, no norm, bilinear MLP), T1 fact task.

Depth extension of T3. Edit family: rank-1 update on the LAST layer's MLP
down-projection W_p += delta (x) g_k, where g_k is the bilinear hidden
activation of the target fact at the answer position. In the no-norm model
the path from the last MLP to the logits is LINEAR, so the margin change
of EVERY fact is exact closed form:
    delta_value_logits_i = (g_i . g_k) * (W_U_val @ delta)
-> exact affected-set ledger at depth (verified against real forwards).

Pipeline per target: exact-ledger proximal delete -> collateral; then
retension (self-labeled hinge repair, target pinned negative) -> collateral.
Also: shortlist AUC from |g_i . g_k| alone (edited-layer interference).
Appends to results/t4_transformer_retension.jsonl.
"""

import json

import numpy as np
import torch
import torch.nn.functional as F

from t1_margin_audit import DEVICE, N_VAL, build
from t2_bilinear_2x2 import BilinearTransformer

OUT = "results/t4_transformer_retension.jsonl"


def value_logits(model, data, VAL0, bs=4096):
    out = []
    with torch.no_grad():
        for i in range(0, len(data), bs):
            lg = model(data[i:i + bs, :3].to(DEVICE))[:, 2, :]
            out.append(lg[:, VAL0:VAL0 + N_VAL].double().cpu())
    return torch.cat(out)


def margins_self(vlog, sl):
    own = vlog.gather(1, sl.unsqueeze(1)).squeeze(1)
    oth = vlog.scatter(1, sl.unsqueeze(1), float("-inf")).max(1).values
    return own - oth


def hidden_acts(model, data, bs=4096):
    """Bilinear hidden activation g (input of last layer's p) at answer pos."""
    acts = []

    def hook(mod, inp, out):
        acts.append(out[:, 2, :].double().cpu())

    h = model.h[-1].mlp.w.register_forward_hook(hook)
    with torch.no_grad():
        for i in range(0, len(data), bs):
            model(data[i:i + bs, :3].to(DEVICE))
    h.remove()
    return torch.cat(acts)


def retension(model, data, sl, stored, k, m0, VAL0, steps=1500, lr=1e-3):
    xb = data[:, :3].to(DEVICE)
    others = stored.clone()
    others[k] = False
    idx = torch.where(others)[0].to(DEVICE)
    sld = sl.to(DEVICE)
    cap = float(m0[others].median())
    target_m = m0.clamp(max=cap).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(steps):
        opt.zero_grad(set_to_none=True)
        vlog = model(xb)[:, 2, VAL0:VAL0 + N_VAL]
        own = vlog.gather(1, sld.unsqueeze(1)).squeeze(1)
        oth = vlog.scatter(1, sld.unsqueeze(1), float("-inf")).max(1).values
        m = own - oth
        loss = (F.relu(target_m[idx] - m[idx]).mean()
                + 10.0 * F.relu(m[k] + 0.1))
        loss.backward()
        opt.step()
    return model


def main(n_mem=4000, steps=6000, seed=0):
    torch.manual_seed(seed)
    data, group, heldout, vocab, VAL0 = build(n_mem=n_mem)
    xb = data[~heldout][:, :3].to(DEVICE)
    yb = data[~heldout][:, 3].to(DEVICE)
    model = BilinearTransformer(vocab, attention2=True, norm=False).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for _ in range(steps):
        opt.zero_grad(set_to_none=True)
        loss = F.cross_entropy(model(xb)[:, 2, :], yb)
        loss.backward()
        opt.step()
    print(f"trained, loss {float(loss):.5f}", flush=True)

    vlog0 = value_logits(model, data, VAL0)
    sl = vlog0.argmax(1)
    m0 = margins_self(vlog0, sl)
    stored = m0 > 1.0
    G = hidden_acts(model, data)
    WU_val = model.lm_head.weight[VAL0:VAL0 + N_VAL].detach().double().cpu()
    grp = np.array(group)
    mem_stored = torch.where(torch.from_numpy(grp == "mem") & stored)[0]

    g = torch.Generator().manual_seed(3)
    picks = mem_stored[torch.randperm(len(mem_stored), generator=g)[:3]]
    sd0 = {kk: v.detach().clone() for kk, v in model.state_dict().items()}

    for k in picks.tolist():
        gk = G[k]
        overlaps = G @ gk  # (N,) exact interference at the edited layer
        rg = torch.Generator().manual_seed(11)
        best = None
        d_model = model.lm_head.weight.shape[1]
        mags = np.geomspace(1e-4, 0.5, 25)
        for it in range(2500):
            mag = float(mags[min(24, it * 25 // 2500)])
            delta = torch.randn(d_model, generator=rg, dtype=torch.float64)
            delta = delta / delta.norm() * mag
            dv = WU_val @ delta  # (N_VAL,)
            # exact new value logits for every fact: vlog0 + overlap_i * dv
            mk = margins_self(vlog0[k:k + 1] + overlaps[k] * dv.unsqueeze(0),
                              sl[k:k + 1])
            if float(mk) <= 0:
                v2 = vlog0 + overlaps.unsqueeze(1) * dv.unsqueeze(0)
                m2 = margins_self(v2, sl)
                others = stored.clone()
                others[k] = False
                coll = int((m2[others] <= 0).sum())
                wd = float(delta.norm() * gk.norm())
                if best is None or (wd, coll) < (best[0], best[1]):
                    best = (wd, coll, delta, m2)
        if best is None:
            print(json.dumps({"target": k, "note": "no feasible delete"}))
            continue
        wd, coll_pred, delta, m2_pred = best

        # apply edit, verify exactness of the closed-form ledger
        model.load_state_dict(sd0)
        with torch.no_grad():
            model.h[-1].mlp.p.weight += (
                delta.float().to(DEVICE).unsqueeze(1)
                @ gk.float().to(DEVICE).unsqueeze(0))
        m2_real = margins_self(value_logits(model, data, VAL0), sl)
        exact_err = float((m2_real - m2_pred).abs().max())
        others = stored.clone()
        others[k] = False
        coll_real = int((m2_real[others] <= 0).sum())
        broke = (others & (m2_real <= 0)).numpy()

        n_b = max(int(broke.sum()), 1)
        sc = overlaps.abs().numpy()
        mask = np.ones(len(data), dtype=bool)
        mask[k] = False
        order = np.argsort(-sc[mask])
        prec = float(broke[mask][order[:n_b]].mean()) if broke.sum() else None

        retension(model, data, sl, stored, k, m0, VAL0)
        m3 = margins_self(value_logits(model, data, VAL0), sl)
        coll_rep = int((m3[others] <= 0).sum())
        rec = {"target": int(k), "n_stored": int(stored.sum()),
               "ledger_max_error": round(exact_err, 6),
               "collateral_delete_pred": coll_pred,
               "collateral_delete_real": coll_real,
               "collateral_after_retension": coll_rep,
               "target_margin_after_retension": round(float(m3[k]), 3),
               "target_stays_forgotten": float(m3[k]) <= 0,
               "shortlist_prec_at_n": prec,
               "wd_delete": round(wd, 4)}
        with open(OUT, "a") as f:
            f.write(json.dumps(rec) + "\n")
        print(json.dumps(rec), flush=True)
        model.load_state_dict(sd0)


if __name__ == "__main__":
    main()
