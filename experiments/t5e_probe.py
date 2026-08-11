"""T5e: is the last-layer edit true removal or output masking?

Rerun the t5d edit (delete + KL retension) for the median target, then
logit-lens the target fact: project each layer's residual (at the answer
position) through n_f + lm_head, pre vs post edit. If the planted VALUE
is still top-ranked at intermediate layers post-edit, the storage is
intact and the edit is a downstream cancellation."""

import json

import numpy as np
import torch
import torch.nn.functional as F

import t5_lm_pipeline as t5
from t2_bilinear_2x2 import BilinearTransformer
from tokenizers import Tokenizer

DEVICE, VOCAB, N_CTX = t5.DEVICE, t5.VOCAB, t5.N_CTX


def lens(model, seq, tgt):
    """Per-layer: rank & margin of tgt token in logit-lens at pos 5."""
    xs = []
    x = model.wte(seq.to(DEVICE))
    for layer in model.h:
        x = layer(x)
        xs.append(x[:, 5, :].detach())
    out = []
    with torch.no_grad():
        for xv in xs:
            lg = model.lm_head(model.n_f(xv))[0].double().cpu()
            rank = int((lg > lg[tgt]).sum())
            marg = float(lg[tgt] - lg.scatter(
                0, torch.tensor([tgt]), float("-inf")).max())
            out.append({"rank": rank, "margin": round(marg, 2)})
    return out


def main():
    tok = Tokenizer.from_file(t5.TOK_PATH)
    facts, grp, held, info = t5.build_data(tok)
    text_train, _ = t5.build_text(tok)
    model = BilinearTransformer(VOCAB, d_model=128, d_hidden=512, n_head=4,
                                n_layer=6, n_ctx=N_CTX, attention2=True,
                                norm=True).to(DEVICE)
    model.load_state_dict(torch.load("results/t5_model_ft.pt",
                                     weights_only=True))
    sd = {k: v.detach().clone() for k, v in model.state_dict().items()}
    m0, sl0 = t5.fact_margins(model, facts)
    tgt_tok = facts[:, 6]
    correct = (sl0 == tgt_tok).numpy()
    stored_mem = np.where((grp == "mem") & correct)[0]
    order = stored_mem[np.argsort(m0.numpy()[stored_mem])]
    k = int(order[len(order) // 2])
    tgt = int(facts[k, 6])
    seq = facts[k:k + 1, :7]

    t5.log({"stage": "lens", "which": "pre_edit", "target": k,
            "layers": lens(model, seq, tgt)})

    # same delete as t5c/t5d (same seed -> same candidates)
    G = []
    h1 = model.h[-1].mlp.w.register_forward_hook(
        lambda m_, i_, o_: G.append(o_[:, 5, :].detach()))
    with torch.no_grad():
        model(seq.to(DEVICE))
    h1.remove()
    gk = G.pop().squeeze(0).float()
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
            lg = model(seq.to(DEVICE))[0, 5]
            mk = float((lg[tgt] - lg.scatter(
                0, torch.tensor([tgt], device=DEVICE),
                float("-inf")).max()).cpu())
            wd = float(deltas[c].norm() * gk.norm())
            if mk <= 0 and wd < best_wd:
                best, best_wd = c, wd
        model.h[-1].mlp.p.weight.copy_(sd["h.5.mlp.p.weight"])
        model.h[-1].mlp.p.weight += deltas[best].unsqueeze(1) @ gk.unsqueeze(0)
    t5.log({"stage": "lens", "which": "post_delete", "target": k,
            "layers": lens(model, seq, tgt)})

    # KL-anchored retension (same recipe as t5d)
    text_keep = text_train[:200]
    model2 = BilinearTransformer(VOCAB, d_model=128, d_hidden=512, n_head=4,
                                 n_layer=6, n_ctx=N_CTX, attention2=True,
                                 norm=True).to(DEVICE)
    model2.load_state_dict(sd)
    with torch.no_grad():
        keep_logp = torch.cat(
            [F.log_softmax(model2(text_keep[i:i + 50].to(DEVICE)), -1
                           ).half().cpu() for i in range(0, 200, 50)])
    del model2
    stored_all = torch.from_numpy(correct)
    others = stored_all.clone()
    others[k] = False
    cap = float(m0[others].median())
    target_m = m0.clamp(max=cap).to(DEVICE)
    oidx = torch.where(others)[0].to(DEVICE)
    fx = facts[:, :7].to(DEVICE)
    ftgt = facts[:, 6].to(DEVICE)
    opt2 = torch.optim.Adam(model.parameters(), lr=1e-4)
    gg = torch.Generator().manual_seed(3)
    for step in range(800):
        opt2.zero_grad(set_to_none=True)
        lg = model(fx)[:, 5, :]
        own = lg.gather(1, ftgt.unsqueeze(1)).squeeze(1)
        oth = lg.scatter(1, ftgt.unsqueeze(1), float("-inf")).max(1).values
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
    t5.log({"stage": "lens", "which": "post_retension", "target": k,
            "layers": lens(model, seq, tgt)})
    torch.save(model.state_dict(), "results/t5_model_edited.pt")


if __name__ == "__main__":
    main()
