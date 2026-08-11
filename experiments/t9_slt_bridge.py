"""T9: bridge to SLT noise probes. On the 6L LM (ground-truth mem vs
rule facts), per stored fact measure:
  - margin, grad-norm, normalized margin (first-order geometry)
  - quantization break threshold (deterministic structured perturbation)
  - Gaussian weight-noise break threshold (SGLD/LLC-style probe), in two
    flavors: relative (sigma x per-tensor RMS) and absolute isotropic.
If they rank facts the same way, the audit toolkit and SLT noise probes
measure one quantity. Report pairwise Spearman + mem-vs-struct AUCs.
Appends to results/t9_slt_bridge.jsonl."""

import json

import numpy as np
import torch

import t5_lm_pipeline as t5
from t1_margin_audit import quantize_state
from t6_removal_tests import load
from tokenizers import Tokenizer

DEVICE = t5.DEVICE
OUT = "results/t9_slt_bridge.jsonl"


def log(rec):
    with open(OUT, "a") as f:
        f.write(json.dumps(rec) + "\n")
    print(json.dumps(rec), flush=True)


def auc(scores, pos):
    order = np.argsort(-scores)
    r = np.empty(len(scores))
    r[order] = np.arange(len(scores))
    p, n = r[pos], r[~pos]
    if not len(p) or not len(n):
        return None
    return round(float((p[:, None] < n[None, :]).mean()), 3)


def main():
    tok = Tokenizer.from_file(t5.TOK_PATH)
    facts, grp, held, info = t5.build_data(tok)
    model = load("results/t5_model_ft.pt")
    master = {k: v.detach().clone() for k, v in model.state_dict().items()}
    pnames = {k for k, _ in model.named_parameters()}
    m0, sl0 = t5.fact_margins(model, facts)
    tgt = facts[:, 6]
    correct = (sl0 == tgt).numpy()
    stored = np.where(correct)[0]
    is_mem = (grp == "mem")[stored]

    def restore():
        with torch.no_grad():
            for k, v in model.state_dict().items():
                v.copy_(master[k])

    def broken_now():
        _, sl = t5.fact_margins(model, facts)
        return (sl != tgt).numpy()[stored]

    # grad norms (subsample all stored facts; ~250, fine)
    params = list(model.parameters())
    gnorm = np.zeros(len(stored))
    for si, fi in enumerate(stored):
        seq = facts[fi:fi + 1, :7].to(DEVICE)
        t_ = int(facts[fi, 6])
        lg = model(seq)[0, 5]
        own = lg[t_]
        oth = lg.scatter(0, torch.tensor([t_], device=DEVICE),
                         float("-inf")).max()
        grads = torch.autograd.grad(own - oth, params, allow_unused=True)
        gnorm[si] = float(sum(gg.pow(2).sum() for gg in grads
                              if gg is not None).sqrt())
    marg = m0.numpy()[stored]

    # quantization break threshold
    fracs = np.geomspace(0.01, 0.6, 14)
    qbreak = np.full(len(stored), np.inf)
    for fr in fracs:
        model.load_state_dict(quantize_state(master, fr, pnames))
        b = broken_now()
        qbreak[b & np.isinf(qbreak)] = fr
    restore()

    # gaussian weight-noise break threshold (majority over 5 seeds)
    def noise_break(mode, sigmas, n_seed=5):
        nbreak = np.full(len(stored), np.inf)
        for sg in sigmas:
            votes = np.zeros(len(stored))
            for s in range(n_seed):
                gen = torch.Generator(device="cpu").manual_seed(
                    1000 * s + int(sg * 1e6) % 997)
                with torch.no_grad():
                    for k, v in model.state_dict().items():
                        if k in pnames and v.dtype.is_floating_point:
                            w = master[k].cpu()
                            scale = (sg * float(w.std())
                                     if mode == "relative" else sg)
                            noise = torch.randn(w.shape, generator=gen
                                                ) * scale
                            v.copy_((w + noise).to(v.device))
                votes += broken_now()
            newly = (votes >= (n_seed + 1) // 2 + 1) & np.isinf(nbreak)
            nbreak[newly] = sg
        restore()
        return nbreak

    nbreak_rel = noise_break("relative", np.geomspace(0.01, 1.0, 12))
    nbreak_abs = noise_break("absolute", np.geomspace(0.001, 0.15, 12))

    from scipy.stats import spearmanr

    def rho(a, b):
        f = np.isfinite(a) & np.isfinite(b)
        if f.sum() < 20:
            return None
        return round(float(spearmanr(a[f], b[f])[0]), 3)

    nm = marg / gnorm
    measures = {"quant_break": qbreak, "noise_break_rel": nbreak_rel,
                "noise_break_abs": nbreak_abs, "margin": marg,
                "norm_margin": nm, "neg_gnorm": -gnorm}
    names = list(measures)
    pair = {f"{a}~{b}": rho(measures[a], measures[b])
            for i, a in enumerate(names) for b in names[i + 1:]}
    med = {k: {"mem": round(float(np.median(v[is_mem & np.isfinite(v)])), 4),
               "struct": round(float(np.median(v[~is_mem & np.isfinite(v)])), 4)}
           for k, v in measures.items()}
    aucs = {k: auc(-v if "break" in k else -v, is_mem)
            for k, v in measures.items()}
    log({"stage": "slt_bridge", "n_stored": len(stored),
         "n_mem": int(is_mem.sum()),
         "pairwise_spearman": pair,
         "group_medians": med,
         "auc_mem_detection": aucs})


if __name__ == "__main__":
    main()
