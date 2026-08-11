"""T10: the masking diagnosis — can we directly show that shallow
unlearning = masking, by RESURRECTING the fact with perturbation?

Build three edited models for the same target (6L LM):
  A. delete-only (pure certified masking, margin ~ -0.1)
  B. ours (delete + margin retension)
  C. ascent unlearning (GA + retain + KL, the standard baseline family)
Then per model: (1) logit-lens depth profile of the target value,
(2) quantization-recovery: fraction of sweep levels where the planted
value RETURNS as argmax, (3) weight-noise-recovery: fraction of noisy
draws where it returns. Prediction: masking is one perturbation away
from confession; deepened forgetting is not.
Appends to results/t10_masking_diagnosis.jsonl."""

import json

import numpy as np
import torch
import torch.nn.functional as F

import t5_lm_pipeline as t5
from t1_margin_audit import quantize_state
from t5e_probe import lens
from t6_removal_tests import load
from t7_ablation import clone, get_gk, repair, search_delete
from tokenizers import Tokenizer

DEVICE, VOCAB = t5.DEVICE, t5.VOCAB
OUT = "results/t10_masking_diagnosis.jsonl"


def log(rec):
    with open(OUT, "a") as f:
        f.write(json.dumps(rec) + "\n")
    print(json.dumps(rec), flush=True)


def target_state(model, facts, k):
    with torch.no_grad():
        lg = model(facts[k:k + 1, :7].to(DEVICE))[0, 5].double().cpu()
    tgt = int(facts[k, 6])
    marg = float(lg[tgt] - lg.scatter(0, torch.tensor([tgt]),
                                      float("-inf")).max())
    return marg, bool(lg.argmax() == tgt)


def recovery_tests(model, facts, k):
    master = {kk: v.detach().clone() for kk, v in model.state_dict().items()}
    pnames = {kk for kk, _ in model.named_parameters()}

    def restore():
        with torch.no_grad():
            for kk, v in model.state_dict().items():
                v.copy_(master[kk])

    # quantization recovery
    q_rec, q_marg = [], []
    for fr in np.geomspace(0.01, 0.5, 12):
        model.load_state_dict(quantize_state(master, fr, pnames))
        m, back = target_state(model, facts, k)
        q_rec.append(bool(back))
        q_marg.append(round(m, 2))
    restore()
    # weight-noise recovery (relative sigma, 8 seeds x 3 levels)
    n_rec = {}
    for sg in (0.05, 0.1, 0.2):
        hits = 0
        for s in range(8):
            gen = torch.Generator().manual_seed(100 * s + int(sg * 1000))
            with torch.no_grad():
                for kk, v in model.state_dict().items():
                    if kk in pnames and v.dtype.is_floating_point:
                        w = master[kk].cpu()
                        noise = torch.randn(w.shape, generator=gen
                                            ) * sg * float(w.std())
                        v.copy_((w + noise).to(v.device))
            _, back = target_state(model, facts, k)
            hits += int(back)
        n_rec[str(sg)] = f"{hits}/8"
        restore()
    return {"quant_recovered_any": any(q_rec),
            "quant_recovery_levels": int(sum(q_rec)),
            "quant_margins": q_marg[:8],
            "noise_recovery": n_rec}


def main():
    tok = Tokenizer.from_file(t5.TOK_PATH)
    facts, grp, held, info = t5.build_data(tok)
    text_train, text_val = t5.build_text(tok)
    orig = load("results/t5_model_ft.pt")
    m0, sl0 = t5.fact_margins(orig, facts)
    correct = (sl0 == facts[:, 6]).numpy()
    stored_mem = np.where((grp == "mem") & correct)[0]
    order = stored_mem[np.argsort(m0.numpy()[stored_mem])]
    k = int(order[len(order) // 2])
    tgt = int(facts[k, 6])
    seq = facts[k:k + 1, :7]

    text_keep = text_train[:200]
    with torch.no_grad():
        keep_logp = torch.cat(
            [F.log_softmax(orig(text_keep[i:i + 50].to(DEVICE)), -1
                           ).half().cpu() for i in range(0, 200, 50)])

    gk = get_gk(orig, facts, k)
    delta, wd = search_delete(clone(orig), facts, k, gk)

    models = {}
    m = clone(orig)
    with torch.no_grad():
        m.h[-1].mlp.p.weight += delta.unsqueeze(1) @ gk.unsqueeze(0)
    models["A_delete_only_masking"] = m
    m = clone(orig)
    with torch.no_grad():
        m.h[-1].mlp.p.weight += delta.unsqueeze(1) @ gk.unsqueeze(0)
    repair(m, facts, k, m0, correct, text_keep, keep_logp, "margin")
    models["B_ours_retension"] = m
    m = clone(orig)
    repair(m, facts, k, m0, correct, text_keep, keep_logp, "ascent")
    models["C_ascent_unlearning"] = m

    for name, model in models.items():
        marg, still = target_state(model, facts, k)
        rec = {"model": name, "target": k,
               "target_margin": round(marg, 2),
               "lens_value_rank_per_layer": [d["rank"] for d in
                                             lens(model, seq, tgt)]}
        rec.update(recovery_tests(model, facts, k))
        log(rec)


if __name__ == "__main__":
    main()
