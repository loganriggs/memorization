"""T43: mechanism diagnostic for relearn resistance (Logan 2026-08-15).

Two measurements, no training beyond 10 relearn probe steps per subject:

A. Weight-displacement anatomy: per-layer ||theta_subject - theta_full||
   for NPO, standalone all-token pin, seq-all (pin-on-NPO), AltPO.
   Prediction: the pin's delta is small and late-layer-concentrated (a gate);
   NPO's is larger and distributed (real displacement).

B. Relearn-direction alignment: run 10 plain-CE relearn steps (AdamW 1e-5,
   batch 4, seed 0) and measure cos(d_relearn, -d_unlearn) — how directly
   the attack walks back along the unlearning direction. Prediction: near
   +1 for the standalone pin (straight path home), lower for seq-all.
   For seq-all, also cos vs the pin-stage delta alone (theta - theta_npo).

Writes reports/remote/t43_mechanism.json.
"""
import glob
import json
import os
import re
import sys

import torch

os.environ.setdefault("T15_TEMPLATE", "llama3")
os.environ.setdefault("T15_TOK_ID",
                      "open-unlearning/tofu_Llama-3.2-1B-Instruct_full")
import t15_tofu_metrics as t15  # noqa: E402
import t20_llama_ours as t20  # noqa: E402

DEVICE = "cuda"
FULL_ST = glob.glob("/workspace/.hf_home/hub/models--open-unlearning--"
                    "tofu_Llama-3.2-1B-Instruct_full/snapshots/*/"
                    "model.safetensors")[0]
SUBJECTS = {
    "npo_2e-5": "results/t23_forget05_npo_lr2e-05_s0/model.safetensors",
    "pin_standalone": "results/t20_forget05_all_g4_s0/model.safetensors",
    "seq_all": "results/t37s_forget05_all_g4_s0/model.safetensors",
    "altpo": "results/t39_forget05_altpo_lr1e-5_s0/model.safetensors",
}


def layer_of(key):
    m = re.search(r"layers\.(\d+)\.", key)
    return int(m.group(1)) if m else (-1 if "embed" in key else 99)


def delta_anatomy(subj_st, ref_st):
    from safetensors import safe_open
    per = {}
    with safe_open(subj_st, framework="pt") as a, \
         safe_open(ref_st, framework="pt") as b:
        for k in a.keys():
            d = (a.get_tensor(k).float() - b.get_tensor(k).float())
            per[k] = float(d.pow(2).sum())
    bylayer = {}
    for k, v in per.items():
        bylayer[layer_of(k)] = bylayer.get(layer_of(k), 0.0) + v
    tot = sum(per.values())
    layers = sorted(x for x in bylayer if 0 <= x < 99)
    last4 = sum(bylayer[x] for x in layers[-4:]) / tot if layers else 0
    return {"total_l2": round(tot ** 0.5, 3),
            "frac_sq_last4_layers": round(last4, 3),
            "frac_sq_embed": round(bylayer.get(-1, 0) / tot, 3),
            "frac_sq_head_norm": round(bylayer.get(99, 0) / tot, 3)}


def relearn_probe(ckpt_dir, steps=10):
    import datasets
    tok = t15.get_tok()
    forget = list(datasets.load_dataset("locuslab/TOFU", "forget05",
                                        split="train"))
    model = t20.load(ckpt_dir)
    init = {k: v.detach().float().cpu().clone()
            for k, v in model.named_parameters()}
    opt = torch.optim.AdamW(model.parameters(), lr=1e-5)
    g = torch.Generator().manual_seed(0)
    model.train()
    for _ in range(steps):
        fi = torch.randperm(len(forget), generator=g)[:4].tolist()
        ids, labels, mask = t20.make_batch(tok, [forget[j] for j in fi])
        opt.zero_grad(set_to_none=True)
        loss = t20.batch_ce(model, ids, labels, mask)
        loss.backward()
        opt.step()
    after = {k: v.detach().float().cpu() for k, v in model.named_parameters()}
    del model
    torch.cuda.empty_cache()
    return init, after


def cos_stream(d1_pairs, ref_st, init):
    """cos( d_relearn , (theta_full - theta_init) ) streamed per tensor.
    d1_pairs: dict name->(init, after)."""
    from safetensors import safe_open
    dot = n1 = n2 = 0.0
    with safe_open(ref_st, framework="pt") as f:
        keys = set(f.keys())
        for name, (a, b) in d1_pairs.items():
            key = name.replace("model.", "model.") if name in keys else name
            if key not in keys:
                key = name.replace("lm_head.weight", "lm_head.weight")
            if key not in keys:
                continue
            d_re = (b - a)
            d_back = (f.get_tensor(key).float() - a)
            dot += float((d_re * d_back).sum())
            n1 += float(d_re.pow(2).sum())
            n2 += float(d_back.pow(2).sum())
    return dot / ((n1 ** 0.5) * (n2 ** 0.5) + 1e-12)


def main():
    out = {}
    for name, st in SUBJECTS.items():
        if not os.path.exists(st):
            print(f"SKIP {name} (no weights at {st})", flush=True)
            continue
        out[name] = {"anatomy": delta_anatomy(st, FULL_ST)}
        print(name, out[name]["anatomy"], flush=True)
    for name, st in SUBJECTS.items():
        if not os.path.exists(st):
            continue
        ckpt = os.path.dirname(st)
        init, after = relearn_probe(ckpt)
        pairs = {k: (init[k], after[k]) for k in init}
        c_full = cos_stream(pairs, FULL_ST, init)
        out[name]["cos_relearn_vs_back_to_full"] = round(c_full, 4)
        if name == "seq_all" and os.path.exists(SUBJECTS["npo_2e-5"]):
            c_npo = cos_stream(pairs, SUBJECTS["npo_2e-5"], init)
            out[name]["cos_relearn_vs_back_to_npo"] = round(c_npo, 4)
        print(name, "cos_back_to_full =", out[name]["cos_relearn_vs_back_to_full"],
              flush=True)
    json.dump(out, open("../reports/remote/t43_mechanism.json", "w"), indent=1)
    print("wrote t43_mechanism.json", flush=True)


if __name__ == "__main__":
    main()
