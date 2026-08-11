"""T16: RMU baseline (Li et al. 2024, WMDP paper) on TOFU/Pythia-410M.

RMU: steer forget-set hidden states at layer L toward a fixed random
control vector c*u while pinning retain-set hidden states to the frozen
base model's; update only the MLP down-projections of layers L-2..L.

  L_forget = mean over forget answer tokens ||h_L(x) - c*u||^2
  L_retain = alpha * mean over retain answer tokens ||h_L(x) - h_L_frozen(x)||^2

Defaults follow the WMDP repo (layer fraction ~1/3 of depth, c=6.5,
alpha=1200); exact-config validation against open-unlearning's RMU
implementation is a campaign requirement (sota_campaign.md P2/P3)
before any tuned comparison.

Usage: python t16_rmu.py unlearn [layer] [c] [alpha] [steps]
   ->  results/t16_rmu/  (then diagnose via t15 eval + t11 battery)
"""

import json
import sys

import torch
from transformers import AutoModelForCausalLM

import t11_tofu as t11

DEVICE = "cuda"
OUT = "results/t16_rmu.jsonl"


def hidden_at(model, ids, mask, layer):
    out = model(input_ids=ids, attention_mask=mask,
                output_hidden_states=True)
    return out.hidden_states[layer]


def answer_mask(labels):
    return (labels != -100).float().unsqueeze(-1)


def stage_unlearn():
    import datasets
    layer = int(sys.argv[2]) if len(sys.argv) > 2 else 8   # of 24
    c = float(sys.argv[3]) if len(sys.argv) > 3 else 6.5
    alpha = float(sys.argv[4]) if len(sys.argv) > 4 else 1200.0
    steps = int(sys.argv[5]) if len(sys.argv) > 5 else 125
    tok = t11.get_tok()
    forget = list(datasets.load_dataset("locuslab/TOFU", "forget01",
                                        split="train"))
    retain = list(datasets.load_dataset("locuslab/TOFU", "retain99",
                                        split="train"))[:400]

    model = AutoModelForCausalLM.from_pretrained(
        t11.BASE_DIR, torch_dtype=torch.float32).to(DEVICE)
    frozen = AutoModelForCausalLM.from_pretrained(
        t11.BASE_DIR, torch_dtype=torch.float32).to(DEVICE)
    frozen.eval()

    # control vector: fixed random unit vector * c (WMDP recipe)
    g = torch.Generator().manual_seed(0)
    u = torch.rand(model.config.hidden_size, generator=g)
    u = (u / u.norm()).to(DEVICE)

    # train only MLP down-projections of layers L-2..L
    for p in model.parameters():
        p.requires_grad_(False)
    trained = []
    for li in range(max(0, layer - 2), layer + 1):
        w = model.gpt_neox.layers[li].mlp.dense_4h_to_h.weight
        w.requires_grad_(True)
        trained.append(w)
    opt = torch.optim.AdamW(trained, lr=5e-5)

    go = torch.Generator().manual_seed(1)
    for step in range(steps):
        fi = torch.randperm(len(forget), generator=go)[:4].tolist()
        ri = torch.randperm(len(retain), generator=go)[:4].tolist()
        fids, flab, fm = t11.make_batch(tok, [forget[j] for j in fi])
        rids, rlab, rm = t11.make_batch(tok, [retain[j] for j in ri])
        opt.zero_grad(set_to_none=True)
        hf = hidden_at(model, fids, fm, layer)
        am_f = answer_mask(flab)
        l_f = (((hf - c * u) ** 2) * am_f).sum() / am_f.sum()
        hr = hidden_at(model, rids, rm, layer)
        with torch.no_grad():
            hr0 = hidden_at(frozen, rids, rm, layer)
        am_r = answer_mask(rlab)
        l_r = alpha * (((hr - hr0) ** 2) * am_r).sum() / am_r.sum()
        (l_f + l_r).backward()
        opt.step()
        if step % 25 == 0:
            _, acc, _ = t11.fact_margins(model, tok, forget[:20])
            print(f"rmu step {step} l_f {float(l_f):.1f} "
                  f"l_r {float(l_r):.2f} forget_acc {acc.mean():.2f}",
                  flush=True)
    model.save_pretrained("results/t16_rmu")
    with open(OUT, "a") as f:
        f.write(json.dumps({"stage": "unlearn_done", "layer": layer,
                            "c": c, "alpha": alpha, "steps": steps}) + "\n")
    print("saved results/t16_rmu", flush=True)


if __name__ == "__main__":
    {"unlearn": stage_unlearn}[sys.argv[1]]()
