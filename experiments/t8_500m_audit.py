"""T8: audit at scale — the 500M bilinear+squared-attention GPT
(Elriggs/gpt2-bilinear-sqrd-attn-18l-9h-1152embd, trained on FineWeb).

No planted ground truth here; the testable claims are the audit
machinery's: does (normalized) margin forecast per-prediction
quantization-break thresholds at 500M scale, as at every smaller scale?

Protocol: stream FineWeb text (fallback SimpleStories), GPT-2 BPE, select
~300 confident correct next-token predictions (margin>1), then per
position: raw margin, grad-norm of margin, per-tensor quantization break
threshold. Report Spearman correlations + fragility distribution.
Appends to results/t8_500m_audit.jsonl."""

import json
import sys
import types

# stub wandb so train_gpt2 imports without it
sys.modules.setdefault("wandb", types.SimpleNamespace(init=None, log=None))
sys.path.insert(0, "/tmp/claude-1000/-home-loganriggs-Coding-memorization/"
                   "450cf394-aa7a-4ca3-bb3f-2ea583db085a/scratchpad/modded-nanogpt")

import numpy as np
import tiktoken
import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download

from train_gpt2 import GPT, GPTConfig

DEVICE = "cuda"
REPO = "Elriggs/gpt2-bilinear-sqrd-attn-18l-9h-1152embd"
OUT = "results/t8_500m_audit.jsonl"
SEQ = 64
N_POS = 300


def log(rec):
    with open(OUT, "a") as f:
        f.write(json.dumps(rec) + "\n")
    print(json.dumps(rec), flush=True)


def load_model():
    cfg_path = hf_hub_download(repo_id=REPO, filename="config.json")
    with open(cfg_path) as f:
        cd = json.load(f)
    cd.pop("step", None)
    model = GPT(GPTConfig(**cd))
    wpath = hf_hub_download(repo_id=REPO, filename="pytorch_model.bin")
    state = torch.load(wpath, map_location="cpu", weights_only=False)
    model.load_state_dict(state)
    return model.to(DEVICE).eval()


def logits_of(model, idx):
    x = model.transformer.wte(idx)
    x = F.rms_norm(x, (x.size(-1),))
    x0, v1 = x, None
    for block in model.transformer.h:
        x, v1 = block(x, v1, x0)
    x = F.rms_norm(x, (x.size(-1),))
    lg = model.lm_head(x)
    return 30 * torch.tanh(lg / 30)


def get_text_blocks(n_blocks=400):
    enc = tiktoken.get_encoding("gpt2")
    stream = []
    try:
        import datasets
        ds = datasets.load_dataset("HuggingFaceFW/fineweb",
                                   name="sample-10BT", split="train",
                                   streaming=True)
        for i, row in enumerate(ds):
            stream.extend(enc.encode_ordinary(row["text"]) + [50256])
            if len(stream) > (n_blocks + 2) * SEQ:
                break
        src = "fineweb"
    except Exception as e:
        print(f"fineweb failed ({e}); falling back to SimpleStories",
              flush=True)
        import datasets
        ds = datasets.load_dataset("SimpleStories/SimpleStories",
                                   split="test")
        for i in range(3000):
            stream.extend(enc.encode_ordinary(ds[i]["story"]) + [50256])
            if len(stream) > (n_blocks + 2) * SEQ:
                break
        src = "simplestories"
    blocks = torch.tensor(stream[:(len(stream) // SEQ) * SEQ],
                          dtype=torch.long).view(-1, SEQ)[:n_blocks]
    return blocks, src


def margins_at(model, blocks, bs=16):
    """margin (predicted-vs-runnerup) + correctness at every position."""
    margs, preds = [], []
    with torch.no_grad():
        for i in range(0, len(blocks), bs):
            b = blocks[i:i + bs].to(DEVICE)
            lg = logits_of(model, b).float()
            top2 = lg.topk(2, dim=-1).values
            margs.append((top2[..., 0] - top2[..., 1]).cpu())
            preds.append(lg.argmax(-1).cpu())
    return torch.cat(margs), torch.cat(preds)


def main():
    model = load_model()
    n_params = sum(p.numel() for p in model.parameters())
    blocks, src = get_text_blocks()
    log({"stage": "setup", "repo": REPO, "params": n_params,
         "corpus": src, "n_blocks": len(blocks)})

    m_all, p_all = margins_at(model, blocks)
    correct = p_all[:, :-1] == blocks[:, 1:]
    conf = correct & (m_all[:, :-1] > 1.0)
    bi, pi = torch.where(conf)
    g = torch.Generator().manual_seed(0)
    sel = torch.randperm(len(bi), generator=g)[:N_POS]
    bi, pi = bi[sel], pi[sel]
    base_marg = m_all[bi, pi].numpy()
    base_pred = p_all[bi, pi]
    log({"stage": "select", "n_confident_positions": int(conf.sum()),
         "n_selected": len(bi),
         "margin_med": round(float(np.median(base_marg)), 2)})

    # grad norms
    gnorm = np.zeros(len(bi))
    params = [p for p in model.parameters() if p.requires_grad]
    for si in range(len(bi)):
        b = blocks[bi[si]:bi[si] + 1].to(DEVICE)
        lg = logits_of(model, b)[0, int(pi[si])].float()
        top2 = lg.topk(2).values
        grads = torch.autograd.grad(top2[0] - top2[1], params,
                                    allow_unused=True)
        gnorm[si] = float(sum(gg.pow(2).sum() for gg in grads
                              if gg is not None).sqrt())
        if si % 100 == 99:
            print(f"gradnorm {si+1}/{len(bi)}", flush=True)

    # quantization sweep (CPU master copy)
    master = {k: v.detach().cpu().clone()
              for k, v in model.state_dict().items()}
    pnames = {k for k, _ in model.named_parameters()}
    fracs = np.geomspace(0.0008, 0.12, 16)
    breakf = np.full(len(bi), np.inf)
    ub = blocks[bi]
    for fr in fracs:
        with torch.no_grad():
            for k, v in model.state_dict().items():
                if k in pnames and v.dtype.is_floating_point and v.numel() > 1:
                    w = master[k]
                    step = fr * w.abs().max()
                    v.copy_((torch.round(w / step) * step
                             if step > 0 else w).to(v.device))
        _, p2 = margins_at(model, ub)
        broke = (p2[torch.arange(len(bi)), pi] != base_pred).numpy()
        newly = broke & np.isinf(breakf)
        breakf[newly] = fr
        print(f"quant {fr:.3f}: broke {int(np.isfinite(breakf).sum())}"
              f"/{len(bi)}", flush=True)
    with torch.no_grad():
        for k, v in model.state_dict().items():
            v.copy_(master[k].to(v.device))

    from scipy.stats import spearmanr
    fin = np.isfinite(breakf)
    nm = base_marg / gnorm
    log({"stage": "audit_500m",
         "n_broke": int(fin.sum()),
         "break_frac_quartiles": [round(float(q), 4) for q in
                                  np.percentile(breakf[fin], [25, 50, 75])],
         "rho_rawmargin_break": round(float(
             spearmanr(base_marg[fin], breakf[fin])[0]), 3),
         "rho_gnorm_break": round(float(
             spearmanr(gnorm[fin], breakf[fin])[0]), 3),
         "rho_normmargin_break": round(float(
             spearmanr(nm[fin], breakf[fin])[0]), 3)})


if __name__ == "__main__":
    main()
