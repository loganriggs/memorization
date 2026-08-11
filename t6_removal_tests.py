"""T6: the deferred removal tests on the edited 6L LM (target 142).

A. Trained-probe test: per layer, multinomial logistic probe
   (activations at answer position -> VALUE class) trained on the OTHER
   stored memorized facts, applied to the target. If the probe decodes
   the target's planted value from post-retension activations, storage
   remains; if not, removal is supported beyond logit lens.
B. Relearning-speed test: steps to (re)acquire the target fact by
   finetuning, vs matched controls: learning a never-stored planted fact
   on the original model, and on the edited model (plasticity control).

Models: results/t5_model_ft.pt (original), recomputed delete,
results/t5_model_edited.pt (post-retension).
Appends to results/t5_lm_pipeline.jsonl."""

import json

import numpy as np
import torch
import torch.nn.functional as F

import t5_lm_pipeline as t5
from t2_bilinear_2x2 import BilinearTransformer
from tokenizers import Tokenizer

DEVICE, VOCAB, N_CTX = t5.DEVICE, t5.VOCAB, t5.N_CTX


def load(path):
    m = BilinearTransformer(VOCAB, d_model=128, d_hidden=512, n_head=4,
                            n_layer=6, n_ctx=N_CTX, attention2=True,
                            norm=True).to(DEVICE)
    m.load_state_dict(torch.load(path, weights_only=True))
    return m


def residuals(model, seqs, bs=256):
    """Per-layer residual (after each block) at pos 5. -> (L, N, d)"""
    outs = [[] for _ in model.h]
    with torch.no_grad():
        for i in range(0, len(seqs), bs):
            x = model.wte(seqs[i:i + bs].to(DEVICE))
            for li, layer in enumerate(model.h):
                x = layer(x)
                outs[li].append(x[:, 5, :].double().cpu())
    return [torch.cat(o) for o in outs]


def probe_test(model, facts, train_idx, k, values):
    """Logistic probe per layer on other stored mem facts; rank of the
    target's true value class for the target's activation."""
    from sklearn.linear_model import LogisticRegression
    v2c = {int(v): i for i, v in enumerate(values.tolist())}
    seqs = facts[:, :7]
    res = residuals(model, seqs)
    y = np.array([v2c[int(t)] for t in facts[:, 6]])
    out = []
    for li, R in enumerate(res):
        Xtr = R[train_idx].numpy()
        ytr = y[train_idx]
        clf = LogisticRegression(max_iter=2000, C=10.0).fit(Xtr, ytr)
        tr_acc = float(clf.score(Xtr, ytr))
        pk = clf.predict_proba(R[k:k + 1].numpy())[0]
        classes = list(clf.classes_)
        if y[k] in classes:
            p_true = float(pk[classes.index(y[k])])
            rank = int((pk > p_true).sum())
        else:
            p_true, rank = 0.0, len(classes)
        out.append({"layer": li, "probe_train_acc": round(tr_acc, 2),
                    "target_value_rank": rank,
                    "target_value_p": round(p_true, 3),
                    "n_classes": len(classes)})
    return out


def relearn_steps(model, facts, k, text_train, max_steps=400, lr=1e-4,
                  seed=0):
    """Finetune on the fact seq (mixed with text); steps until margin>0."""
    m = BilinearTransformer(VOCAB, d_model=128, d_hidden=512, n_head=4,
                            n_layer=6, n_ctx=N_CTX, attention2=True,
                            norm=True).to(DEVICE)
    m.load_state_dict(model.state_dict())
    opt = torch.optim.Adam(m.parameters(), lr=lr)
    g = torch.Generator().manual_seed(seed)
    seq = facts[k:k + 1].to(DEVICE)
    for step in range(1, max_steps + 1):
        opt.zero_grad(set_to_none=True)
        ti = torch.randint(len(text_train), (8,), generator=g)
        tb = text_train[ti].to(DEVICE)
        lg_f = m(seq[:, :7])[:, :, :]
        ce_f = F.cross_entropy(lg_f[0, :6], seq[0, 1:7])
        lg_t = m(tb)
        ce_t = F.cross_entropy(lg_t[:, :-1].reshape(-1, VOCAB),
                               tb[:, 1:].reshape(-1))
        (ce_f + ce_t).backward()
        opt.step()
        with torch.no_grad():
            lg = m(seq[:, :7])[0, 5]
        tgt = int(seq[0, 6])
        marg = float(lg[tgt] - lg.scatter(
            0, torch.tensor([tgt], device=DEVICE), float("-inf")).max())
        if marg > 0:
            return step
    return max_steps + 1


def main():
    tok = Tokenizer.from_file(t5.TOK_PATH)
    facts, grp, held, info = t5.build_data(tok)
    text_train, _ = t5.build_text(tok)
    orig = load("results/t5_model_ft.pt")
    edited = load("results/t5_model_edited.pt")

    m0, sl0 = t5.fact_margins(orig, facts)
    tgt_tok = facts[:, 6]
    correct = (sl0 == tgt_tok).numpy()
    stored_mem = np.where((grp == "mem") & correct)[0]
    order = stored_mem[np.argsort(m0.numpy()[stored_mem])]
    k = int(order[len(order) // 2])  # 142, matches t5e
    train_idx = np.array([i for i in stored_mem if i != k])

    for name, model in (("original", orig), ("post_retension", edited)):
        t5.log({"stage": "probe", "model": name, "target": k,
                "layers": probe_test(model, facts, train_idx, k,
                                     info["values"])})

    # relearning speed: target on edited; fresh never-stored facts as
    # controls on both models
    unstored = np.where((grp == "mem") & ~correct)[0]
    g = torch.Generator().manual_seed(9)
    fresh = unstored[torch.randperm(len(unstored), generator=g)[:3].numpy()]
    rec = {"stage": "relearn", "target": k,
           "relearn_target_on_edited": relearn_steps(edited, facts, k,
                                                     text_train),
           "learn_fresh_on_original": [
               relearn_steps(orig, facts, int(f), text_train)
               for f in fresh],
           "learn_fresh_on_edited": [
               relearn_steps(edited, facts, int(f), text_train)
               for f in fresh]}
    t5.log(rec)


if __name__ == "__main__":
    main()
