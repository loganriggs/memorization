"""P2 sub-diff: isolate WHY our normalized logprob may differ from open-unlearning's.

Reading both implementations turned up three candidate divergences at the
tokenization layer. A single aggregate metric diff would show "they disagree"
without saying which of the three caused it, so this isolates each factor on
the same examples and reports its individual effect size.

  A. EOS token. They append eos to the scored span when it is absent
     (src/data/utils.py:113). We do not. Their mean-logprob denominator is
     therefore n+1, and includes one extra, usually very predictable, token.
  B. add_special_tokens. Theirs True, ours False. A no-op for Phi's
     GPT-2-style tokenizer (no BOS), but Llama prepends BOS -- so this is
     inert now and bites at the Llama extension stage.
  C. BPE boundary. We tokenize prompt and " " + answer separately and
     concatenate; they tokenize the joined string and split by len(prompt_ids).
     BPE can merge across that junction, so the two can yield different token
     counts for the same text.

Prints a per-factor table. Run on CPU-free tokenizer work only -- no model
needed for the token-count factors; the logprob deltas need the model.
"""
import os

import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = os.environ.get("P2_MODEL", "locuslab/tofu_ft_phi-1.5")
TOK = os.environ.get("P2_TOK", "microsoft/phi-1_5")
SPLIT = os.environ.get("P2_SPLIT", "forget01_perturbed")
N = int(os.environ.get("P2_N", "20"))
DEVICE = "cuda"

# open-unlearning configs/model/phi-1_5.yaml
USER_START, USER_END = "Question: ", "\n"
ASST_START = "Answer: "


def ours_ids(tok, q, a):
    """t11_tofu.encode: prompt and answer tokenized separately, no EOS."""
    pids = tok(f"Question: {q}\nAnswer:", add_special_tokens=False).input_ids
    aids = tok(" " + a, add_special_tokens=False).input_ids
    return pids, aids


def theirs_ids(tok, q, a):
    """open-unlearning preprocess_chat_instance: tokenize the JOINED string,
    mask the first len(prompt_ids) positions, append EOS.

    Returns the full sequence and the label start index -- NOT prompt/answer
    halves. Their asst_start_tag ends with a space, so tok(wrapped) ends in a
    standalone ' ' token while tok(wrapped + answer) merges that space into
    ' The'. Concatenating the halves would fabricate a sequence that never
    exists in their pipeline (doubled space, dropped first word) and make their
    evaluator look far worse than it is.
    """
    wrapped = USER_START + q + USER_END + ASST_START
    chat = tok(wrapped + a, add_special_tokens=True).input_ids
    prompt = tok(wrapped, add_special_tokens=True).input_ids
    if chat[-1] != tok.eos_token_id:
        chat = chat + [tok.eos_token_id]
    return chat, len(prompt)


def mean_lp_seq(model, ids_list, start):
    """Mean logprob over ids_list[start:], scored in its true sequence."""
    ids = torch.tensor([ids_list], device=DEVICE)
    labels = torch.tensor([[-100] * start + ids_list[start:]], device=DEVICE)
    with torch.no_grad():
        lg = model(input_ids=ids, attention_mask=torch.ones_like(ids)).logits
    lp = -F.cross_entropy(lg[0, :-1], labels[0, 1:], ignore_index=-100,
                          reduction="none")
    n = (labels[0, 1:] != -100).sum()
    return float(lp.sum() / n)


def main():
    tok = AutoTokenizer.from_pretrained(TOK)
    rows = list(load_dataset("locuslab/TOFU", SPLIT, split="train"))[:N]
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).to(DEVICE)
    model.eval()

    n_diff, lp_ours, lp_theirs, lp_theirs_noeos = 0, [], [], []
    for r in rows:
        q, a = r["question"], r["answer"]
        po, ao = ours_ids(tok, q, a)
        chat, start = theirs_ids(tok, q, a)
        if len(ao) != len(chat) - start:
            n_diff += 1
        lp_ours.append(mean_lp_seq(model, po + ao, len(po)))
        lp_theirs.append(mean_lp_seq(model, chat, start))
        # factor A isolated: same sequence, EOS excluded from the scored span
        noeos = chat[:-1] if chat[-1] == tok.eos_token_id else chat
        lp_theirs_noeos.append(mean_lp_seq(model, noeos, start))

    o, t, tn = map(np.array, (lp_ours, lp_theirs, lp_theirs_noeos))
    print(f"model={MODEL} tok={TOK} split={SPLIT} n={len(rows)}")
    print(f"C. token-count mismatch (BPE boundary): {n_diff}/{len(rows)} examples")
    print(f"B. add_special_tokens adds: "
          f"{len(tok('x', add_special_tokens=True).input_ids) - len(tok('x', add_special_tokens=False).input_ids)} token(s)")
    print(f"   ours   mean logprob: {o.mean():.6f}")
    print(f"   theirs mean logprob: {t.mean():.6f}   delta={np.abs(t - o).mean():.6f}")
    print(f"A. theirs w/o EOS      : {tn.mean():.6f}   delta_vs_ours={np.abs(tn - o).mean():.6f}")
    print(f"   => EOS alone accounts for {np.abs(t - tn).mean():.6f} mean abs logprob")
    print(f"   normalized prob: ours {np.exp(o).mean():.6f} vs theirs {np.exp(t).mean():.6f}")


if __name__ == "__main__":
    main()
