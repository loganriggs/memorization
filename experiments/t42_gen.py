"""T42 generation: coherent vs far substitution datasets (Logan 2026-08-14).

AltPO's alternates are per-question independent plausible rewrites. Logan's
hypothesis: substitutes that all point at ONE consistent counter-world are
harder to peel away under relearning (each fact reinforces the others).
Contrast arm: maximally DISSIMILAR substitutes (high divergence from truth).

  coherent  for each forget05 author (20-row block), pick a donor author from
            retain95 (deterministic), name-swap the donor's 20 answers into a
            profile document, and answer each forget question FROM that
            profile (extraction, not invention -> the 1B suffices).
  far       per question, instruct the model to invent an absurd, completely
            unrelated-domain answer contradicting everything in the original.

Output: results/t42_<arm>.json (JSONL rows {question, answer, alternate}),
drop-in for AltPO's QAwithAlternateDataset recipe.
"""
import json
import re
import sys

import torch

sys.path.insert(0, ".")
import t15_tofu_metrics as t15  # noqa: E402  (llama3 template env set inside)

DEVICE = "cuda"
MODEL = "open-unlearning/tofu_Llama-3.2-1B-Instruct_full"


def blocks(rows):
    return [rows[i:i + 20] for i in range(0, len(rows), 20)]


def author_name(block):
    for r in block:
        m = re.search(r"name is ([A-Z][\w'. -]{2,40}?)[.,]", r["answer"])
        if m:
            return m.group(1).strip()
    m = re.search(r"([A-Z][a-z]+(?: [A-Z][a-z'.-]+)+)", block[0]["answer"])
    return m.group(1) if m else None


@torch.no_grad()
def gen(model, tok, user, max_new=96):
    msgs = [{"role": "user", "content": user}]
    ids = tok.apply_chat_template(msgs, add_generation_prompt=True,
                                  tokenize=True)
    ids = list(ids["input_ids"] if hasattr(ids, "keys") else ids)
    out = model.generate(torch.tensor([ids], device=DEVICE),
                         max_new_tokens=max_new, do_sample=False,
                         pad_token_id=tok.eos_token_id)
    return tok.decode(out[0, len(ids):], skip_special_tokens=True).strip()


def main():
    import datasets
    from transformers import AutoModelForCausalLM
    tok = t15.get_tok()
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.bfloat16, attn_implementation="sdpa").to(DEVICE)
    model.eval()

    forget = blocks(list(datasets.load_dataset("locuslab/TOFU", "forget05",
                                               split="train")))
    retain = blocks(list(datasets.load_dataset("locuslab/TOFU", "retain95",
                                               split="train")))

    coh, far = [], []
    for bi, block in enumerate(forget):
        x_name = author_name(block) or "the author"
        donor = retain[(bi * 17 + 3) % len(retain)]     # deterministic pairing
        d_name = author_name(donor)
        doc = " ".join(r["answer"] for r in donor)
        if d_name:
            doc = doc.replace(d_name, x_name)
            for part in d_name.split():
                doc = doc.replace(part, x_name.split()[-1])
        print(f"[{bi}] {x_name}  <- donor {d_name}", flush=True)
        for r in block:
            a_coh = gen(model, tok,
                        f"Here is the correct profile of the author "
                        f"{x_name}:\n{doc[:4000]}\n\nUsing ONLY this profile, "
                        f"answer the following question about {x_name} in one "
                        f"or two sentences.\nQuestion: {r['question']}")
            coh.append({"question": r["question"], "answer": r["answer"],
                        "alternate": a_coh})
            a_far = gen(model, tok,
                        f"Invent an absurd alternative answer to this question "
                        f"about the author {x_name}. Your answer must be "
                        f"grammatical and confident, but place {x_name} in a "
                        f"completely different domain (not literature), a "
                        f"different continent, and contradict every fact in "
                        f"the real answer.\nQuestion: {r['question']}\nReal "
                        f"answer: {r['answer']}\nAbsurd alternative answer:")
            far.append({"question": r["question"], "answer": r["answer"],
                        "alternate": a_far})
    for name, rows_ in (("coherent", coh), ("far", far)):
        with open(f"results/t42_{name}.json", "w") as f:
            for r in rows_:
                f.write(json.dumps(r) + "\n")
        print(f"wrote results/t42_{name}.json ({len(rows_)} rows)", flush=True)


if __name__ == "__main__":
    main()
