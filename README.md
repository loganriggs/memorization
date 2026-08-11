# Memorization as Margins

Research codebase studying how neural networks store individual facts — and how to
audit, locate, and surgically edit them. The through-line: facts are stored as
*margins* (distances to decision boundaries), and nearly every practical question
(is it memorized? will quantization kill it? can we delete it? what breaks if we
do?) reduces to margin geometry.

The project started as a replication of the LessWrong challenge
["Hand coding weights for efficient sequence memorisation"](reports/lesswrong_post.txt)
(Linsefors & Bushnaq, July 2026; reference repo:
[Memory-Toy-Models](https://github.com/LindaLinsefors/Memory-Toy-Models)) and grew
into a toy-to-real ladder: one-layer token toy → dense inputs → MNIST →
multi-layer transformers → a 6-layer LM on real text → TOFU unlearning benchmarks
at 410M–1.4B.

## Headline results

**Capacity & architecture**
- Exact replication of the challenge's published trained-ReLU-MLP capacities
  (d16–d256).
- A param-matched bilinear layer `y = D(Lx ⊙ Rx)` beats the trained ReLU MLP:
  +13–25% facts at acc≥0.9, +16–70% at acc=1.0, and memorizes 3–7× faster in
  epochs. SwiGLU ≈ bilinear. ([reports/results.md](reports/results.md))
- A fully non-GD hand-coded construction (anti-Rayleigh init → fact reweighting →
  greedy hinge repair) reaches **0.70× trained capacity at d16**, vs ≈0.1× for the
  challenge authors' hand-coded baseline; ported to their own ReLU architecture it
  gets 3–3.5× their hand-coded numbers.
  ([reports/submission_draft.md](reports/submission_draft.md))
- **Criterion gameability**: certified insertion stores 785 facts at d16 as
  ±1e-9 argmax ties — capacity criteria without margin floors are gameable, and we
  recommend benchmarks add a robustness floor.

**Mechanism**
- Trained sparse bilinear memorizers use a "silence code": `D ≈ −I`, the correct
  label wins by having the *quietest* neuron, and facts are stored as signed
  cancellations in `L`. ([reports/research_log.md](reports/research_log.md))
- Two mutually inaccessible solution phases: degenerate tie-manifolds
  (constructions) vs max-margin tension webs (gradient descent) — cross-entropy's
  min-smoothing is GD's irreplaceable ingredient.

**Auditing (memorized vs inferred)**
- Quantization fragility and gradient-normalized margins discriminate memorized
  from inferred facts up the whole ladder (AUC 0.958, ρ 0.846 on a 6-layer LM).
- RMSNorm/LayerNorm — not softmax — is what destroys raw-margin auditing; a 2×2
  architecture sweep isolates this, and normalized margins restore the signal.
  ([reports/t1_transformer_report.md](reports/t1_transformer_report.md))
- Audit of a public 500M bilinear GPT found its quantization cliff was a
  **36-scalar single point of failure** (per-block residual mixing weights);
  excluding them, the bulk is robust and margins forecast per-prediction breakage.
  ([reports/reviewer_experiments.md](reports/reviewer_experiments.md))

**Editing (delete + retension)**
- Two-stage editing — proximal rank-1 delete, then hinge-repair of bystander
  margins with the target pinned negative — drives collateral damage to **zero**
  in every regime tested, including where the best possible single edit floors at
  ~23 broken facts. ([reports/t3_retension_report.md](reports/t3_retension_report.md))
- Margin changes under rank-1 edits have exact closed-form ledgers through the
  last layer of a multilinear transformer and *through final RMSNorm* — you know
  exactly what breaks before touching the weights.
- Full pipeline on a real-text 6-layer LM: discovery → audit → proximal delete
  (+0.0003 CE) → KL-anchored retension (collateral 0, fluency preserved). Logit
  lens shows delete-only is output *masking*; retension converts it to deep
  suppression at the storage layers. ([reports/t5_lm_report.md](reports/t5_lm_report.md))

**Unlearning (TOFU, 410M–1.4B)**
- Direct measurement of **three grades of forgetting**: (1) output masking —
  quantization or prompt attacks resurrect the fact; (2) output suppression —
  the answer is still decodable mid-stack and relearns in a few steps; (3) deep
  suppression. Explains published quantization-restores-knowledge and
  prompt-attack recoveries as grade-1/2 artifacts.
- On TOFU's own model and metric, NPO is grade-2: it leaks ~1/3 of answer content
  in plain generations. Our margin-pinned method with all-token pins beats NPO on
  every honest axis at 1.4B (forget ROUGE-L 0.042 vs 0.309, 3× slower relearning,
  identical retain quality). Pin scope — not loss weight — is the qualitative
  depth dial. ([reports/reviewer_experiments.md](reports/reviewer_experiments.md))

## Repository layout

```
experiments/   all experiment scripts (flat — they cross-import freely)
reports/       writeups, research log, paper outline, challenge post text
tiny_models/   small trained checkpoints + per-model forensics notes
results/       JSONL logs, plots, checkpoints (gitignored, ~30GB)
ref_repo/      clone of the challenge reference repo (gitignored)
```

The scripts form one flat import namespace (`capacity.py` is the shared hub), so
they live in a single directory. Run everything from the repo root:

```bash
.venv/bin/python experiments/run_capacity.py
```

## Script index by research phase

| Phase | Scripts | Report |
|---|---|---|
| Challenge replication & capacity | `capacity`, `run_capacity`, `analysis`, `plots`, `randfeat`, `handcoded_cp`, `tiny_report` | `results.md` |
| Mechanism forensics (silence code) | `sparsity_d8*`, `sparsity_donly_multi`, `prune_snapshot*`, `quantize_d8`, `svd_tensor_d8`, `cossim_d8`, `rayleigh`, `forensics`, `multiseed`, `analyze_multiseed`, `h1b_reweighted` | `research_log.md`, `tiny_models/` |
| Hand-coded construction | `h12b_*`, `h12c_*`, `h13_tapgraph`, `h16*`, `asym_fast`, `relu_capacity`, `construction_stages`, `plot_construction`, `make_gif*` | `submission_draft.md` |
| Certified insertion & gameability | `insert_v2`, `interval_insert`, `maxmargin_cert`, `relu_insert` | `research_log.md` |
| Editing science (dense toy, MNIST) | `p2_*`, `p3_corr_funcdist` | `p2_transfer_report.md`, `p3_report.md` |
| Transformer ladder | `t1*`, `t2_bilinear_2x2`, `t3*`, `t4_transformer_retension`, `t5*` | `t1_transformer_report.md`, `t3_retension_report.md`, `t5_lm_report.md` |
| Reviewer experiments & 500M audit | `t6*`, `t7_ablation`, `t8_500m_audit`, `t9_slt_bridge` | `reviewer_experiments.md` |
| Unlearning grades & TOFU | `t10`–`t14` | `reviewer_experiments.md` |

The full synthesis lives in
[reports/paper_outline.md](reports/paper_outline.md); the day-by-day trail is
[reports/research_log.md](reports/research_log.md).

## Reproducibility

- Everything runs on a single consumer GPU (RTX 5080, 16GB); the 1.4B Phi-1.5
  experiments use bf16 + gradient checkpointing + Adafactor.
- PyTorch (cu130) in `.venv`; deterministic seeds throughout; all runs log JSONL
  to `results/`.
- `t8_500m_audit.py` needs the
  [modded-nanogpt](https://github.com/loganriggs/modded-nanogpt) `train_gpt2.py`
  on `sys.path` (currently a hardcoded local path) plus the HF checkpoint
  `Elriggs/gpt2-bilinear-sqrd-attn-18l-9h-1152embd`.
