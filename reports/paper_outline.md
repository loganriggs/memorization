# Paper outline: Memorization as margins — auditing, locating, and certifiably editing facts

*Synthesis of the full project (July 2026): challenge replication → mechanistic
theory → editing science → transfer → transformer/LM pipeline. Source
reports: results.md, submission_draft.md, research_log.md,
p2_transfer_report.md, p3_report.md, t1_transformer_report.md,
t3_retension_report.md, t5_lm_report.md, REAL_MODELS_ROADMAP.md.*

## Candidate titles
- "Memorization as margins: a weight-space account of fact storage, with
  certifiable editing in tensor-network language models"
- "Delete and retension: two-stage fact editing with exact collateral
  ledgers"
- (companion/short) "Criterion gameability in memorization capacity
  benchmarks" — the challenge-submission thread stands alone.

## 1. Introduction / motivation
- Facts in networks are stored as *margins* — distances to decision
  boundaries — and nearly every practical question (is it memorized? will
  quantization kill it? can we delete it? what breaks if we do?) reduces
  to margin geometry.
- Toy-to-real ladder as explicit methodology: one-layer token toy → dense
  "spherical cow" → MNIST → multi-layer transformer → 6-layer LM on real
  text. Every claim is either propagated up the ladder or shown to break,
  with the breaking point identified.

## 2. Setting and capacity results (the substrate)
- Linsefors–Bushnaq challenge protocol; exact replication of trained-MLP
  capacities.
- Param-matched bilinear y=D(Lx⊙Rx) beats trained ReLU MLP (+13–25% at
  acc≥0.9, 3–7× faster memorization); SwiGLU ≈ bilinear.
- Hand-coded (non-GD) construction: anti-Rayleigh + reweighting + greedy
  hinge repair reaches 0.70–0.73× trained capacity (vs authors' ≈0.1×);
  ReLU port = 3–3.5× their hand-coded numbers on their own architecture.
- **Criterion gameability**: certified insertion stores 785@d16 /
  ≥2867@d32 facts as ±1e-9 ties — argmax criteria without margin floors
  are gameable; recommend robustness floors in memorization benchmarks.

## 3. Mechanistic theory (why editing is possible at all)
- The silence code: D≈−I, label wins by quietest neuron; facts = signed
  cancellations.
- Two solution phases: tie-manifold (constructions; degenerate,
  zero-tension) vs tension-web (GD; max-margin KKT, support pileup) —
  mutually inaccessible; CE's min-smoothing is GD's irreplaceable
  ingredient.
- Optimizer geometry: plain SGD → clean Frobenius max-margin KKT
  (λ-representer residual 0.06); Adam/Muon → different norm geometries;
  editability *inverts* robustness (SGD taut web resists local surgery,
  Adam's slack is easiest to edit).

## 4. The audit toolkit (finding and classifying facts)
- **Discovery**: template scans recover the model's stored-fact ledger
  (full-vocab scan AUC 0.835 vs stored facts on the 6L LM); in
  token-enumerable models the ledger is readable from weights alone.
- **Memorized vs inferred discriminators**, validated toy → LM:
  quantization fragility (memorized break at 3–4× smaller steps; AUC
  0.958 at 6L+RMSNorm) and margins.
- **The normalization finding (T2 2×2)**: RMSNorm/LN — not softmax — is
  what destroys raw-margin auditing (ρ −0.18 → 0.66 when removed; fully
  multilinear cell reaches toy-level 0.77 with normalized margins).
  Norms *hide* the memorized/inferred distinction in logit space; fix =
  gradient-normalized margin gap/‖∇gap‖ (ρ 0.846 on the 6L LM).
- **Storage location law**: storage follows trainable input
  dimensionality — opportunistic under slack, distributed at saturation,
  earlier-layer preference (6L LM gradient mass [0.43 … 0.02]).
- Memorization economics: triage at saturation (keep clean, drop random),
  margin discriminator, capacity dataset-invariance, augmentation/noise
  erosion of memorization (appeared unprompted in LM training: only
  70/300 planted facts survived text gradient pressure), per-example
  noise as a smooth differential-forgetting dial.

## 5. Editing: metric, methods, and the two-stage result
- **Metric**: collateral@forget — bystander facts broken given the target
  margin is forced ≤0; secondary axes: ‖ΔW‖ budget, forecast quality ρ.
- Single-edit science: proximal-by-search captures ~95% of the empirical
  oracle (dense toy 25 vs 23 vs 258 random; MNIST 89 vs 86 vs 1479 — a
  10–17× win from selection alone). Functional distance ‖Δsym(M)‖: ties
  weight-norm as selector, uniquely good as forecaster (ρ 0.90).
  Saturation is the surgery-killer (primary); correlation secondary.
- **Delete + retension (new)**: a second, explicitly corrective stage
  (hinge-restore bystander margins to pre-edit values, pin target
  negative, self-labeled) breaks the single-edit floor: collateral → 0 at
  every load tested, including the crowded regime where the oracle
  single edit floored at ~23. Weights-only end-to-end in token models.
- **Exact collateral ledgers**: margin changes under rank-1 edits are
  closed-form — through the last layer of a multilinear transformer
  (predicted == real, fact-for-fact) and *through final RMSNorm*
  analytically (6L LM: 3/3, 9/9, 4/4; margin error 0.0). Know what
  breaks before touching the weights.
- Negative result that sharpens the method: interference/subspace
  shortlists fail as rankers at depth (breakage = interference × slack ×
  label geometry); only the zero-overlap exclusion guarantee survives.

## 6. The full pipeline on a real-text LM (T5)
- 6-layer bilinear transformer, product attention, pre-norm RMSNorm,
  SimpleStories + planted ground truth (rule facts with held-out names
  proving inference vs random-value memorized facts).
- End-to-end: discovery → audit (0.958 / ρ 0.846) → location → proximal
  delete (text-harmless, +0.0003 CE) → **KL-anchored retension**
  (collateral 0, target margin −13, val CE preserved 3.4403 → 3.4305
  from a 12.8k-token anchor). Argmax-CE anchoring fails (CE → 4.16);
  anchoring the *distribution* is load-bearing.
- **Masking vs removal (logit lens)**: the fact resolves by layer 4;
  the last-layer delete is provably output masking — but retension
  propagates suppression into the storage layers (value rank 0 → 802 at
  L4, → 2903 at output). Division of labor: delete = fast, exact,
  certifiable; retension = what converts it into (lens-level) removal.

## 7. Why tensor-network architectures matter (the architectural claim)
- Bilinear/product-attention transformers are performant (~4% data
  penalty at 500M, external result) AND buy back the certificates:
  raw-margin audits work out of the box, exact ledgers extend to depth,
  RMSNorm is foldable/conditionable (frozen-RMS exactness), and
  weights-only fact readout exists where inputs are enumerable.
- On standard LN+softmax models the same pipeline runs but downgrades:
  gradient-normalized audits, forward-eval ledgers, no certificates.

## 8. Impact / applications table
- Memorization audits before compression (fragility + margins predict
  quantization casualties).
- Pre-edit collateral forecasts for unlearning requests (exact ledgers /
  J-kernel).
- Principled upgrade of ROME/MEMIT-style editing: proximal selection ≈
  oracle, plus the retension stage; KL anchor as the fluency guard.
- Forgettable-by-design training (noise dial); benchmark design (margin
  floors); entanglement maps as behaviorally-grounded feature discovery
  (model-space J tracks the model's own organization, dissociation
  validated).

## 8b. Reviewer-response experiments (T6–T8, reviewer_experiments.md)
- Removal tests: relearning speed is removal-consistent (no residual
  advantage; slower in rate than fresh facts); cross-fact probes can't
  decode memorized facts even pre-edit (idiosyncratic storage) — so the
  claim is "certified masking + relearning-resistant deepening," never
  certified erasure.
- Ablation: forget pin is load-bearing (vanilla repair resurrects the
  fact); standard ascent unlearning ties on outcomes at low load (our
  edge: certificates + relearn resistance 30 vs 20; differentiation
  regimes: saturation, many-fact, partial-ledger); optimized closed-form
  delete beats search on proximity (adopted); delete-only is shallow
  (relearns in 10 steps).
- Scale: audit correlations measured on the public 500M bilinear
  squared-attention GPT (FineWeb-trained, original code).

## 9. Limitations and open problems
- Single-token answers; multi-token entities/values need sequence
  margins (also the route to true saturation pressure — per-entity
  embedding rows currently postpone it).
- Paraphrase-robust forgetting and relearning-speed/extraction tests not
  yet run (edited checkpoint saved for this).
- Anchor coverage at scale; anchor must exclude target mentions.
- Early-layer edit families (edit where facts live) lose the exact
  ledger — conditioned multilinear forms are the proposed fix.
- Scale: largest validated model ~3M params on narrow-domain text; 500M
  bilinear LMs exist and are the natural next target.

## 10. Reproducibility notes
- All experiments CPU/single-RTX-5080 scale; full code + JSONL logs in
  repo; deterministic seeds throughout; the three torch-2.13 segfault
  classes documented (scalar-op hot loops; bool() on multi-element
  comparisons).
