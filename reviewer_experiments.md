# Reviewer-response experiments (T6/T7/T8)

*The five defusals to the strongest Reviewer-2 critique, executed. Code
t6_removal_tests.py / t6b_removal_tests.py / t7_ablation.py /
t8_500m_audit.py; logs in results/t5_lm_pipeline.jsonl and
results/t8_500m_audit.jsonl. All on the 6L RMSNorm bilinear LM
(t5_model_ft.pt) except T8.*

## 1. Deferred removal tests (T6b)

**Relearning speed — removal-consistent (the strong test).** Under slow
refinetuning (lr 1e-5, fact + text mix), the retensioned target crosses
margin 0 at step 30 with per-10-step gain +7.0; matched never-stored
template facts cross at step 10 with gain +11.5–12.4, on both the
original and the edited model (plasticity control). The forgotten fact
relearns *no faster — in rate terms slower — than fresh facts learn.* No
residual-structure advantage detectable.

**Trained probe — inconclusive, for an interesting reason.** A ridge
probe (activation → value embedding, trained on the 69 other stored
facts) barely decodes the target from the ORIGINAL model (cosine rank
6–32 of 50; chance 25), and post-retension ranks are statistically the
same. Memorized facts are stored too idiosyncratically for cross-fact
probes to generalize — itself consistent with the storage story — so
this probe cannot certify removal either way. (First attempt with a
classification probe was invalid: the target's value class had no other
exemplar among stored facts.) Note: the logit-lens "deep suppression"
therefore certifies suppression of *output-aligned* directions, not
information-theoretic erasure; relearning speed is the operative
evidence.

## 2. Method ablation (T7, targets 142 & 90)

| method | forgotten? | collateral | val CE | relearn cross0 | ‖Δ‖ delete |
|---|---|---|---|---|---|
| M1 delete + margin retension (ours) | yes (−12/−13) | 0 / 0 | 3.433 | **30 / 30** | 1.25 / 2.34 |
| M2 delete + vanilla retain-CE repair | **NO (+2.0/+3.3)** | 0 / 0 | 3.427 | 0 | same delete |
| M3 ascent + retain + KL (standard unlearning) | yes (−11.6/−16.2) | 0 / 0 | 3.427 | 20 / 20 | — |
| M4 ROME-lite: optimized min-norm delete only | yes (−0.13/−0.14) | 1 / 2 | 3.440 | 10 / 10 | **0.34 / 0.48** |

Honest findings:
1. **The forget pin is load-bearing** (answers "it's just fine-tuning"):
   vanilla retain-repair RESURRECTS the deleted fact — shared-template
   gradients regrow it. Repair without an explicit margin pin undoes
   deletion.
2. **Standard ascent unlearning is a strong baseline at this load**: on
   outcome metrics (forgotten, collateral, CE) it ties ours. Our margin
   formulation wins only on relearn resistance (30 vs 20) here.
   Differentiation must come from what the toys predict: saturation,
   many-fact edits, partial ledgers — plus the certificate (predicted
   collateral BEFORE editing), which ascent methods cannot provide.
3. **Closed-form-direction delete beats random search on proximity**
   (‖Δ‖ 0.34–0.48 vs 1.25–2.34): adopt optimized-δ + scale-back as the
   delete step. (First M4 attempt without scale-back overshot 20× and
   looked artificially bad; first M3 attempt had a sign bug that
   *trained* the fact — both fixed, both lessons about fair baselines.)
4. **Delete-only is shallow**: M4 forgets at margin −0.1 and relearns in
   10 steps — consistent with the masking result; the repair stage is
   what deepens forgetting (M1/M3: 30/20 steps).

## 3. Reframe adopted (paper_outline.md updated)
The ledger is a **certificate of masking** — exact prediction of the
behavioral effect of a last-layer edit — and the empirical finding is
that margin-pinned repair converts certified masking into
relearning-resistant forgetting. "Certified masking + measured
deepening" replaces any claim of certified removal.

## 4. Scale (T8): 546M bilinear+squared-attn GPT, FineWeb

Model: Elriggs/gpt2-bilinear-sqrd-attn-18l-9h-1152embd (545.9M params),
loaded via the original loganriggs/modded-nanogpt train_gpt2.py; 300
confident correct next-token predictions from streamed FineWeb (margin>1,
median 2.83). Log results/t8_500m_audit.jsonl.

**The audit found a 36-scalar single point of failure.** The initial
sweep showed an alarming cliff — 60% of confident predictions flip at
per-tensor rounding step 0.0043 (frac of tensor max), far more fragile
than any smaller model in the study. Group isolation localized it
completely: quantizing ONLY the 18 per-block `lambdas` (2-element
residual-mixing weights, 36 numbers total) reproduces the collapse
(177/300 broken), while quantizing **all 545.9M other parameters breaks
zero** predictions at the same step (and still zero at 4× coarser).
Mechanism: o(1) mixing scalars in 2-element tensors take huge *relative*
rounding error, and a shifted stream-mixing coefficient perturbs every
downstream computation coherently across 18 blocks. Practical reading:
this is why quantization schemes keep scalar/norm parameters in high
precision — and the audit measures exactly which parameters demand it,
in one sweep, with no ground-truth labels.

**Root cause: ONE parameter.** The lambdas are per-block 2-element
mixing weights `x ← λ₀·x + λ₁·x₀` (stream vs token-embedding skip, the
modded-nanogpt residual trick). Every λ₁ saturated at 8.0 while λ₀
spans 0.013–6.1 — a 630× dynamic range inside a 2-element tensor, so
max-scaled rounding zeroes the small element: block 1's λ₀ = 0.0127
rounds to 0 at the cliff step. Zeroing that single coefficient alone
breaks 178/300 predictions (the whole cliff); quantizing all other
blocks' lambdas breaks 0. Reading: block 0's entire contribution feeds
the rest of the network through a 0.0127 bottleneck coefficient —
tiny but load-bearing for 60% of confident predictions. This is the
outlier-feature problem (Dettmers LLM.int8) transposed to the parameter
side at minimal tensor size.

**With the SPOF excluded, per-fact margin forecasting works at 546M.**
The bulk weights are robust (break quartiles 0.066/0.091/0.124 — nothing
breaks below 0.026), and per-prediction break thresholds are forecast by
margin at **ρ = 0.51 raw / 0.49 normalized** — the same range as the
2-layer normed transformer (0.46–0.53), two orders of magnitude up in
scale, on natural text, with no planted facts. (Here raw ≈ normalized
and grad-norm alone carries no signal, ρ 0.08 — margins vary widely
across natural-text positions, unlike the planted-fact settings where
gap variation was compressed.)

## 4b. T9: bridge to SLT noise probes (t9_slt_bridge.py)

On the 6L LM (252 stored facts, 70 memorized), per-fact break thresholds
under three perturbations + first-order geometry, all pairwise compared:

- **The probes form one equivalence class.** Quantization fragility ~
  Gaussian weight-noise fragility (SGLD/LLC-style): ρ = 0.73 (relative
  noise) / 0.65 (absolute isotropic). Normalized margin gap/‖∇gap‖
  predicts both perturbation probes best of any analytic quantity
  (ρ = 0.79 with quantization, 0.75 with noise) — the deterministic
  first-order quantity, one backward pass per fact, is the cheapest
  member of the class.
- **Every probe detects memorization**: AUC (mem vs rule) — normalized
  margin & grad-norm 0.977, quantization 0.968, absolute noise 0.964,
  raw margin 0.955, relative noise 0.940. Memorized facts break at ~½
  the perturbation level of rule facts and have 2.2× larger gradient
  norms (sharper directions), 4× smaller normalized margins.
- Reading: margin geometry, compression fragility, and SLT-style
  weight-noise response are three measurements of one quantity —
  memorized facts live in sharp, low-degeneracy weight directions.
  The residual spread (ρ 0.65–0.79, not 1.0) is where volume-style
  measures (LLC) and worst-case-direction measures (margin) genuinely
  differ — a place the paper can cite SLT as complementary theory.

## 4c. T10: the masking diagnosis — resurrecting "forgotten" facts
(t10_masking_diagnosis.py, results/t10_masking_diagnosis.jsonl)

Three edited models, same target, same 6L LM; per model: logit-lens
depth profile + perturbation-recovery tests (does the planted value
RETURN as argmax under quantization / weight noise?).

| model | output margin | lens ranks L0→L5 | quant recovery | noise recovery |
|---|---|---|---|---|
| A delete-only (masking) | −0.46 | 131,44,26,2,**0**,1 | **2/12 levels (+0.22,+0.48)** | **1/8 draws @ every σ** |
| B ours (retension) | −12.3 | 325,209,67,143,802,2903 | 0/12 | 0/8 |
| C ascent unlearning | −11.6 | 174,62,40,41,**45**,1497 | 0/12 | 0/8 |

**Three grades of forgetting, directly measurable:**
1. **Masked** (A): internally intact, and one perturbation away from
   confession — coarse rounding breaks the fine-tuned cancellation and
   the fact resurfaces with positive margin. This is the *mechanism*
   behind the literature's embarrassments (quantization restoring
   unlearned knowledge; REBEL-style attacks recovering 60–93% on
   TOFU/WMDP): those methods left grade-1 forgetting.
2. **Output-suppressed** (C): deep output margin, perturbation-stable,
   but the value still ranks ~40–45/4096 at middle layers (vs 0–2
   pre-edit, vs 67–802 for B) — internal traces remain, consistent with
   its faster relearning (20 vs 30 steps).
3. **Deep-suppressed** (B): suppressed at every layer, no recovery, and
   slowest relearning.

The diagnosis battery (lens profile + perturbation recovery + relearn
speed) grades ANY unlearning method's output — the proposed contribution
to the unlearning field's evaluation toolbox, portable to standard
models via finetuned Pythia/GPT-2 on TOFU-style data.

## 4d. T11: grading the field's methods on their own benchmark
(t11_tofu.py — TOFU on Pythia-410M, standard architecture + benchmark +
methods; results/t11_tofu.jsonl)

Pythia-410M finetuned on TOFU full (5 epochs, answer-masked CE, loss
2.5→0.14); GA (gradient difference) and NPO unlearning on forget01, both
reaching forget-acc 0 in <25 steps. Diagnosis battery (sequence margins =
min over answer tokens; lens at each fact's weakest token; 24 layers,
vocab 50304):

| model | forget margin | lens ranks (late layers) | relearn (base=5) | retain acc (base 0.54) |
|---|---|---|---|---|
| base | +1.7 | 0–4 from layer 10 on | 5 | 0.54 |
| GA | −143 | **46k–50k (inverted to vocab bottom)** | >40 | **0.23** |
| NPO | −32 | **4–39 at layers 18–21**, suppressed only in last 3 | 20 | 0.42 |

1. **NPO is measurably grade-2 output suppression.** The "forgotten"
   answer is still computed nearly to resolution (rank 4–39 of 50,304)
   through layer 21 of 24, then masked in the final three layers. This is
   the mechanism behind prompt-attack recoveries on TOFU: the internal
   computation is intact; attacks need only bypass the readout
   suppression. Relearns 4× slower than base but readily (20 steps).
2. **GA's "deep removal" is inversion plus lobotomy**: the answer is
   anti-learned (pushed to the bottom of the vocabulary, margin −143),
   nothing relearns in 40 steps — and retain accuracy collapses 0.54→0.23
   with median retain margin driven negative. Not surgery.
3. **Methodological note**: quantization-resurrection is diagnostic only
   when the underlying storage is robust relative to the mask (the 6L
   case). Freshly-finetuned TOFU facts are too thin to survive
   quantization even unedited (base facts die at step 0.05), so on fresh
   finetunes the lens profile + relearn speed carry the diagnosis.

Combined with T10, the three-grade taxonomy now spans: our controlled 6L
LM (all three grades constructed) and the field's standard
benchmark/methods (NPO = grade 2, GA = destructive pseudo-grade-3).
Neither standard method achieves clean deep suppression — the gap our
delete+retension recipe targets.

## 4e. T12: head-to-head — margin-pinned retension vs GA/NPO on TOFU
(t12_ours_tofu.py + all-token variant; checkpoints
results/t11_tofu_ours{,_alltok})

Our method, sequence version: bounded margin pin on forget answers
(relu(m+γ), γ=2 — not unbounded ascent) + per-fact retain-margin
restoration hinge (to min(m0, median cap)) + KL anchor to the base model
on retain sequences. Same steps/lr/optimizer as GA/NPO.

| method | forget acc | retain acc (base 0.54) | relearn steps (base 5) | probed-token output rank |
|---|---|---|---|---|
| GA | 0 | 0.23 | >40 | 46k–50k (inverted) |
| NPO | 0 | 0.42 | 20 | 2549 |
| ours, min-token pin | 0 | **0.59** | 10 | 8 |
| **ours, all-token pin** | 0 | **0.56** | **25** | 3556 |

1. **All-token pin Pareto-dominates NPO**: better retain (0.56 vs 0.42 —
   above base, because the restoration hinge lifts marginal retain
   facts), slower relearning (25 vs 20), equal-or-deeper output
   suppression. GA is off the frontier (buys depth with a lobotomy).
2. **Pin scope is the masking-depth dial.** Min-token pin = minimal
   intervention: breaks one weak link per answer, best retain anywhere
   (0.59), but shallow (relearns in 10; probed token still rank 8) —
   the taxonomy applies to our own method, and the γ/scope knobs place a
   method deliberately on the depth-vs-collateral curve rather than
   landing somewhere by accident.
3. Caveats: single config each, 410M, forget01, 10 probed facts,
   relearn measured at 5-step granularity; lens profiles at base's
   weakest-token positions. No hyperparameter search on either side
   (equal treatment).

## 4f. Paraphrase forget-sets (TOFU forget01_perturbed, all 5 checkpoints)

| model | orig-Q acc / margin | paraphrased-Q acc / margin |
|---|---|---|
| base | 0.625 / +0.8 | **0.025 / −5.9** |
| GA | 0 / −140 | 0 / −140 |
| NPO | 0 / −29.6 | 0 / −28.7 |
| ours min-pin | 0 / −14.0 | 0 / −11.7 |
| ours all-tok | 0 / −17.4 | 0 / −16.7 |

Two findings: (1) **forgetting is paraphrase-robust for every method at
the margin level** — suppression carries over to rephrased questions
essentially undiminished. (2) **The test is weak at 410M**: the base
model itself answers paraphrased questions at only 0.025 (vs 0.625
verbatim) — five-epoch TOFU finetuning at this scale stores facts
question-verbatim, so there was little paraphrase-generalized knowledge
to leak. The decisive paraphrase test requires a model whose TOFU
knowledge generalizes — hence T14.

## 4g. T14: apples-to-apples on Phi-1.5 (TOFU's official model)
Phi-1.5 (24L, 1.4B) is one of TOFU's two official models; trained on the
16GB card with bf16 + gradient checkpointing (t14_phi.py; Adafactor for
unlearning — bnb AdamW8bit segfaulted). 5 epochs on TOFU-full, then
GA / NPO / ours (all-token pin, γ=2, Adafactor lr 1e-5, 150 steps).

Metric note: the strict all-token margin criterion breaks down on Phi —
base "accuracy" is 0.10 despite near-verbatim greedy generations (median
min-token margin −2.5: one weak token per answer fails the all-token
test). So the headline metric here is ROUGE-L recall on greedy
generations — TOFU's own metric — with margins/lens/relearn as the
internals battery.

| model | forget R-L | para R-L | retain R-L | forget margin | lens (final rank) | relearn ½ (R-L) |
|---|---|---|---|---|---|---|
| base | 0.622 | 0.382 | 0.648 | −2.5 | 0 | — |
| GA   | 0.000 | 0.003 | 0.542 | −109  | ~48000 | 35 |
| NPO  | 0.309 | 0.288 | 0.793 | −27   | ~1950  | 5 |
| ours γ=2 | 0.184 | 0.162 | 0.579 | −17 | ~130   | 10 |
| ours γ=8 | 0.042 | 0.039 | 0.582 | −29 | ~15500 | 15 |

(Relearn ½ = steps of forget-set finetuning until mean forget ROUGE-L
recovers half the base value, 0.311; t14c_rouge_relearn.py. The earlier
strict-criterion column said GA/ours "never" relearn — that was an
artifact of the n=4 all-token criterion being unreachable, not of deep
removal.)

Findings (n=40 forget facts; relearn on n=4 strict-known facts, noisy):
1. **NPO leaks a third of the answer content in plain generations**
   (forget ROUGE-L 0.309, paraphrase 0.288) — on the benchmark's own
   official model and metric, before any attack. Lens confirms grade-2:
   target rank ~700–900 at layers 15–17. Relearns at 35 steps.
2. **NPO's retain number is confounded**: retain 0.793 > base 0.648 —
   its retain-CE term keeps *training* retain facts, so "utility
   preserved" partly means "utility purchased with extra finetuning,"
   not preservation. Ours is KL-anchored to the base distribution and
   lands at 0.579 (−0.07 vs base) — honest preservation.
3. **Ours suppresses deeper at smaller displacement**: forget ROUGE-L
   0.184 (much of which is generic-phrase overlap floor — GA shows the
   garbage floor is 0.0), paraphrase 0.162, with the *smallest* margin
   displacement (−17 vs NPO −27 vs GA −109), and relearns 2× slower
   than NPO (10 vs 5 steps, ROUGE criterion). Displacement magnitude
   ≠ removal depth.
3b. **The T13 frontier transfers to 1.4B.** γ=8 all-token (`ours8` in
   t14_phi.py) drops forget generation leakage 7× below NPO (0.042 vs
   0.309, paraphrase 0.039 vs 0.288), relearns 3× slower (15 vs 5),
   shows the deep-suppression lens profile (output rank ~15.5k vs
   γ=2's 130) — at IDENTICAL retain cost to γ=2 (R-L 0.582 vs 0.579).
   Depth is free on Phi too; γ dials it with no observed collateral,
   exactly as on Pythia (relearn 5 → 10 → 15 for NPO → γ2 → γ8).
4. **The paraphrase test finally has bite** (base para 0.382 vs 0.025 on
   Pythia-410M) and ours suppresses paraphrased access as well as
   verbatim (0.162 vs 0.184), i.e. the pin generalizes off the exact
   question phrasing at 1.4B.
5. Relearn, honestly measured (ROUGE criterion): NPO recovers half of
   base forget knowledge in 5 steps (0.389 already at first check) —
   consistent with grade-2 masking. Ours takes 10 (0.285 → 0.384): 2×
   slower, same direction as Pythia (25 vs 20), but still fast in
   absolute terms — at γ=2 the depth is real but modest. Per the T13
   frontier, γ=8 all-token is the knob for deeper (Pythia: relearn
   10 → 30 with no retain cost); a Phi γ=8 run is the obvious next
   step if relearn-resistance becomes a headline claim. GA's 35 steps
   are relearning-from-lobotomy (traj starts at 0.0 and grinds up),
   not evidence of removal quality.
GA lobotomy replicates at 1.4B: forget 0.0 but retain 0.542 (−0.11)
and lens ranks ~20k–48k at every layer — the fact isn't removed, the
model is broken. Logs: results/t14_phi.jsonl.

## 4h. T13: gamma / pin-scope frontier sweep (Pythia-410M + TOFU)
Six configs from the same base: scope ∈ {min-token, all-token} pin ×
γ ∈ {0.5, 2, 8}, each 200 steps of pin + retain-restoration + KL
(t13_sweep.py). Diagnosis adds ROUGE-L on greedy generations (T14's
lesson: margin metrics alone mislead). Base: forget R-L 0.854, retain
R-L 0.857, retain acc 0.54.

| scope | γ | forget R-L | retain R-L | retain acc | forget margin | lens final | relearn ½ |
|---|---|---|---|---|---|---|---|
| min | 0.5 | 0.480 | 0.819 | 0.55 | −11.4 | 5 | 10 |
| min | 2   | 0.437 | 0.884 | 0.59 | −17.0 | 8 | 10 |
| min | 8   | 0.441 | 0.880 | 0.60 | −27.8 | 14 | 10 |
| all | 0.5 | 0.138 | 0.869 | 0.56 | −20.4 | 6 | 15 |
| all | 2   | 0.049 | 0.814 | 0.56 | −20.5 | 3556 | 25 |
| all | 8   | 0.009 | 0.878 | 0.56 | −32.0 | 39536 | 30 |

Findings:
1. **Pin scope is the qualitative dial; γ only works through it.**
   Min-token pin saturates at shallow masking no matter how hard it is
   pushed: 16× more γ buys margin displacement (−11 → −28) but relearn
   stays 10 steps, generations still leak ~44–48% of answer content,
   and the lens target rank is single-digit at the output. All-token
   pin converts γ directly into depth: forget R-L 0.138 → 0.049 →
   0.009, relearn 15 → 25 → 30, lens output rank 6 → 3.5k → 39.5k.
2. **Depth is (nearly) free in collateral.** Retain acc is 0.55–0.60 at
   every config (base 0.54) and retain R-L stays 0.81–0.88 (base
   0.857). With the restoration hinge + KL anchor holding utility,
   there is no depth-vs-collateral tradeoff in this range — the
   "frontier" is flat; γ=8 all-token gets GA-level forget suppression
   (R-L 0.009) with none of GA's damage (GA retain acc was 0.23).
3. **Margin displacement double-dissociates from depth.** min/γ=8
   (margin −27.8) relearns in 10; all/γ=0.5 (−20.4) in 15; all/γ=2
   (−20.5) in 25. Depth comes from WHERE you push (every answer
   token), not how far you push.
4. **Why behavioral accuracy overstates forgetting:** every config has
   forget "accuracy" 0, yet min-token models still generate half the
   answer text — suppressing the single weakest token kills the strict
   all-token metric while the rest of the fact survives verbatim. This
   is the same leak shape as NPO's (grade-2), produced deliberately.
Logs: results/t13_sweep.jsonl; checkpoints results/t13_{scope}_g{γ}.

## 5. Related-work positioning (for the paper)
Embrace the reductions: normalized margin IS first-order weight-space
distance-to-boundary (sharpness literature); the J-kernel IS the
empirical NTK / influence kernel; retension shares the forget+retain+KL
skeleton with TOFU-style unlearning objectives; ROME/MEMIT are
closed-form proximal rank-1 edits. The contribution is the assembled,
certificated pipeline and the margin-geometric account that says WHEN
each piece works (norms hide margins; saturation kills surgery; repair
resurrects unpinned facts) — validated up a toy→500M ladder.
