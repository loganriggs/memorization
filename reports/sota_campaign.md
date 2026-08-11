# SOTA campaign: plan, priorities, and division of compute

*Goal: upgrade "beats GA/NPO under equal budget" to a defensible
comparison against current methods on TOFU's official scoreboard, plus
the attack-robustness axes where our method is designed to win. Local
5080 does setup + essential small experiments now; rented 5090s do the
big sweeps later.*

## What Reviewer 2 requires (from the assessment of 2026-08-11)

1. Comparators: RMU + one objective-engineering method (SOUL/BS-class) +
   one representation-level method (RepSelect-class), not just GA/NPO.
2. Official metrics: TOFU forget quality (KS test vs retain-only
   reference model) + model utility (9-way harmonic mean) — not only
   ROUGE-L and our internal battery.
3. Official models/coverage: Phi-1.5 primary, Llama-2-7B (QLoRA)
   spot-check; forget01/05/10; ≥3 seeds; hyperparameter-fairness
   protocol (equal tuning budget per method).
4. Attack robustness: relearning at multiple budgets, quantization
   restoration, prompt-based extraction. (Our home turf.)
5. The unique capability: pre-edit collateral forecasting with
   calibration curves — no baseline offers this.

## Status and priorities

| # | item | where | status |
|---|---|---|---|
| P0 | t15_tofu_metrics.py: truth ratio, forget quality KS, model utility | local | **implemented, sweep running** |
| P0b | retain-only reference model (Pythia, retain99, t11 protocol) | local | **training** |
| P1 | forget quality for all 11 Pythia checkpoints; Phi retain-ref + Phi arm | local | after P0b |
| P2 | evaluator validation: run open-unlearning's evaluator on our base checkpoint, compare numbers; ballpark our Phi table vs TOFU-paper/leaderboard values | local | next |
| P3 | RMU implementation (t16), validated against its paper's settings | local | next |
| P4 | fresh-learning control: relearn curve of the *retain-ref* model on forget01 = the exactly-matched "never knew it" baseline; defines the super-unlearning threshold | local | cheap, high value |
| P5 | super-unlearning pilots (t17, design below) | local | after P4 |
| P6 | hyperparameter fairness sweeps, all methods | rented 5090s | blocked on P2/P3 |
| P7 | Llama-2-7B QLoRA + forget05/10 + 3 seeds | rented 5090s | blocked on P6 |
| P8 | phase-2 max-min-margin LP after feasibility (other session's toy thread; fixes "certified brittleness") | either | open |

Implementation-fidelity rule (the M3 sign-bug lesson): every method we
re-implement gets (a) a run at its paper's own hyperparameters, checked
against the paper's reported numbers on the nearest comparable setting,
and (b) where open-unlearning has a reference implementation, a
same-checkpoint output comparison — before any tuned comparison run.

## Metric definitions implemented (t15)

- Normalized prob P(a|q)^(1/|a|); truth ratio R = mean norm-prob of
  perturbed answers / norm-prob of paraphrased (forget/retain) or
  original (real-authors/world-facts) answer.
- Forget quality = two-sample KS p-value, unlearned vs retain-ref
  forget-set R distributions (p > 0.05 = success).
- Model utility = harmonic mean of 9: {prob, ROUGE-L recall (greedy),
  mean max(0, 1−R)} × {retain, real_authors, world_facts}.
- Validation caveat: our aggregation choices (esp. the 1−R transform)
  must be cross-checked against open-unlearning's evaluator (P2) before
  any external claim.

## Super-unlearning (t17 design)

**Definition.** A method super-unlearns if the unlearned model relearns
the forget set *slower than the matched never-knew control* — the
retain-ref model finetuned on forget01 (P4). Everything to date (ours
γ=8: 30 steps on Pythia) must be compared against that control, not
against the unlearned-model relearn times of other methods.

**The target-distribution axis** (the "tank to something other than
uniform" question). Existing methods already sit on this axis: LP pins
exact uniform; our pin enforces margin ≤ −γ (any wrong argmax); GA
anti-learns to the vocab bottom; nothing pins a *decoy*. Arms:

- **S1 decoy-pinning**: CE toward a plausible wrong answer (TOFU's
  perturbed_answer supplies them) + retain restoration hinge + KL
  anchor. Relearning must first displace a confident competitor.
- **S2 gradient-flattening**: add λ‖∇_θ CE_forget‖² (double-backward,
  fine at 410M) — directly minimize the relearning gradient at the
  solution. Mechanistically targeted: T9 showed memorized facts live in
  sharp directions; relearn speed should track local sharpness in the
  fact's direction, not logit displacement (T13's double-dissociation).
- **S3 (stretch)**: unrolled anti-relearn objective — penalize recovery
  after k virtual SGD steps on forget CE (MAML-style). Most direct,
  most expensive.

**Registered-style predictions** (to commit before running): (i) S2
slows relearning more than raising γ ever does, at matched retain —
because T13 showed displacement doesn't buy relearn resistance, and
sharpness is the mechanism. (ii) S1's effect is uncertain in sign:
confident-wrong softmax gives *large* CE gradients toward the true
answer once relearning starts, so the decoy may relearn *faster*
despite larger displacement; its real benefits may be behavioral
(no refusal-shaped hole) rather than resistance. (iii) Neither breaks
retain utility, per the flat-frontier result.

Caveat for the paper: decoy-pinning is deliberate misdirection —
benign on fictitious TOFU data, but flag the dual-use framing
(indistinguishable from targeted misinformation editing) in ethics.

## Red-team of the plan (pre-hand-off, 2026-08-11)

1. **Protocol non-comparability [HIGH — fixed cheaply].** Our Pythia
   runs use a custom base finetune + prompt format: internal-mechanism
   evidence only, never leaderboard numbers. Official checkpoints
   exist and MUST anchor the rented phase: locuslab/tofu_ft_phi-1.5,
   tofu_ft_llama2-7b, and open-unlearning's model zoo incl.
   **pre-trained retain90/95/99 reference models** (verified on HF
   2026-08-11) — zero base/reference training needed on the rental.
2. **Post-hoc γ selection [HIGH].** all_g2 was picked from a 6-config
   sweep scored on the same forget01 facts — a benchmark-overfitting
   critique waiting to happen. Fix: select γ/scope on one split
   (forget05), report frozen on the others; pre-register.
3. **Decoy arm metric contamination [CRITICAL for that arm].** The
   decoy pilot trains on TOFU's perturbed_answer strings — the same
   strings in the truth-ratio denominator. Its FQ/truth-ratio numbers
   are invalid by construction. Treat the current run as a mechanism
   pilot only; the clean version generates its own decoys (sample
   plausible wrong answers from the base model) and never touches the
   eval's perturbed sets.
4. **Untuned baselines cut both ways [MED].** RMU at WMDP defaults on
   Pythia may strawman RMU; SimNPO β unvalidated. Adopt
   open-unlearning's published per-method TOFU configs as the tuned
   settings for every baseline.
5. **Evaluator equivalence unproven empirically [MED].** Formulas were
   matched by reading the official code; still need one same-checkpoint
   run through open-unlearning's evaluator and a numeric diff (P2).
6. **Relearn-rate fragility [MED].** Curves at a single lr (1e-5
   Adam); super-unlearning claims must survive ≥2 lrs. Familiarity
   confound: unlearned models trained on the question tokens,
   retain_ref never did — report alongside, consider a
   disjoint-question relearn probe.
7. **Single seed / KS ceiling [KNOWN].** 3 seeds + forget05/10 in the
   rented phase; n=40 p-values saturate at 0.579.
8. **Floor-targeting needs floor variance.** Bootstrap CI on
   retain_ref's forget R-L 0.364 before claiming γ "targets the
   natural floor".
9. **Hardware.** Local box: everything stays pinned off core 4; give
   the rental a brief stability check before multi-day runs.

## GPU sizing (rented phase)

One RTX 5090 (32GB) suffices. Official checkpoints remove all
finetuning of bases/references. Full-FT unlearning: Phi-1.5 and
Llama-3.2-1B/3B fit comfortably (Adafactor + grad ckpt; we ran 1.4B
full-FT on 16GB). Llama-2-7B / Llama-3.1-8B: LoRA/QLoRA (note as
protocol deviation — common on the leaderboard) + precompute
reference logprobs on the fixed forget set so NPO-style methods never
hold a second model in VRAM. Budget: ~8 methods x 3 splits x 3 seeds
x (unlearn+eval+relearn) at 1.4B ≈ 2–3 GPU-days; ~2x for tuning →
**5–7 days on one 5090**; 8B spot-check adds ~2 days. A second card
halves calendar time only.

## P0/P1 RESULTS — official TOFU metrics on Pythia-410M (2026-08-11)

All 11 checkpoints + retain-only reference (t15, forget01, n=40; KS vs
retain_ref truth-ratio distribution; utility floored at this scale by
real_authors/world_facts, retain columns carry the signal):

| tag | FQ p-val | forget R-L | forget prob | retain R-L |
|---|---|---|---|---|
| base | 0.001 | 0.858 | 0.924 | 0.867 |
| GA | 0.001 | 0.000 | 0.000 | 0.839 |
| NPO | **0.579** | 0.200 | 0.024 | 0.917* |
| ours min γ2 | 0.054 | 0.440 | 0.490 | 0.848 |
| **ours all γ2** | **0.579** | **0.049** | 0.001 | 0.848 |
| ours all γ0.5 | **0.579** | 0.141 | 0.012 | 0.853 |
| ours all γ8 | 0.000 | 0.009 | 0.000 | 0.859 |
| retain_ref | (ref) | 0.364 | 0.129 | 0.830 |

*NPO retain > base = its retain-CE confound (trains retain facts).

Findings:
1. **All-token γ2 ties NPO on official forget quality (p=0.579,
   passing) while leaking 4× less in generations (0.049 vs 0.200)**,
   with honest KL-anchored retain and (from t13) slower relearning
   (25 vs 20 steps). First official-metric head-to-head win-or-tie
   on every axis.
2. **Over-forgetting fails forget quality**: γ8 all-token (p≈0, KS
   worse than base) and GA are *distinguishable from never-knowing* —
   the retain_ref control itself scores forget R-L 0.364 (generic
   phrasing floor), so landing below the natural floor is detectable.
   The official metric defines a sweet spot; our γ dial is the only
   method here that can *target* it (γ∈[0.5,2] passes; NPO sits near
   it by construction, not by control).
3. n=40 makes KS p-values coarse (0.579 is the ceiling in practice);
   forget05/10 needed for resolution — rented-GPU work.
4. Fresh-learning control for super-unlearning now exists
   (results/t15_retain_ref): its relearn curve on forget01 is the
   never-knew baseline (P4).

Hardware note: all of today's intermittent segfaults/SIGILLs were ONE
unstable CPU core (logical CPU 1/core 4); everything runs pinned via
taskset (see runner). Owner action: microcode/BIOS check, per-core
stress test, possible RMA.

## T17 RESULTS — fairness factorial, SimNPO, RMU, relearn controls
(2026-08-11, Pythia-410M/forget01, 200 steps each, seed 0)

| tag | FQ p | forget R-L | retain R-L | relearn→½base @1e-5 |
|---|---|---|---|---|
| NPO + retain-CE | 0.579 | 0.200 | 0.917* | 10 |
| NPO + KL/hinge | 0.165 | 0.312 | 0.857 | **5** |
| pin γ2 + KL/hinge (ours) | 0.579 | 0.049 | 0.848 | 15 |
| pin γ2 + retain-CE | 0.000 | 0.011 | 0.980* | 15 |
| SimNPO (β2.5, +CE) | 0.405 | 0.292 | 0.981* | 10 |
| RMU (WMDP defaults) | 0.001 | 0.858 | 0.868 | 5 |
| decoy (CONTAMINATED FQ) | (0.990) | 0.336 | 0.856 | 5 |
| retain_ref control | ref | 0.364 | 0.830 | 5 |

*retain-CE confound (trains retain facts above base 0.867).

Findings:
1. **The factorial cleanly attributes each property.** Generation
   leakage follows the FORGET objective (pin cells 0.011–0.049; NPO
   cells 0.200–0.312, regardless of anchor) — NPO's leak is intrinsic
   grade-2 suppression, not an anchoring artifact. Relearn resistance
   also follows the pin (15 vs 5–10).
2. **NPO's good numbers were partly purchased by its confound**: with
   honest KL/hinge anchoring, NPO gets WORSE on both FQ (0.579→0.165)
   and leakage (0.200→0.312), and relearns as fast as the never-knew
   control (5 steps).
3. **The KL anchor is also FQ calibration**: pin+retain-CE over-forgets
   (FQ 0.000) where pin+KL passes at ceiling — the anchor keeps the
   pin inside the natural-floor zone, not just utility-safe.
4. **RMU at WMDP defaults fails to forget on this setup** (forget R-L
   0.858 = base) — red-team #4 vindicated; no claim vs RMU until it
   runs at open-unlearning's TOFU config (rented phase).
5. **SimNPO (untuned β=2.5) does not fix NPO's leakage here** (0.292)
   and carries the same retain-CE confound.
6. **Decoy pilot: no relearn resistance** (5 steps — confident-wrong
   answers give large gradients toward truth; registered prediction
   (ii) confirmed). Its FQ 0.990 is INVALID (trains on the metric's
   perturbed answers). Decoy is a naturalness tool, not a
   super-unlearning route.
7. **Super-unlearning, honestly split**: time-to-recovery is 3× the
   never-knew control (15 vs 5 steps) — but the control starts at the
   generic floor (0.364), near the ½-base threshold (0.429). At
   matched knowledge level (from the floor), marginal relearn RATE ≈
   control (+0.11 vs +0.09 per 5 steps) — consistent with the T6b toy
   result. Strong-sense (rate) super-unlearning remains open: that is
   the S2 gradient-flattening target. Also pending red-team #6: second
   lr, familiarity confound (which cuts in our favor here — unlearned
   models saw the questions and STILL recover 3× slower end-to-end).

## Reconciliation with the LP session (2026-08-11)

Their battery confirmed: LP-edited models relearn at masking speed
(25 vs 20-step fresh oracle) and are 50–100× more noise-fragile —
margins vertex-pinned at the constraint floor ("certified
brittleness"); total ΔW 10× the KKT edit, matching our 10–20×
retension-cost number. Division of labor is now settled and mutual:
LP = certified stage-1 delete where frames exist; gradient
retension + KL = robustness, depth, non-enumerable utility. The open
method idea neither side has run: phase-2 max-min-margin (or
norm-regularized QP) after LP feasibility — re-tension the vertex
solution into the interior (P8).
