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
