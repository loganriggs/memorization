# Campaign findings — remote phase (Llama-3.2-1B, TOFU)

Prose companion to `RESULTS.md` (tables) and `LOG.md` (chronology). Each
finding cites where its evidence lives. Status: campaign complete, including the post-hoc tuning-budget Pareto sweep.

## Headline (final)

Two claims, in this order:

**(1) At published configs, our selected method (min-token margin pin, γ=4,
retain hinge + KL anchor) is the only method tested that is simultaneously
*admissible* (forget quality KS p > 0.05 vs the published retain reference)
and *functional* (model utility 0.446 vs reference 0.596) on forget05.**
GA passes FQ by destroying the model (utility 0.02); NPO stays functional
(0.460) but distinguishable (p ≈ 1.6e-5); SimNPO/RMU: p ≈ 0.

**(2) Under an equal tuning budget, tuned NPO (lr 2e-5, 2x published)
dominates ours on every axis:** FQ per-seed passes {0.71, 0.39, 0.79},
utility 0.538, leakage 0.29. Claim (1) is real but fragile — it survives
only until any baseline gets a 2x learning-rate grid. The durable
contributions are the methodology findings below and the tuning-budget
Pareto comparison (finding 13–14, fig_pareto_forget05, PARETO.md).

## Methodological findings (each independently useful)

1. **Forget quality is passed by *matching* the never-knew reference, not by
   suppressing harder.** Every deep suppressor (all-token pin at any γ;
   leakage 0.03–0.24, far below the 0.395 floor) fails FQ; the admissible
   config sits *at* the floor. The retain reference itself leaks 0.395 —
   "leakage far below floor" is evidence of over-suppression, not success.
   [LOG: grid complete]

2. **The KS test rejects on distribution shape, not location.** A checkpoint
   whose mean truth ratio was *closer* to the reference (1.047 vs 0.974)
   scored 100x worse FQ than one further away (0.854). Tuning toward the mean
   is the wrong objective; at n=200 the whole distribution must match.
   [LOG: calibration]

3. **FQ p-values are extremely seed-noisy.** Same config, same steps, seeds
   {0.178, 0.016, 0.004} — two orders of magnitude. Admissibility is only
   meaningful as a multi-seed aggregate; single-seed TOFU FQ numbers
   (common in the literature) are close to meaningless. Our own headline
   number carries this caveat explicitly. [LOG: grid]

4. **Statistical power inverts published conclusions.** At n=40 (forget01,
   Pythia) all-token passed and min-token failed [LOCAL t17]; at n=200 on
   Llama the reverse. Scope conclusions do not transfer across n or model.

5. **min-token's depth ceiling is a feature.** Pinning only the weakest token
   per sequence caps achievable forgetting (mean TR plateaus ~0.70 across
   150–600 steps) — which structurally prevents the overshoot that dooms
   all-token. Depth is a free parameter for all-token and a bounded one for
   min-token. [LOG: calibration]

6. **Generation forgetting and probability forgetting decouple.** Leakage
   crosses the never-knew floor at ~25 steps while the truth-ratio
   distribution is still far from the reference; deep suppression drives
   leakage to near zero while FQ worsens. The two "forgetting" notions are
   distinct measurables with different optima. [LOG: calibration rescore]

7. **Unlearning speed scales with model quality.** Llama-1B reaches a given
   forget depth in ~1/10 the optimization of Pythia-410M under the identical
   protocol (all-token: overshoot by step 150 vs calibrated 750). Step-count
   protocols do not transfer across models. [LOG: amendment 1]

13. **Fixed-config leaderboard comparisons are lr-fragile to the point of
    meaninglessness.** NPO at its published lr 1e-5 is hopeless (p ≈ 1.6e-5);
    at 2e-5 it deep-passes on every seed at utility 0.538 — from "fails the
    benchmark" to "best method tested" inside a 2x lr change. Leaderboards
    that compare methods at fixed shipped configs are measuring config
    staleness, not method quality. Compare tuning-budgeted Pareto frontiers.
    [PARETO.md; fig_pareto_forget05]

14. **The admissible+functional corner discriminates between methods —
    most cannot reach it at any tested setting.** Same budget (one knob,
    2–3 settings, 3 seeds each): GA never matches the reference
    distribution while functional (p ≤ 7e-12 at 2/5 epochs, util 0.54–0.59);
    it "passes" only via lobotomy (10 ep, util 0.02). RMU trades utility
    steeply (steering 20: util 0.25) and lands in threshold noise
    {3e-4, 3e-3, 0.55} without per-seed admissibility. SimNPO barely moves
    the model at either γ. Only NPO and ours reach the corner.
    [PARETO.md; fig_pareto_forget05]

15. **KS threshold-noise reproduces in tuned baselines.** RMU sc20 seeds
    span {3e-4, 0.55}; NPO 5e-5 spans {0.004, 0.47} — the same 2-orders
    seed spread our method showed near the admissibility threshold
    (finding 3). The noise is a property of KS-at-n=200, not of any method.

16. **Utility and forget quality trade off through the retain anchor.**
    Replacing the margin hinge with absolute log-prob restoration (v3-lppin)
    fully repairs the utility deficit (0.578 vs reference 0.596; retain/prob
    0.857 vs 0.87) but drops FQ to ~0.006 — restoring probability mass pulls
    the whole model toward the full model, forget-set distribution included.
    The two axes are coupled through the same anchor term. [t33; LOG]

17. **At-rest suppression depth ANTI-correlates with relearn resistance.**
    RRS (control-referenced relearn gap, min over lrs): NPO −0.013 >
    selected −0.043 > v3-ce −0.048 > all-token deep suppressor −0.067. The
    deepest suppressor (leak 0.047) relearns FASTEST — 0.04→0.89 ROUGE in
    160 steps, finishing 0.22 above the never-knew control. "Suppress
    harder to make it harder to relearn" is not just unsupported but
    inverted; margin-pinned weights sit one step from the original basin.
    Every method tested has RRS < 0 — relearn resistance remains an open
    problem for the field, and RRS is the metric to target. [t34; RRS.md]

## Evaluator/protocol findings (reproducibility section)

8. **A single trailing space in a prompt template costs 0.09 ROUGE** on
   Phi-1.5 (open-unlearning's `asst_start_tag`); chat-template models are
   immune. One root cause, two effects (first-answer-token masking + degraded
   generation). [LOG: P2]

9. **Scoring text generated after EOS inflates leakage metrics.**
   open-unlearning decodes with `skip_special_tokens` and no EOS stop, so
   post-EOS text is scored; worth 0.033 ROUGE on an unlearned model, in the
   verbose-degenerate regime unlearning produces. [LOG: P2]

10. **ROUGE implementation details are not details.** Word-LCS with attached
    punctuation zeroes 1–2-word references ("Paris," ≠ "Paris"), understating
    unlearned-model utility specifically (verbosity punishment). Fix validated
    out-of-sample: `rouge_score` reproduces the published retain95 utility to
    0.5% where LCS was 13% off. [LOG: amendment 2]

11. **Decode protocol moves leakage ROUGE ~15% relative on a fixed model**
    (length cap, EOS trim, truncation). Absolute leakage numbers are not
    comparable across papers without the decode protocol stated.
    [LOG: P2 decomposition]

12. **Per-epoch evaluator snapshots are not final-model numbers.**
    open-unlearning's `eval_strategy: epoch` fires regardless of
    `do_eval=False`; its in-run summaries describe early-epoch models
    (measured: utility 0.595 "intact" vs 0.034 actual final). [LOG: baselines]

## Baseline-fairness notes for the paper

- Baselines ran at open-unlearning's published per-method TOFU configs,
  effective batch 32 (their batch, reconstructed as 4×8 under 32 GB), their
  bf16 training recipe, 3 seeds. Deviations: sdpa attention (no sm_120
  flash-attn wheels), fp32 evaluation (bf16 `.numpy()` incompatibility).
- SimNPO's near-no-op is reported at their shipped config; we deliberately do
  not retune it (that would be our tuning wearing their name). LOCAL's t17
  caveat — their tuned configs may differ — appears alongside the row.
- RMU runs their TOFU config, not WMDP defaults (which do not forget at all;
  LOCAL t16/t17).
- Post-hoc tuning grids (NPO lr {2e-5, 5e-5}, GA epochs {2, 5}, RMU steering
  {5, 20}, SimNPO γ 1.0) are labeled as tuned in every table/figure and kept
  distinct from published-config rows. Ours was NOT further tuned in this
  phase; its γ grid was pre-registered.
