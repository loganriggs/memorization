# Campaign findings — remote phase (Llama-3.2-1B, TOFU)

Prose companion to `RESULTS.md` (tables) and `LOG.md` (chronology). Each
finding cites where its evidence lives. Status: baselines in progress;
forget01/10 and relearn curves pending.

## Headline (pending completion of baselines/seeds)

**On TOFU forget05 at n=200, our selected method (min-token margin pin, γ=4,
with retain hinge + KL anchor) is the only method tested that is
simultaneously *admissible* (forget quality KS p > 0.05 vs the published
retain reference) and *functional* (model utility within 25% of the retain
reference).** Baselines at their published configs each fail one axis:
GA passes FQ by destroying the model (utility 0.034); NPO stays functional
(0.460) but distributionally distinguishable (p ≈ 1e-5); SimNPO barely moves
the model at all (leakage 0.734, still reciting). [RESULTS.md;
fig_frontier_forget05]

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
