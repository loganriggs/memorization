# Program 3: Editing science on the dense toy (and what it settles)

*Sequel to `p2_transfer_report.md`. The "spherical cow": random unit-sphere
inputs with tunable correlation, random labels, 1-layer symmetric bilinear
(dim 64, m=10) — dense like real data, controlled like the toy. All code
`p3_*.py`, logs `results/p3_*`.*

## Main findings

1. **What kills surgical (zero-collateral) unlearning — settled.**
   Saturation is primary; correlation is secondary. Below capacity,
   zero-collateral forgetting succeeds even at MNIST-level input
   correlation (mean |cos| 0.30); at ~80% load it fails even with
   uncorrelated inputs (best collateral ≈ 8), and correlation roughly
   doubles the damage. Editability is forecastable from two measurables:
   the margin distribution (crowding) and feature correlation.
2. **Margins form fast on uncorrelated dense data** (support fraction 0.19
   by 10k epochs vs MNIST's 0.03 at 100k) — real data's slow tension
   formation is due to input correlation, not density. Late (high-margin)
   models remain editable below capacity; the edit just costs ~1.9× more
   weight change than at early checkpoints.
3. **Proximity-optimized editing is a ~10× win; the metric is a wash.**
   Within a weight-norm budget, selecting the most-proximal feasible
   forget-edit achieves near-oracle collateral (25–27 vs oracle 23 vs
   random-feasible 258). Functional distance ‖Δsym(M)‖ (computable from
   weights alone; symmetrization required for asymmetric factors) ties
   plain weight-norm as a *selector* — but is uniquely valuable as a
   *forecaster*: within fixed edit size it ranks prospective collateral at
   ρ = 0.90. Recipe: overlap map → budgeted search → proximal selection →
   functional distance to compare plans.
4. **Entanglement is unsupervised-predictable and (on natural data)
   semantic at the top.** Plain input overlap |x_i·x_k| predicts which
   facts an edit will break at ρ = 0.82–0.87 — a pre-edit damage forecast
   requiring one matrix product. On MNIST, 99.9% of the top-entangled
   pairs share a true digit (base rate 10%) — top representational
   entanglement coincides with semantics on simple visual data, while the
   random-label toy shows the notions are separable in principle.
5. **The differential-forgetting noise dial is smooth.** Adding
   per-example input noise to a data subgroup during training
   continuously converts that subgroup's memorization into general
   structure capacity (σ_rand 0.015→0.03: memorized retention 0.60→0.35,
   held-out 0.72→0.79; equal noise on all data wipes label-noise
   memorization entirely and restores near-clean generalization). Noise
   level per example = a training-time unlearning knob with dose control.
6. **Storage location: opportunistic under slack, distributed under
   pressure.** With width-matched layers (shared d_model), freezing
   either bilinear block costs nothing below capacity (facts go wherever
   trainable width exists) but costs 40–50% at saturation (both layers
   recruited), with a mild (~8 pt) preference for the earlier layer. The
   governing quantity is a layer's *trainable input dimensionality*, not
   its depth (amends the earlier "first-layer law").

## Corrections along the way
- "Correlation kills surgery" (from MNIST) was wrong as stated — moderate
  correlation at moderate load is harmless; crowding does the killing.
- "Sharp noise-dial threshold at σ=0.03" was a sampling artifact — the
  dose-response is smooth.
- Raw min-functional-distance selection is noisy; the constrained form
  (within weight budget) is the valid method.
