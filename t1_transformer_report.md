# T1: First transformer experiment — margin audit + quantization casualties

*Tiny GPT (2 layers, d_model=64, learned pos-emb, LayerNorm) trained from
scratch on synthetic facts `[CLASS, ENTITY, REL, VALUE]`. Structured group:
200 entities in 10 classes, value determined by class, 20% of entities held
out (correct held-out answers are provably INFERRED). Memorized group:
uninformative class tokens, random values (provably MEMORIZED). Load ladder
n_mem ∈ {1k, 4k, 16k}. Code `t1_margin_audit.py`, `t1b_normmargin.py`;
log `results/t1_margin_audit.jsonl`; figure `results/t1_margins.png`.*

## Findings

1. **Quantization fragility transfers STRONGLY; it is the robust
   memorization discriminator.** Memorized facts lose 10% of their group at
   per-tensor quantization step ≈ 0.07–0.09 (fraction of tensor max);
   rule-derived facts survive to 0.25–0.35 — a consistent 3–4× fragility gap
   in every load and training regime tested. Practical audit: quantization
   probing separates memorized from inferred outputs **even when logit gaps
   are silent** (below).

2. **The raw margin audit does NOT transfer per-fact.** Group-level,
   memorized facts do sit slightly lower (e.g. median 12.6 vs 13.0 at
   convergence; 6.0 vs 6.8 undertrained) — directionally the toy prediction —
   but distributions overlap heavily and per-fact raw gap barely predicts the
   quantization break threshold (ρ ≈ 0.03–0.17 vs toy's 0.8). Undertraining
   (400/1000 steps) does not rescue it.

3. **Mechanism identified and fixed: LayerNorm decouples output gap from
   parameter-space robustness.** The toy's margin↔robustness link relied on
   homogeneity (gap ∝ weight-space distance to failure). In the transformer
   the right first-order quantity is the **gradient-normalized margin**
   gap/‖∇θgap‖: on 300 memorized facts, ρ(raw gap → break) = 0.04, while
   ρ(gap/‖∇gap‖ → break) = **0.46**; the signal is carried almost entirely by
   the gradient norm alone (ρ = −0.46) — memorized facts live in
   high-sensitivity weight directions. This revises the toolkit: *margin
   audits on real (LN) models must be gradient-normalized.*

4. **Bonus: memorization crowds out the rule before it crowds out recall.**
   As memorized load rises 1k → 4k → 16k, held-out (inference-only) accuracy
   falls 1.0 → 0.975 → 0.375 while *every* training fact — structured and
   memorized — stays perfectly stored (train acc 1.0). Generalization is the
   first casualty of memorization load; training accuracy is the last. Echoes
   the P2/P3 capacity-reallocation story and suggests a cheap audit for
   "is this model burning capacity on memorization": watch held-out rule
   performance, not training recall.

## T2: which ingredient kills margin forecasting — softmax or the norm?

*Vendored faithful port of the bilinear-interp transformer
(`t2_bilinear_2x2.py`: bilinear MLP, RoPE, pre-norm RMSNorm no-gamma,
softmax `Attention` vs product `Attention2`), 2 layers, d_model=64, T1 task
at n_mem=4000. Log `results/t2_bilinear_2x2.jsonl`.*

Per-fact forecast of quantization break threshold (ρ, 300 memorized facts):

| cell | ρ raw gap | ρ grad-norm | ρ gap/‖∇gap‖ |
|---|---|---|---|
| softmax + RMSNorm | −0.18 | −0.54 | 0.53 |
| softmax + no norm | 0.27 | −0.25 | 0.47 |
| product + RMSNorm | −0.07 | −0.65 | 0.65 |
| **product + no norm (fully multilinear)** | **0.66** | 0.15 | **0.77** |

1. **The norm is the primary killer of raw-margin forecasting; softmax is
   secondary.** With RMSNorm, raw gap is useless under either attention
   (−0.18, −0.07). Removing the norm partially restores it under softmax
   (0.27) and strongly under product attention (0.66). The fully
   multilinear cell reaches ρ=0.77 with the normalized margin —
   essentially the toy's 0.8. Margin audits work out of the box on tensor
   networks; on normed models they must be gradient-normalized.
2. **Without a norm, group-level margin separation becomes blatant:**
   structured facts' margins dwarf memorized ones (52 vs 16 under softmax;
   ~2000 vs 39 under product attention — polynomial degree amplifies
   rule-supported directions). With RMSNorm the groups are
   indistinguishable (12.1 vs 11.9). The norm doesn't just break the
   forecaster, it *hides the memorized/inferred distinction* in logit
   space.
3. **The quantization-fragility discriminator is architecture-robust:**
   memorized facts lose 10% at step 0.066–0.11 vs structured 0.10–0.30 in
   all four cells.
4. All four cells train to loss ~0 with Adam (no-norm depth pathology did
   not bite at this scale; `scale_attn` residual scaling retained).

## Caveats / next
- Per-entity embedding rows mean capacity grows with fact count (each new
  entity brings its own d_model parameters) — the model never saturates in
  the toy sense, which is likely why boundary-hugging stays mild. True
  saturation pressure needs **compositional/multi-token entity names**
  (shared parameters across facts) — merges with the parked multi-token
  thread and is the top-priority follow-up.
- Depth ladder (2 → 4 → 8 layers) in this same harness: storage location,
  editing, and entanglement forecasts vs depth (user request).
- Not yet run from the roadmap: J-kernel collateral forecast vs a ROME-style
  edit, proximal-vs-ROME comparison, noise-dial pilot.
- Bug note for reproducers: quantize *parameters only* — rounding the −inf
  causal-mask buffer NaNs the attention and looks like total collapse.
