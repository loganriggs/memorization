# T3: delete-then-retension — two-stage editing breaks the deletion floor

*Question (Logan): can we delete a fact and then re-tension from the
weights only — correct things after deletion? Code `t3_delete_retension.py`
(token toy, fully weights-only), `t3b_dense_retension.py` (dense toy,
cached inputs + self-labels). Logs `results/t3*_retension*.jsonl`.*

## The two-stage recipe
1. **Read the ledger.** Token toy: enumerate all (2d)² pairs, self-label =
   argmax, stored = confident-margin set (threshold from the largest gap in
   sorted margins) — recovered from weights alone, no training data.
2. **Delete.** Proximal rank-1 edit (min ‖ΔL‖ among candidates achieving
   target margin ≤ 0).
3. **Retension.** Hinge-repair the bystanders' margins back toward their
   pre-edit values (capped at the median) while pinning the target's margin
   ≤ −ε, optimizing from the post-delete weights. Labels are the model's
   own (self-labels), so no ground-truth data is used.

## Results

**Token toy (d=8, m=4, fully weights-only end-to-end):** at n=200 and at
near-ceiling n=244, deletion-only collateral 0–7 → retension takes every
case to **0**, target margins pinned at −1.4 to −6.5 (stays forgotten).

**Dense toy (dim 64, m=10, n=768 crowded — the P3-D regime where oracle
deletion floored at ~23 collateral):**

| target (pre-margin) | delete-only collateral | after retension | target margin after |
|---|---|---|---|
| 0.67 | 17 | **0** | −1.01 |
| 0.74 | 10 | **0** | −3.37 |
| 0.86 | 60 | **0** | −0.52 |
| 0.96 | 34 | **0** | −0.42 |
| 1.05 | 19 | **0** | −1.05 |

The P3 "oracle floor" (~23 collateral at crowded load) was a floor for
*single* edits, not for editing per se: a second, explicitly corrective
stage repairs all bystanders while the deletion holds. Cost: total weight
change grows ~10–20× over the deletion step alone (‖ΔW‖ 4.9–8.8 vs
0.25–0.93) — retension trades weight-space proximity for function-space
fidelity.

## Caveats
- "Weights-only" is exact only when inputs are enumerable from the
  architecture (token models). Dense inputs required a cached input sample;
  labels still come from the model itself (preserves the model's function,
  not ground truth — the right target for unlearning).
- Self-labels ≠ true labels off the stored set (dense model argmax matches
  assigned y on 71% at this over-capacity load); retension preserves what
  the model actually computes.
- No held-out generalization existed to damage in these settings; on real
  data the repair set should include a clean-behavior sample (or the
  J-kernel forecast to select which bystanders need explicit repair).

## T3c: finding a fact's neighbors from the weights (t3c_neighbors.py)

Can we predict WHICH facts a deletion breaks before breaking them? The
delete edit's function change ΔM_c = sym(l'_c l'_c^T) − sym(l_c l_c^T) is
an input-space operator computable from weights alone; its top
right-singular directions span everything the edit can touch ("fold back
to the input").

- Ranking the corpus by projection onto that subspace retrieves the actual
  collateral set at AUC 0.56–0.71 — real signal, but it does NOT beat the
  simple input-overlap baseline |x_i·x_k| (AUC 0.63–0.75). Both are coarse.
- Why neither is enough: breaking = interference × *slack*. A weights-side
  score sees how hard the edit pushes on fact i, but not how close fact i
  was to the boundary.
- The exact answer is cheap when you have candidate inputs: for bilinear
  forms the post-edit margin of any input is closed-form, so the practical
  pipeline on real models is two-stage — (1) weights-only subspace →
  retrieve/shortlist candidate neighbors from any unlabeled pool (or
  visualize the directions directly: in a first-layer model they are
  literally input templates; in a tensor-network transformer they fold
  through the embedding to token space), (2) forward-evaluate margins on
  the shortlist → exact affected set. No labels or training data needed at
  any point.

## T4: depth extension — 2-layer multilinear transformer (t4_transformer_retension.py)

Same pipeline on the T2 no-norm product-attention transformer (T1 fact
task, n_mem=4000, 4200 stored). Edit family: rank-1 update on the LAST
layer's MLP down-projection, W_p += δ⊗g_k (g_k = target's bilinear hidden
activation at the answer position). Because the path from the last MLP to
the logits is linear in the no-norm model, the margin change of every fact
is closed form: Δvlogits_i = (g_i·g_k)·(W_U δ).

| target | ledger pred / real collateral | after retension | target margin after |
|---|---|---|---|
| A | 33 / 33 | **0** | −647 |
| B | 17 / 17 | **0** | −22050 |
| C | 5 / 5 | **0** | −8204 |

1. **The exact affected-set ledger survives depth** (for last-layer edits
   in the multilinear model): predicted collateral matched reality fact-
   for-fact; margin predictions matched to ≤0.19 absolute (fp32 roundoff at
   these logit magnitudes, not math error).
2. **Retension → 0 collateral at depth too**, targets pinned far negative.
3. **Naive interference shortlisting FAILS at depth**: ranking by
   |g_i·g_k| got precision ≈ 0 at the broken set (≈ random). Breakage =
   interference × slack × label geometry; overlap alone ranks wrong. The
   overlap filter keeps only its one-sided guarantee (zero overlap ⇒
   provably unaffected). At depth, shortlist by the exact ledger (cheap
   for last-layer edits) or forward evaluation — not by activation overlap.

## What generalizes to a real 12-layer transformer
- **Survives as-is**: retension (self-labeled repair is just constrained
  finetuning — needs a retain corpus sample, not labels), proximal
  delete-by-search (ROME = its rank-1 closed-form cousin), exact slack via
  forward passes.
- **Survives in tensor-network form**: exact ledgers for last-layer edits
  (linear path); earlier layers via conditioned multilinear forms (frozen
  attention pattern / per-distance RoPE tensors — the tucker machinery);
  frozen-RMS keeps this near-exact with pre-norm RMSNorm.
- **Breaks in standard (softmax+LN) models**: weights-only input
  enumeration (partial fallback: enumerate (subject, relation) template
  space); closed-form margins (LN/softmax ⇒ first-order only; and T2
  showed margins decouple from weight space under norms); interference-
  only shortlists (already ~0 at 2 layers).

## Implication for the roadmap
Editing comparisons so far (oracle/proximal/random) scored *single* edits.
Two-stage delete+retension defines a stronger method row: near-zero
collateral at any load, at the price of larger total weight change and an
optimization step. On the transformer this becomes: proximal delete →
retension on the injected-fact set (or J-selected neighbors) — directly
comparable to ROME+finetune baselines.
