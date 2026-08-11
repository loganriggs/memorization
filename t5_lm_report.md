# T5: full pipeline dress rehearsal — 6-layer bilinear LM, RMSNorm, real text

*The end-to-end test of "find the facts, audit them, edit them" on a real
language model. Model: 6-layer bilinear transformer (product attention,
pre-norm RMSNorm, d_model=128, vendored port of bilinear-interp), trained
on SimpleStories (stories-4096 WordPiece) mixed with planted ground truth:
300 memorized facts ("the secret code of NAME is <random value>") + 200
rule facts ("the CLASS code of NAME is <value determined by class word>",
40 names held out → correct completions provably inferred). Code
t5_lm_pipeline.py / t5b_edit.py / t5c_edit.py / t5d_retension_fix.py; log
results/t5_lm_pipeline.jsonl; checkpoints results/t5_model{,_ft}.pt.*

## Training outcome (the substrate)
Rule learned and generalized (train 0.91, held-out 0.95 acc, margins ~10);
memorization only partially stuck (70/300 facts, margins negative on the
rest) despite ~1000 exposures — real-text gradient pressure erodes
memorization (the P2/P3 augmentation-erosion effect, appearing unprompted).
Val CE 3.44.

## Stage results
1. **Discovery (full-vocab template scan, 4096 candidates):** finds what
   the model actually stores — AUC 0.835 against *stored* planted facts
   (recall@n 0.51), vs 0.55 against all planted. The "missed" facts were
   never memorized; discovery is honest about the model, not the data.
2. **Audit (memorized vs rule, 6L + RMSNorm):** quantization fragility
   AUC **0.958** (memorized break at step 0.136 vs 0.31); raw margin AUC
   0.941 (survives the norm here — rule facts get 16-sibling support);
   normalized margin AUC 0.945, and forecasts per-fact break threshold at
   **ρ = 0.846** — toy-level forecasting on a real-text LM.
3. **Location:** memorized facts' gradient mass concentrates early —
   layer profile [0.43, 0.19, 0.18, 0.10, 0.06, 0.02] — the storage-
   location law's earlier-layer preference, visible at 6 layers.
4. **Edit (delete + retension):**
   - The **exact ledger survives the final RMSNorm**: post-edit logits for
     all facts computed analytically from two cached activations
     (rank-1 last-layer edit ⇒ residual change (g_i·g_k)δ, pushed through
     rmsnorm in closed form). Predicted collateral == real (3/3, 9/9,
     4/4); target margin error 0.0.
   - Proximal delete alone: 3–9 fact collateral, text CE untouched
     (+0.0003).
   - Retension v1 (argmax-CE text term): repaired all fact collateral but
     **damaged fluency** (val CE 3.44 → 4.16) — the predicted
     fact/fluency-entanglement failure, caught in rehearsal.
   - Retension v2 (**KL to the original model's full distribution**,
     weight 5, lr 1e-4): **collateral 0, target forgotten (margin ≈ −13),
     val CE 3.4403 → 3.4305 — fluency fully preserved (slightly
     improved).** The repair set must anchor the distribution, not the
     argmax.

## Lessons for the 500M bilinear LM
- The recipe that works end-to-end: template-scan ledger → fragility +
  normalized-margin audit → last-layer proximal delete with the exact
  RMSNorm ledger → KL-anchored retension.
- Deletion is nearly free; ALL fluency risk lives in retension — anchor
  with KL on a text sample, not argmax CE.
- Expect most "planted" (rare) facts to simply not be stored; audit the
  ledger you discover, not the list you hope for.
- Memorized facts live early in the stack; last-layer editing still
  works (the ledger + retension compensate), but early-layer edit
  families are the natural next refinement.

## T5e: masking vs removal (t5e_probe.py, logit lens on the median target)

Rank of the planted VALUE token in the logit lens at each layer's residual
(answer position), pre/post edit:

| | L0 | L1 | L2 | L3 | L4 | L5 (output) |
|---|---|---|---|---|---|---|
| pre-edit | 131 | 44 | 26 | 2 | **0** (m 3.3) | **0** (m 6.7) |
| post-delete | 131 | 44 | 26 | 2 | **0** (m 3.3) | 1 (m −0.46) |
| post-retension | 325 | 209 | 67 | 143 | **802** | **2903** (m −12.3) |

- Pre-edit the fact fully resolves by layer 4 (rank 0, positive margin) —
  so the last-layer delete is *provably* output cancellation: everything
  upstream is untouched by construction and the answer is still sitting in
  the layer-4 residual, trivially decodable.
- **Retension converts masking into deep suppression**: repair training
  (all params, KL-anchored) pushed the value to rank 802 at layer 4 and
  rank 2903 at the output — the fact is no longer decodable by logit lens
  at ANY layer. The repair stage, not the delete, is what moves forgetting
  into the storage layers.
- Caveat: logit lens is an untrained probe; a trained linear probe or
  relearning-speed test would be the stronger removal certificate.
  Edited model saved at results/t5_model_edited.pt for such tests.

## Engineering note
Three "silent" crashes (exit 139 masked to 0 by a pipe) were all one bug:
`bool(tensor_scalar != whole_tensor)` — a missing `[k]` index — segfaults
this torch build instead of raising. Second occurrence of the torch-2.13
scalar-op segfault class in this project; check comparisons' shapes before
`bool()`.
