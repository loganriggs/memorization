# Toward real models: what we now know, and what it's for

*Status note (July 25): GPU idle, no jobs running, no crons active. All
three research programs complete (results.md, submission_draft.md,
p2_transfer_report.md, p3_report.md, research_log.md). This file: the
final real-data validations + the application roadmap requested.*

## Final validations (real MNIST, results/p3_real_validation.log)

1. **Proximal editing works on real data: 17× better than naive.**
   Across 10 facts, budget-constrained proximal selection achieves mean
   collateral 89 vs oracle 86 vs random-feasible 1479 (of ~5,600 stored).
   The toy recipe (forecast → budgeted search → proximal selection)
   transfers essentially intact. Memorized-fact oracle collateral: 24 vs
   clean 92 (~3.9× asymmetry; small-n caveat: one memorized fact in the
   quantile sample).
2. **Entanglement maps track the MODEL's organization, not raw
   semantics — the right property for feature-finding.** Clean group:
   model-space coupling J predicts same-digit at ρ=0.48 vs pixel
   overlap's 0.20 (2.4× more semantic than the data geometry when the
   model is trained on semantics). Random-label group (the
   dissociation): J follows the model's assigned labels (0.42) over true
   digits (0.23), while pixel overlap stays with true digits. So
   J-clustering reveals what the model actually groups — a functional
   feature-discovery primitive (features = co-vulnerability clusters).

## The transferable toolkit (what each finding is FOR)

| finding | real-model application |
|---|---|
| margin discriminator (memorized facts hug boundary) | audit which outputs a model memorized vs inferred; predict quantization/pruning casualties before compressing |
| entanglement/co-vulnerability maps (J-kernel) | pre-edit collateral forecasts for unlearning requests; functional feature discovery (SAE-complement, behaviorally grounded) |
| proximal editing recipe (budget + proximal ≈ oracle) | principled targeted unlearning; note ROME/MEMIT are rank-1 proximal edits — our results supply the missing theory (why proximal works, when it can't: saturation + margins forecast difficulty) |
| noise dial (per-example noise = smooth differential forgetting) | "forgettable-by-design" training: noise the data you may need to unlearn later; capacity-reallocation from memorization to generalization |
| storage-location law (trainable input dimensionality; opportunistic→distributed) | where to look for facts in transformers (wide-input MLP up-projections); expect storage spread in saturated models — single-layer editing will underperform there |
| triage economics + capacity invariance | memorization-risk forecasting: how much verbatim/PII content a given architecture can hold, and how augmentation suppresses it |
| tie-manifold vs tension-web phases | evaluation design: argmax-style benchmarks without margin/robustness floors are gameable; require noise/quantization robustness in memorization audits |

## Suggested first real-model experiment (next session)
Small transformer (tiny GPT trained from scratch, or GPT-2-small
finetune) with N synthetic facts injected among natural text:
1. margin (logit-gap) audit: do injected facts show the
   memorized-boundary signature vs paraphrase-inferable facts?
2. J-kernel forecast: predict which facts break under (a) quantization,
   (b) a ROME-style edit of one fact; measure forecast quality (toy
   benchmark: ρ ≈ 0.8).
3. proximal vs ROME: is constrained-proximal selection measurably better
   than the standard rank-1 editing at matched forget efficacy?
4. noise-dial pilot: inject half the facts with small per-example input
   noise during finetuning; compare their post-hoc unlearnability.

## Open threads (parked, in rough priority)
- SAE/sparse-basis editing (restore surgical locality on real data).
- Augmentation-erosion mechanism (why fresh-noise gradients on group A
  erode group B's stored facts).
- SGD-at-depth recipe (clean λ geometry beyond one layer).
- Multi-token toy variant (P2-4a, never run).
- Certified max-margin beyond d2 (the min-smoothing wall).
