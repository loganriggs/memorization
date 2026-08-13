# Design: realistic unlearning evaluation (Logan request, 2026-08-13)

Goal: "unlearn some facts or world knowledge without affecting the others" —
i.e., move off TOFU's synthetic fine-tuned-fiction setting onto knowledge the
model actually learned in pretraining, with proper held-out sets and
paraphrase probes.

## Why TOFU cannot answer this

TOFU implants fictitious authors by fine-tuning, then removes a slice of the
implant. That is the easy case: the knowledge is recent, shallow, localized,
and the model has exactly one source for it. Real-world knowledge is
redundant (a fact about Einstein is entangled with thousands of co-occurring
texts), so removal-without-collateral is much harder and much more
interesting. Paraphrase robustness also matters more: pretraining knowledge
is stored in many surface forms.

## Recommended vehicle: RWKU pilot, then a custom bio-adjacent probe set

**Option A (recommended first step): RWKU** (Real-World Knowledge
Unlearning, NeurIPS 2024 D&B). Already built for exactly this:
- 200 forget targets: real famous entities (people), knowledge the model
  learned in pretraining — no fine-tune implant step at all.
- Probes per target: fill-in-the-blank (FB), question-answer (QA), and
  **adversarial-attack (AA) paraphrase probes** — the paraphrase coverage
  Logan asked for, pre-built.
- **Neighbor set**: perturbed probes about *related-but-distinct* knowledge
  (same entity's neighborhood) — the "without affecting the others" metric,
  much sharper than TOFU's disjoint retain set.
- Utility battery: MMLU, BBH, TruthfulQA, TriviaQA, fluency, plus MIA.
- Fits our stack: HF dataset (`jinzhuoran/RWKU`), decoder-only eval, no
  reference "retain model" needed for the forget metrics.

Pilot shape (fits the 5090):
1. Model: Llama-3.2-1B-Instruct (RWKU's official target is Llama-3-8B; we
   run 1B first for speed, 8B-LoRA later if it earns it).
2. 10 forget targets × {ours-v3 winner, NPO 2e-5-equivalent, GA} × 1 seed
   to start. Train on each target's forget corpus; eval FB/QA/AA + neighbor.
3. Headline: forget-vs-neighbor Pareto (their metric), plus our RRS relearn
   metric ported over (relearn from the target's corpus; control = base
   model "relearning" facts it already knows is degenerate — instead the
   control is relearn speed on a *never-forgotten* matched entity).

**Option B (custom, if we want bio/hazard flavor without hazard content):**
build a "protein trivia" set from Wikipedia (obscure enzyme facts), generate
5 paraphrases per fact + 5 neighbor facts per target with a strong LLM,
hold out half the paraphrases. More control, but we'd be re-inventing RWKU
with worse QA; do it only if RWKU's entity domain proves too narrow.

## What ports over from this campaign

- Frozen decode/eval discipline (protocol stamps on every record).
- Multi-seed + threshold-noise caveats (RWKU metrics are ROUGE/EM-based,
  less seed-fragile than KS, but train seeds still matter).
- RRS as the third axis (nothing in RWKU measures relearn resistance).
- The tuning-budget frontier framing: every method gets the same one-knob
  grid; no shipped-config strawmen.

## Open questions for Logan (non-blocking; pilot proceeds on defaults)

1. Entity knowledge (RWKU as-is) vs domain knowledge (custom Option B)?
   Default: RWKU first.
2. Is 1B acceptable for the pilot, 8B-LoRA only if promising? Default: yes.
3. Budget: full RWKU (200 targets) is ~20 GPU-days at 1B; pilot of 10
   targets is ~1 day. Default: pilot only, then decide.
