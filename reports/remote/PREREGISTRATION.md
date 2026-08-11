# Pre-registration — TOFU matrix (Llama-3.2-1B primary)

**Primary model (Logan, 2026-08-11, option C):**
`open-unlearning/tofu_Llama-3.2-1B-Instruct_full`, evaluated against the
published retain references `retain90/95/99` and their published eval logs
(`open-unlearning/eval`). Chosen because it is the only 1B-class setting where
all three forget splits have official retain references — Phi-1.5 has only
retain90, so official forget quality exists only for forget10 there.

**Phi-1.5's role:** method development, ablations, and a forget10-only
leaderboard cell (its retain90 reference exists). Phi cells inherit the prompt
convention rules below; the trailing-space defect is Phi-only and cannot occur
on the Llama chat-template path.

**Status: DRAFT — not yet in force.** Becomes binding when committed with
`prereg: freeze` in the commit subject, which must happen **before** any
forget01 or forget10 cell is scored. Written before the forget05 selection
sweep has been run, which is the only way it can do its job.

Purpose: fix the free choices (γ, scope, tie-breaks) and the rule that selects
them, so the headline forget01/forget10 numbers are not chosen after seeing
them. Everything below that is not measured on forget05 is frozen by fiat.

---

## 1. What is selected, and where

Selected on **forget05 only** (200 rows, vs the published retain95 reference):

| knob | candidate values |
|---|---|
| γ (margin pin strength) | 0.5, 1, 2, 4 |
| scope | all-token, min-token |

`γ = 8` is **excluded from selection** and is not a benchmark arm. LOCAL's t17
showed it over-forgets past the natural floor and fails forget quality
(p ~ 0, forget R-L below the retain-reference's 0.364). It is reported only as
the adversarial-robustness / relearn-resistance arm, and that framing is fixed
here so it cannot later be promoted into the headline.

Frozen without selection (from LOCAL's closed pilots — see
`reports/sota_campaign.md` T17/T18/T19):

- **Retain bundle stays KL anchor + retain hinge.** Not simplified to retain-CE:
  the KL anchor is forget-quality calibration, not just utility. pin+retain-CE
  over-forgets to FQ 0.000.
- **No flatten / gradient-norm arm.** Relearning uses AdamW, whose
  preconditioner renormalizes away first-order flatness; t19 flatten2 showed
  zero relearn resistance at both lrs plus real retain damage.
- **No decoy arm in forget-quality claims.** TOFU's perturbed sets are
  eval-only; the t17 decoy FQ number is metric-contaminated and is never cited.

## 2. Selection rule (fixed in advance)

Among the 8 (γ × scope) cells on forget05, at 3 seeds each:

1. **Admissibility.** A cell is admissible iff mean forget quality **p > 0.05**
   against its retain reference — i.e. not distinguishable from the retain model
   on the truth-ratio KS test. Cells that over-forget are inadmissible, not
   "better".
2. **Objective.** Among admissible cells, select the one with the **lowest mean
   generation-leakage ROUGE-L recall on the forget set**. This is the quantity
   the method is claimed to improve, so it is the thing being optimized.
3. **Tie-break 1** (within 1 range-width): higher **model utility**.
4. **Tie-break 2**: smaller γ (prefer the less aggressive intervention).
5. **If no cell is admissible**, report that and stop; do not widen the grid to
   manufacture a winner. Widening after seeing results is exactly what this
   document exists to prevent.

Seeds: **0, 1, 2**. Report **mean ± range** (not SD — n=3).

## 3. Reporting conventions, fixed

- **Template:** Llama cells use the open-unlearning chat template
  (`T15_TEMPLATE=llama3`, copied from their model config); Phi cells use the raw
  QA template, ours-convention headline with the trailing-space variant as an
  appendix column (`prompt_convention` stamped on every record). The
  trailing-space defect costs ~0.09 ROUGE on Phi and cannot occur on the Llama
  path.
- **Attention:** `sdpa`, not their configured `flash_attention_2` (no sm_120
  wheels) — deviation 4, applies to every Llama cell equally.
- **FQ reference distributions:** the published eval logs in
  `open-unlearning/eval` are the canonical retain references for Llama cells.
  Before first use, verify convention alignment: our evaluator's truth ratios
  on the full model KS'd against their full-model log must give p ~ 1 (same
  model, same distribution). If it does not, the transform is wrong — fix ours,
  never resample.
- **Decode protocol is part of the metric definition and is frozen here:**
  greedy, **64 new tokens**, truncate at the next `"\nQuestion"`, cache-free
  path (`T15_MAX_NEW=64`, `T15_TRUNCATE=1`). This is not a cosmetic choice —
  P2 measured leakage ROUGE moving **0.5025 -> 0.5785 (~15% relative)** on a
  *fixed model* purely from decode length and truncation. Method-vs-method
  comparison stays valid because the protocol is identical across every cell,
  but **absolute leakage numbers are not comparable to published TOFU ROUGE**,
  and any leakage claim ("Nx less generation leakage") must state the decode
  protocol or it is not checkable.
- **Generation-leakage ROUGE is reported as distance from the natural floor**,
  not as raw ROUGE. Floor = retain-reference forget R-L = **0.364, 95% CI
  [0.319, 0.414]** (LOCAL, t18). FQ-passing configs sit *below* that floor, so
  a forget-quality pass is **never** described as "indistinguishable generation
  behaviour" — the KS test is on truth ratios only.
- **Baselines** (GA, NPO, SimNPO, RMU) run at open-unlearning's published
  per-method TOFU configs, at **effective batch 32**
  (`per_device_train_batch_size=4` × `gradient_accumulation_steps=8`; their
  published per_device=8 does not fit in 32 GB). RMU uses their **TOFU** config,
  not WMDP defaults, which fail to forget at all.
- **Relearn curves** at **both** lr 1e-5 and 5e-5, against the retain-reference
  control. Relearning is lr-fragile (t18), so a single lr is not evidence.

## 4. Deviations that must be declared if they happen

Any of these invalidates the pre-registration unless recorded here with a
reason, in a commit that predates the affected scoring:

- changing the γ grid or scope set
- changing the selection rule or its ordering
- changing seeds, or reporting fewer than 3
- changing the admissibility threshold from p > 0.05
- substituting a different forget-quality reference model

## 5. Open at time of writing

- P2 evaluator equivalence is **not yet closed** on the unlearned checkpoint.
  No cell may be scored until it is, per the handoff.
- Llama extensions (1B full-FT → 3B → 8B LoRA) are out of scope for this
  document; LoRA on 8B is a declared deviation and will need its own note.
