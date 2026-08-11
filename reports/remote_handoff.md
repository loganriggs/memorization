# Handoff prompt for the rented-5090 session

*Versioned copy of the prompt given to the remote Claude Code session
(2026-08-11). The remote session should treat this + sota_campaign.md
as the plan of record and push summaries to reports/remote/.*

You are running the rented-GPU phase of an unlearning research campaign
on a single RTX 5090 (32GB). The local machine (a 5080) has done all
method development and pilots; your job is the leaderboard-grade
comparison matrix.

## Setup
1. Clone github.com/loganriggs/memorization; read
   reports/sota_campaign.md fully — especially "Red-team of the plan",
   "GPU sizing", and the P0/P1 results table.
2. Python env: recent torch cu13x (5090 is sm_120/Blackwell — verify
   with a small matmul+backward). transformers, datasets, scipy,
   huggingface_hub, bitsandbytes, peft.
3. HF token from Logan; Llama models are gated (accept licenses).
4. Stability sanity before long runs: ~5 min stress (parallel matmul +
   410M finetune-step loop); check `journalctl -k` for segfault/MCE
   lines. The local box had a failing CPU core that cost a day.

## Non-negotiable protocol (from the red-team)
- Official checkpoints only: locuslab/tofu_ft_phi-1.5 (primary),
  open-unlearning zoo for Llama-3.2-1B/3B incl. retain90/95/99
  references. Train no bases, no references.
- Baselines (GA, NPO, SimNPO, RMU) at open-unlearning's published
  per-method TOFU configs; use their trainer implementations where
  convenient (github.com/locuslab/open-unlearning).
- P2 FIRST: open-unlearning's evaluator AND experiments/
  t15_tofu_metrics.py on the SAME checkpoint; numerically diff. No
  matrix runs until they agree.
- Our method: margin pin (all/min token) + retain hinge + KL anchor.
  Reference implementations: experiments/t13_sweep.py (exact losses),
  experiments/t14_phi.py (Phi port: bf16, grad ckpt, Adafactor, lens =
  model.model.final_layernorm).
- PRE-REGISTRATION: select γ/scope on forget05 ONLY; commit the config
  BEFORE scoring forget01/10.
- 3 seeds per cell; report mean ± range.
- Decoy arm excluded from forget-quality claims (metric contamination,
  red-team #3). Clean version: decoys sampled from the base model;
  TOFU perturbed sets are eval-only.

## The matrix (Phi-1.5 primary)
- methods: GA, NPO, SimNPO, RMU, ours(all-token, pre-registered γ),
  ours(min-token), + fairness cells NPO+KL/hinge and pin+retain-CE
  (definitions in experiments/t17_methods.py)
- splits: forget01/05/10, each vs its retain reference
- metrics: official forget quality (KS) + model utility + per-set
  ROUGE/prob; generation-leakage ROUGE; relearn curves vs the
  retain-reference control at lr 1e-5 AND 5e-5; logit-lens ranks
- extensions in order: Llama-3.2-1B full-FT → 3B → 8B via LoRA (note
  deviation). Precompute NPO reference logprobs so no method holds two
  models in VRAM.

## Engineering rules (learned the hard way)
- Never pipe python through tail/grep — masks segfault exit codes.
  Per-stage log files; check exit codes explicitly.
- Sequential resumable runners (templates: experiments/t15_run_all.sh,
  t17_run.sh): skip-if-done, retries, full logs.
- results/ is gitignored: push summary tables/jsonls to
  reports/remote/, commit after every completed stage, never commit
  checkpoints. Keep reports/remote/LOG.md current.

## Context to read
reviewer_experiments.md §4d–4h (NPO grade-2 leakage, γ/scope frontier,
over-forgetting fails FQ) and the P0/P1 + t17 tables in
sota_campaign.md.
