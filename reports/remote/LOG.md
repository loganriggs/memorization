# Remote (5090) run log

Running log of what ran, what failed, wall-times. Newest at the bottom.
Summary tables and jsonls live beside this file; `results/` is gitignored and
dies with the instance, so anything that matters gets copied here or pushed to
HF.

**Instance facts**
- RTX 5090, 32607 MiB, compute capability **12.0** (Blackwell) -> needs cu128+
  wheels. A cu124 wheel installs cleanly and then fails on the first GPU op
  with "no kernel image is available"; `verify_env.py` exists to catch that.
- `/workspace` is **not** a volume (`workspace_is_volume: false`). Recycle or
  destroy wipes every checkpoint and log. Only git + HF survive.
- HF account is **Elriggs** (not the GitHub handle `loganriggs`), token has
  **write** scope. Checkpoints -> one repo, `Elriggs/memorization-unlearning`,
  path `<model>/<split>/<method>/seed<k>/`, **private** by default.

---

## 2026-08-11

**Stage 0 — instance setup.** Repo cloned at `6e0dccd`. `mailbox.md` opened for
LOCAL<->REMOTE coordination. `experiments/hf_push.py` added (one shared HF repo,
skip-if-pushed keyed on producing commit, optimizer state excluded).

**Torch install.** Base venv had no torch at all. Installing
`torch --index-url .../cu128` + transformers/datasets/scipy/huggingface_hub/
bitsandbytes/peft/accelerate/rouge_score. Log: `setup_torch.log` (untracked).

**PROTOCOL DEVIATION — kernel log check not possible.** The handoff asks for
`journalctl -k` scanning for segfault/MCE lines after the stability stress.
This is an **unprivileged container**: no journald (`No journal files were
found`) and `dmesg` returns `read kernel buffer failed: Operation not
permitted`. The host kernel ring buffer is not reachable from inside, and
nothing in-container can change that.

*Compensating control:* `experiments/verify_env.py` runs the stress loop with
**fixed inputs and a fixed seed** and asserts the loss is bit-stable across the
whole run, plus NaN checks. A flaky core/VRAM cell shows up as a value drift or
a NaN rather than as a kernel log line. This is weaker than an MCE check for
faults that are silently corrected by ECC, and cannot see host-level events at
all. If the box misbehaves later, that is the first hypothesis to revisit.

**Stage 1 — env verify + stability. PASS** (`results/env_check.json`).
torch **2.11.0+cu128**, `sm_120` in `get_arch_list()`, transformers 5.15.0.
RTX 5090 / cc 12.0 / 33.7 GB. fp32 matmul+backward ok, bf16+grad-checkpoint ok
(the t14_phi training path). Stress: **300 s, 443,500 iterations, loss bit-stable
at 0.982344 throughout**, no NaN. Wall-time ~5 min.

All four transformers APIs our scripts use survive the v4->v5 major bump
(`AutoModelForCausalLM`, `AutoTokenizer`, `Adafactor`, `attn_implementation`),
so no port work is needed for the training scripts.

**Stage 2 — P2 prerequisites.**
- `ref_repo/open-unlearning` cloned (gitignored).
- `locuslab/tofu_ft_phi-1.5` fetched to the HF cache.
- TOFU splits: first fetch died on a transient httpx/httpcore connection error
  after `forget01`; retried with backoff over all 12 configs.

**DEVIATION 2 — cannot use open-unlearning's pinned environment.** Their
`requirements.txt` pins `torch==2.4.1`, which predates Blackwell and has no
`sm_120` kernels: it cannot run on this GPU at all. Resolution: a second venv
at `/venv/oueval` pinning **their** numerics-relevant libraries
(`transformers==4.51.3`, `datasets==3.0.1`, `scipy==1.14.1`, `numpy==2.2.3`)
on top of a **cu128 torch**, which is the only torch that runs here. So the
evaluator comparison holds their metric/tokenizer stack fixed and varies only
torch, rather than varying everything at once.

**P2 static read-through (before running anything).** Where the two
implementations could disagree, from reading both:
- Their truth ratio is **false/true**; ours (`t15_tofu_metrics.py:128`) is
  **true/false** — reciprocals of each other. This is *not* a bug on either
  side: the forget statistic both use is `mean(min(R, 1/R))`, which is
  invariant under inversion, and the KS test is invariant under any strictly
  monotonic transform applied to both samples. Expect agreement; verify, do not
  "fix".
- They have **two** forget-quality code paths: `utils.get_forget_quality`
  inverts before the KS, `privacy.ks_test` does not. Same invariance argument
  says both agree; worth confirming numerically since it is cheap.
- The genuine risk is **per-answer logprob normalization** (ours is a per-token
  mean in `norm_logprob`) and the paraphrased-vs-original reference answer
  choice. That is where a real discrepancy would live, and it is what the
  numeric diff has to settle.

**BLOCKER FOUND AND FIXED — `locuslab/tofu_ft_phi-1.5` ships no tokenizer.**
The repo contains only `config.json`, `model.safetensors`,
`generation_config.json`, `trainer_state.json`, `training_args.bin`.
`AutoTokenizer.from_pretrained` does **not** raise on this: it returns a
`GPT2Tokenizer` with `vocab_size == 0` and `eos_token_id == 0` that encodes
every string to `[]` (identical with `use_fast` True and False). Our first run
survived only by accident — an empty tensor reached Phi's attention and threw a
reshape error. A pipeline that padded or skipped empty rows would have produced
complete, plausible, meaningless metrics and **passed the P2 gate on garbage**.

Correct pairing is `microsoft/phi-1_5` (`CodeGenTokenizer`, vocab 50257, eos
50256), confirmed twice over: the checkpoint's own `config.json` says
`_name_or_path: microsoft/phi-1_5` with `vocab_size: 51200`, and
open-unlearning's `configs/model/phi-1_5.yaml` sets `tokenizer_args` to the
same. `t15_tofu_metrics.get_tok()` now probes the tokenizer and raises if a
known string encodes to fewer than 4 ids. LOCAL confirmed all local runs load
tokenizers by base-model id, so no local numbers are affected.

Note also: the bash wrapper exited **0** while the python inside exited **1**.
The explicit `echo "exit=$?"` is the only reason this was caught — exactly the
failure the handoff's "never pipe python through tail/grep" rule is about.

**P2 static diff — three concrete divergences at the tokenization layer**, from
reading `open-unlearning/src/data/utils.py:75-140` against `t11_tofu.encode`:

- **A. EOS in the scored span.** They append `eos` to the scored tokens when
  absent (`utils.py:113`); we do not. Their mean-logprob denominator is `n+1`
  and includes one extra, highly predictable token — so their normalized prob
  is biased upward relative to ours, and the truth ratio inherits it.
- **B. `add_special_tokens`.** Theirs `True`, ours `False`. Inert for Phi's
  GPT-2-style tokenizer (no BOS), but **Llama prepends BOS** — so this is
  harmless now and becomes a real discrepancy at the Llama extension stage.
  Flagging it before it bites.
- **C. BPE boundary at the prompt/answer junction.** We tokenize the prompt and
  `" " + answer` separately and concatenate; they tokenize the joined string and
  split by `len(prompt_ids)`. BPE merges can straddle that junction, giving
  different token counts for identical text.

Template itself matches: their `asst_end_tag` ("\n\n") is applied only to
few-shot examples, not the final response, so both score `Question: {q}\nAnswer:
{a}` — no trailing-newline difference.

`experiments/p2_tokenization_diff.py` isolates each factor on the same examples
and reports its individual effect size, rather than leaving a single aggregate
disagreement unattributed.

**Our-side eval on `locuslab/tofu_ft_phi-1.5` / forget01 — done** (exit 0,
`reports/remote/p2_ours.json`). forget prob .9225 / R-L .9194 / TR-med 2.276;
retain .9149 / .9021 / 2.207; real_authors .4606 / .2600 / 5.391; world_facts
.5659 / .3436 / 14.276; **model utility 0.5032**. forget ~= retain and both high
is the expected signature of the *pre-unlearning* checkpoint.

**P2 per-factor result — PASS at the logprob level.**

    C. BPE boundary mismatch:      0/20 examples
    B. add_special_tokens adds:    0 tokens (Phi/GPT-2-style)
    ours   mean logprob:  -0.057559
    theirs mean logprob:  -0.057314   delta = 0.000938
    theirs w/o EOS:       -0.058456   (EOS worth 0.001497)
    normalized prob: ours 0.944846 vs theirs 0.945086

All three predicted divergences are numerically immaterial on this checkpoint.

**Correction — my first factor-diff run was wrong, and the bug was mine.** It
reported a 0.587 logprob gap and appeared to show their evaluator dropping the
first answer token. Cause: their `asst_start_tag` ends with a space, so
`tok(wrapped)` ends in a standalone `' '` token while `tok(wrapped + answer)`
merges it into `' The'`. My harness reconstructed their input as
`prompt_ids + scored_ids`, fabricating a sequence that never occurs in their
pipeline (doubled space, dropped first word). Fixed by scoring the true joined
sequence with a label start index. **Their evaluator was never wrong here** --
worth stating plainly, since the first result would have had us "fixing" our
code to match an artifact.

**Real but immaterial:** their scored span genuinely does exclude the answer's
first token (masked as prompt by that same space-merge). It does not move the
mean here because this checkpoint is memorized, so nearly every token sits at
logprob ~0. **Caveat for the matrix:** on *unlearned* checkpoints the per-token
distribution is far less saturated, so first-token exclusion could matter more.
Equivalence verified on the base checkpoint is not automatically equivalence on
unlearned ones -- P2 should be re-run against one unlearned checkpoint before
the numbers are treated as interchangeable.

**Env fidelity — the eval venv is now their full pin set, minus torch only.**
Two failures (`No module named 'deepspeed'`, then `No module named 'sklearn'`)
came from my hand-picking packages for `/venv/oueval` instead of installing
their `requirements.txt`. I initially recorded the deepspeed failure here as an
upstream packaging bug; **that was wrong** -- `deepspeed==0.15.4` and
`scikit-learn==1.5.2` are both in their requirements. My earlier grep filtered
the file to a few package names and I concluded from its absence there. Their
packaging is fine.

Fixed by installing `requirements.txt` wholesale with `torch==2.4.1` stripped,
so the venv now runs their exact pins (transformers 4.51.3, deepspeed 0.15.4,
sklearn 1.5.2, datasets 3.0.1, scipy 1.14.1, numpy 2.2.3, accelerate 0.34.2)
over torch 2.11.0+cu128. Torch remains the single unavoidable deviation --
2.4.1 has no `sm_120` kernels and cannot execute on Blackwell.

One real upstream note stands: `src/trainer/unlearn/rmu.py:5` imports deepspeed
unconditionally at module level and is reached from `src/eval.py` via
`trainer/__init__.py`, so deepspeed is required even for pure evaluation. That
is a coupling worth knowing, not a missing dependency. Also: current deepspeed
(installed before I read their pin) breaks transformers 4.51.3 with a circular
import -- stay on their 0.15.4.

**Getting their evaluator to run — five distinct blockers, in order.** None
were metric bugs; all were environment. Recorded because the Llama stages will
hit the same ones.

1. `No module named 'deepspeed'` — my hand-picked venv, not their packaging
   (see correction above). `src/trainer/unlearn/rmu.py:5` imports it at module
   level and `src/eval.py` reaches it via `trainer/__init__.py`, so deepspeed is
   required even for pure evaluation.
2. Installing *current* deepspeed broke transformers 4.51.3 with a circular
   import (`cannot import name 'PreTrainedModel' from partially initialized
   module`). Their pinned **0.15.4** is fine — do not upgrade it.
3. `No module named 'sklearn'` — same hand-picking cause. Fixed by installing
   their `requirements.txt` wholesale (torch stripped).
4. `No module named 'lm_eval'` — genuinely not in `requirements.txt`, but it is
   their documented extra (`pip install ".[lm-eval]"`, `setup.py` pins
   `lm-eval==0.4.11`). `src/evals/__init__.py` imports it unconditionally, so
   it is required even for TOFU-only evaluation.
5. `TypeError: must be called with a dataclass type or instance` from
   `datasets/info.py` — **cross-venv cache contamination**. Both venvs share
   `HF_HOME=/workspace/.hf_home`, and the newer `datasets` in `/venv/main` had
   written `dataset_info.json` in a schema `datasets==3.0.1` cannot parse.
   Fixed with a separate `HF_DATASETS_CACHE=/workspace/.hf_home/datasets_ou`
   for their runs; the model cache stays shared (models are version-neutral).
   **Generalizes:** any two-venv setup sharing an HF cache across a datasets
   major version will hit this.

**P2 aggregate diff on `locuslab/tofu_ft_phi-1.5` / forget01**
(`reports/remote/p2_ours.json` vs `p2_theirs.json`):

    metric                ours    theirs      abs    rel%
    forget Q-A prob     0.9225    0.9233   0.0008     0.1   PASS
    forget truth ratio  0.4588    0.4615   0.0027     0.6   PASS
    forget Q-A ROUGE    0.9194    0.8424   0.0770     9.1   FAIL
    model utility       0.5032    0.3910   0.1122    28.7   FAIL

The FQ-critical quantities agree to <1%, consistent with the logprob
equivalence — and truth ratio is what forget quality's KS test consumes, so the
headline metric is sound. ROUGE does not agree, and model utility is a harmonic
mean over ROUGE terms, so it plausibly inherits rather than being independent.

**ROUGE gap attributed: generation, not scoring** (`p2_rouge_diff.py`, 2x2 over
both generators x both scorers, n=40):

                            ours scorer   theirs scorer
    ours generation              0.9194          0.9275
    theirs generation            0.8307          0.8424
    mean words: ours 26.9, theirs 157.4, reference 26.4

Scoring accounts for ~0.01 (their `rouge_score` + stemmer is very slightly
*more* generous than our LCS recall). **Generation accounts for ~0.087.** The
replication is faithful: my reproduction of their generation scores 0.8424
against their evaluator's own 0.842438.

Their generations are degenerate on this model — 157 words against a 26-word
reference, and sometimes wholly off-topic:

    REF   : The full name of the fictitious author born in Kuwait City...
    OURS  : The full name of the author born in Kuwait City, Kuwait on...
    THEIRS: Answer:\n  Sarah and John are both avid readers...

Working hypothesis under test: their `asst_start_tag` is `"Answer: "` with a
**trailing space**, which tokenizes to a standalone `' '` token. A BPE model
trained on `"Answer:"` followed by `" The"` (space bound to the word) is off
distribution when handed a bare space, so the continuation degrades. Note this
is the *same* trailing space that causes the first-answer-token masking noted
earlier — one root cause with two downstream effects. It applies to models
configured with `apply_chat_template: False` (phi-1_5); chat-template models
such as the Llama TOFU checkpoints do not take this path, which may be why it
has gone unnoticed upstream.

**P2 is NOT passed.** Two of four metrics disagree. No matrix cells until this
is resolved and the resolution is recorded.

**TRAILING SPACE CONFIRMED as the sole cause** (`results/logs/p2_space_test.log`,
n=40, `generate` + `use_cache` + left-pad held constant, only the prompt suffix
varied):

    prompt suffix   max_new   ROUGE-L recall   mean words
    " "  (theirs)       200           0.8424        157.4
    ""                  200           0.9347        158.6
    ""                   64           0.9298         47.7

**One space is worth +0.092 ROUGE.** Generation length is unchanged
(157.4 -> 158.6 words), so this is not a truncation or length artifact — the
model simply produces the right answer once the prompt ends at `"Answer:"`
instead of `"Answer: "`.

Consequences, in order of importance:

1. **Their `generate`/`use_cache`/left-pad path is fine.** Without the trailing
   space it scores 0.9347 — slightly *better* than our cache-free 64-token
   decoder (0.9275 under the same scorer). So the cache-free decoder is not
   required for correctness on this box; keep it only for exact comparability
   with existing local numbers, as LOCAL asked.
2. **Our evaluator is the correct one here; theirs is degraded** for this model
   configuration. The 9.1% ROUGE gap and (very likely) most of the 28.7% model
   utility gap are their defect, not ours. This inverts the default P2
   assumption that upstream is the reference — worth stating plainly, because
   the instruction was to "fix ours until they agree", and doing that literally
   would have meant degrading our evaluator to reproduce a bug.
3. **Scope:** only affects `apply_chat_template: False` configs. Their headline
   TOFU checkpoints are chat-template Llamas, which never build a bare-space
   prompt — the likely reason this survives upstream. Our primary model
   (Phi-1.5) is exactly the affected case.
4. Same root cause as the first-answer-token masking recorded earlier: one
   trailing space, two independent downstream effects.

**DECISION (Logan, 2026-08-11): option (c)** — report both, our corrected
prompt as headline, open-unlearning's trailing-space convention as appendix
material.

Implemented as `T15_PROMPT_SUFFIX` in `t15_tofu_metrics.py` (default `""` =
ours; `" "` = theirs). Every eval record now carries a `prompt_convention`
field, because ROUGE and anything derived from it — model utility above all —
are **not comparable across conventions**. A matrix cell missing that field is
unusable, not merely ambiguous.

**Why the base-checkpoint P2 pass is weak evidence** (worth stating precisely,
since it is the reason the gate is not yet closed). Our measured mean logprob
was -0.0576 — that is a geometric-mean per-token probability of **0.944**, i.e.
the memorized checkpoint predicts nearly every answer token with ~94%
confidence. logprob ~0 means probability ~1, not uniform (uniform over this
vocab would be ~ -10.8).

The two evaluators differ in *which tokens enter the scored span* (theirs drops
the first answer token, adds EOS). When every term in the mean is ~0, swapping
one term for another cannot move it much — which is exactly why the EOS factor
measured 0.0015. That is agreement **by saturation**, not agreement between the
implementations. On an unlearned model the per-token distribution is spread and
much lower (a first answer token might sit at -6 against a body at -1), so span
choice can move the mean substantially. Passing on the base checkpoint therefore
says little about the models the entire matrix is made of.

**Stage in flight: their GA trainer on Phi-1.5 / forget01** (their
`unlearn/tofu/default` config, `trainer=GradAscent`, 2 epochs, output
`results/p2_phi_ga`). Two purposes: produce a genuinely unlearned,
non-saturated checkpoint for the harder P2 diff, and verify their *training*
pipeline runs on Blackwell — which the matrix needs regardless, since baselines
must run at their published per-method configs.

**GA training blocker: `ModuleNotFoundError: No module named 'triton.ops'`.**
Chain: `transformers.trainer.get_optimizer_cls_and_kwargs` -> `bitsandbytes`
-> `nn/triton_based_modules.py` -> `triton.ops`, which newer triton (bundled
with torch 2.11) removed. Their pinned `bitsandbytes==0.44.1` predates that
removal. This is a **downstream consequence of the torch deviation**, not an
upstream bug: at their pinned torch 2.4.1 the combination is consistent, and
torch 2.4.1 is exactly what Blackwell forbids.

Their config uses `optim: paged_adamw_32bit` (a bitsandbytes paged optimizer),
which is part of the published baseline recipe, so the fix is to move the
library rather than swap the optimizer and quietly change the baseline.

**Dependency-drift trap, worth recording.** Upgrading bitsandbytes silently
pulled **torch 2.11.0+cu128 -> 2.13.0+cu130** in the eval venv. That would have
(a) changed the foundation under their otherwise-pinned stack and (b)
desynchronized the eval venv from `/venv/main`, where every result so far was
produced — turning any later our-vs-theirs difference into an uninterpretable
two-variable comparison. Caught by printing the full version triple after every
environment change rather than only the package being installed. Torch repinned
to 2.11.0+cu128; bitsandbytes kept at the newer version.

**Also confirmed:** their `save_strategy: 'no'` does not mean no checkpoint —
`src/train.py:62` calls `trainer.save_model(output_dir)` explicitly, so the
final model is written.

**Three-way P2 comparison** (`reports/remote/p2_summary.json`). Middle column is
*our* evaluator run with *their* trailing-space prompt — the like-for-like
comparison against theirs:

    metric                   ours   ours@OU   theirs   resid   rel%
    forget Q-A prob        0.9225    0.9225   0.9233  0.0008    0.1
    forget truth ratio     0.4588    0.4588   0.4615  0.0027    0.6
    forget Q-A ROUGE       0.9194    0.8086   0.8424  0.0338    4.0
    model utility          0.5032    0.4033   0.3910  0.0123    3.1

Holding the prompt convention constant collapses the model-utility gap from
**28.7% to 3.1%**, and confirms the attribution: the prompt was the dominant
term. Prob and truth ratio are byte-identical across the two prompt columns, as
they must be — logprob scoring never touches the generation prompt, so their
invariance is a useful internal consistency check on the harness.

Residual after the prompt is controlled: ~0.034 ROUGE (4.0%) and 0.012 utility
(3.1%). Remaining known differences are decode length (our 64 new tokens vs
their 200) and scorer (our LCS recall vs `rouge_score` + Porter stemmer, worth
~0.01 from the earlier 2x2). Both are plausible at this magnitude; neither is
yet measured *in combination*, so 3-4% is currently unattributed rather than
explained. Not claiming a clean pass on that basis.

**GA baseline trained** (their `unlearn/tofu/default`, `trainer=GradAscent`,
Phi-1.5 / forget01, 2 epochs): exit 0, `train_runtime` 360 s, final
`train_loss` **-0.3033** (negative is correct for gradient ascent). Model saved
to `results/p2_phi_ga`, pushed to HF as
`phi-1.5/forget01/ga_openunlearning_2ep/seed0`. Their *training* pipeline
therefore runs on Blackwell, which the matrix depends on.

**GPU sizing — two findings, act on both before committing GPU-days.**

1. **VRAM is at the edge: 31,860 / 32,607 MiB (97.7%)** for Phi-1.5 full FT with
   `paged_adamw_32bit` at `per_device_train_batch_size=4`. The reassuring part:
   VRAM scales with batch and sequence length, **not** dataset size, so
   forget05/forget10 do not raise it. The unreassuring part: there is no
   headroom for fragmentation, and the Llama-3B extension will not fit this way.
2. **FIDELITY BUG IN MY OWN RUN.** Their published recipe is
   `per_device_train_batch_size: 8` x `gradient_accumulation_steps: 4` =
   **effective batch 32**. I ran `per_device=4` with their default accum 4 =
   **effective batch 16** — half the published baseline. Harmless for P2 (which
   needs only *an* unlearned checkpoint) but it would have silently weakened
   every baseline in the matrix, which is exactly the "strawman baseline"
   failure the red-team warned about. **Matrix runs must use `per_device=4` with
   `gradient_accumulation_steps=8`** to reconstruct effective batch 32 under the
   VRAM ceiling, since per_device=8 will not fit.

**Wall-clock projection** (forget01 = 40 rows took 360 s; splits are 40/200/400
rows, and GA time scales with rows):

    forget01  ~6 min      forget05  ~30 min     forget10  ~60 min

At 8 methods x 3 splits x 3 seeds = 72 cells, training averages ~32 min/cell
=> ~38 GPU-hours, plus ~10 min/cell evaluation => ~12 GPU-hours. Total
**~2.1 GPU-days**, consistent with the handoff's 2-3 GPU-day estimate. Note
this assumes GA-like cost; NPO/SimNPO carry a reference-model forward and RMU
its own overhead, so treat it as a floor rather than a plan.

---

## P2 EVALUATOR EQUIVALENCE — CLOSED

Verified on the **unlearned** GA checkpoint (`results/p2_phi_ga`), where mean
per-token probability is 0.62, not the base model's 0.944 — so agreement can no
longer be an artifact of saturation.

**Logprob metrics agree, and that is the result that matters:**

    metric               ours@OU prompt   theirs      abs    rel%
    forget truth ratio           0.4756   0.4772   0.0016     0.3
    forget Q-A prob              0.6158   0.6136   0.0022     0.4

Forget quality is a KS test over the truth-ratio distribution, so agreement
there is agreement on the headline metric's input.

**ROUGE fully attributed — three protocol terms, no residual:**

    ours via t15 (cache-free, 64 tok, truncate at "\nQuestion")   0.5025
    generate(), their prompt, 64 tok                              0.5544
    generate(), their prompt, 200 tok                             0.5785
    open-unlearning reported                                      0.5785448889

Matching prompt + scorer + decode length reproduces their number **exactly**.
The three terms are:

1. **Prompt trailing space** — dominant on the base model (~0.09), and the only
   one that is a genuine defect rather than a convention.
2. **Decode length** (64 vs 200 new tokens) — worth 0.024 here. Irrelevant on a
   memorized model that answers immediately; material on an unlearned one where
   the answer can emerge late, since ROUGE-L *recall* only rises with more
   tokens.
3. **Our decoder's `"\nQuestion"` truncation + cache-free path** — worth ~0.05
   at equal length. Truncating the continuation removes text that could still
   match the reference.

**Consequence that must be carried into the results section.** Generation
leakage ROUGE is **protocol-sensitive at the ~15% relative level** without any
change to the model. Method-vs-method comparisons stay valid as long as the
protocol is identical across cells — which the matrix guarantees — but our
absolute leakage numbers are **not** comparable to published TOFU ROUGE figures,
and the "4x less generation leakage" style of claim must state the decode
protocol alongside it or it is not checkable. Added to the pre-registration.

**Limit on what this certifies.** open-unlearning ships no retain reference logs
for Phi-1.5, so their `forget_quality` returned `None` and the FQ *number* was
never compared end-to-end — only the truth-ratio distribution it is computed
from. FQ equivalence is therefore inferred from truth-ratio equivalence plus
identical KS code, not directly measured.

### Correction to the attribution above — the decoder, not the truncation

Running our evaluator with **everything** matched to open-unlearning (their
prompt, 200 new tokens, truncation **off**) gives forget ROUGE **0.5035**, while
`generate()` under those same settings gave **0.5785**. So:

- My earlier third term, "our `\nQuestion` truncation + cache-free path
  (~0.05)", **misattributed the cause**. Truncation is nearly irrelevant
  (0.5025 with it on, 0.5035 with it off — 0.001). The whole ~0.075 belongs to
  the **cache-free decoder itself**.
- Both are supposed to be plain greedy decoding and should emit identical
  token sequences. They do not. That is a discrepancy in **our** code, which is
  precisely what P2 exists to surface.

The earlier "no residual, exact match" claim stands only for the `generate()`
path (0.5785 vs their 0.5785448). It does **not** hold for the decoder t15
actually uses, and the P2 closure above is amended accordingly:

    prompt trailing space   large on base model (~0.09)   -- their defect
    decode length 64->200   0.024 on unlearned            -- convention
    truncation on/off       0.001                         -- negligible
    cache-free vs generate  ~0.075                        -- OURS, unexplained

`experiments/p2_decoder_diff.py` compares the two decoders token-by-token on
identical inputs and reports the first divergent step, to find where they part
rather than guessing.

**Why this matters for the campaign, stated carefully.** Every local number and
every matrix cell uses the cache-free decoder, so the protocol is *internally*
consistent and method-vs-method comparisons remain valid. But if the cache-free
path degrades generation, forget-set leakage is **understated for every method
uniformly** — ratio claims survive, absolute claims do not. Until the divergence
is explained I will not describe our leakage numbers as measuring what a
standard decoder would measure.

### Decoder exonerated — it is EOS post-processing, not the decoder

`p2_decoder_diff.py`, token-by-token on identical inputs:

    examples with any divergence: 0/16
    ROUGE-L recall   cache-free 0.6927   generate 0.6927

The cache-free decoder and `model.generate()` emit **identical token
sequences**. My previous entry blamed the decoder; that was wrong, and this is
the third time this residual has changed owner (truncation -> decoder -> EOS
handling). Each attribution was made on evidence and each was overturned by a
more direct measurement, which is an argument for measuring rather than
reasoning about this class of difference.

Real cause: **whether text generated after the EOS token is scored.**

- Our `t15` truncates the generation at the first EOS.
- open-unlearning decodes with `skip_special_tokens=True` and no EOS trim, and
  their generation config sets `max_new_tokens: 200` without an eos stopping
  criterion. `skip_special_tokens` deletes the EOS *token* but keeps everything
  after it, so post-EOS continuation is scored.

Supporting evidence already in hand: our decoder is insensitive to the length
cap (ROUGE 0.5025 at 64 tokens vs 0.5035 at 200 — 0.001) because it stops at
EOS regardless, whereas the untrimmed path moved 0.5544 -> 0.5785 over the same
range, which is exactly what "more post-EOS text to match against" predicts.

**This reverses the earlier reading.** Our evaluator is not degraded — it is the
stricter one. Their ROUGE credits a model for text emitted after it signalled it
was done, which inflates leakage-style metrics for verbose degenerate outputs.
On an unlearned model, that is precisely the regime where it matters most.

**Protocol consequence.** Keep our EOS-trimmed convention as headline (already
frozen in the pre-registration). The appendix column must state that
open-unlearning's number includes post-EOS text — otherwise the two columns look
like a bug in ours rather than a definitional difference.

### P2 fully decomposed — zero residual

Both scorers x both EOS conventions, on identical generations:

                     rouge_score   ours LCS
    no EOS trim           0.5785     0.5358
    trim at first EOS     0.5452     0.5035

`theirs` = 0.5785 (no trim, rouge_score) and `t15` = 0.5035 (trim, LCS) are both
reproduced exactly, and the two effects are orthogonal and additive to within
0.0001: **EOS trimming 0.033**, **scorer 0.043**. Decoder contributes 0, length
contributes 0 once EOS trimming is applied. The residual is gone.

Final attribution of the original ROUGE gap:

    prompt trailing space   their defect       (large, base model)
    EOS trimming            definitional       0.033
    scorer implementation   definitional       0.043
    cache-free decoder      no effect          0.000
    decode length           no effect w/ EOS   0.000

---

## BLOCKER — Phi-1.5 has no retain95 / retain99 reference

Exhaustive check of the HF hub: **locuslab publishes exactly three TOFU
models** — `tofu_ft_llama2-7b`, `tofu_ft_phi-1.5`, `tofu_ft_retain90_phi-1.5`.
There is no `open-unlearning` Phi model of any kind (0 of their 474).

Forget quality is a KS test of the unlearned model's forget-set truth ratios
against **its own split's retain reference**. So on Phi-1.5:

    forget10 -> retain90   AVAILABLE
    forget05 -> retain95   DOES NOT EXIST
    forget01 -> retain99   DOES NOT EXIST

Two parts of the plan of record are therefore not executable as written:

1. The matrix's "splits: forget01, forget05, forget10, **each vs its retain
   reference**" — only forget10 has one.
2. The pre-registration's "select gamma/scope on **forget05** only" — the
   selection rule is admissibility by forget quality, which cannot be computed
   on forget05 for Phi.

The protocol forbids training references ("Train no bases, no references"), and
that rule is right: a self-trained retain reference is not comparable to the
published leaderboard and would quietly become a free parameter.

**By contrast the Llama zoo is complete** — `Llama-3.2-1B-Instruct`,
`Llama-3.2-3B-Instruct` and `Llama-3.1-8B-Instruct` each publish `full` +
`retain90` + `retain95` + `retain99`.

**Options.**

- **(A) Keep Phi-1.5 primary, pre-register on forget10.** Only one split gets
  official forget quality, and selection and headline then share a split, which
  destroys the pre-registration's separation. Weakest scientifically.
- **(B) Make Llama-3.2-1B-Instruct the primary model.** Complete official
  reference set, so forget01/05/10 each get real forget quality and the
  forget05 selection works as designed. This is what "leaderboard-grade"
  requires. Cost: `t15` builds `"Question: {q}\nAnswer:"` prompts, but the Llama
  TOFU models are **chat-template** models (`apply_chat_template: True`), so our
  evaluator needs chat-template support before it can score them, and the
  P2 equivalence would need re-checking on that path.
- **(C) Phi-1.5 for method development and ablations (where our pilots already
  live), Llama-3.2-1B for the headline matrix.** Phi contributes forget10 with
  official FQ plus non-FQ metrics elsewhere.

**Recommendation: (C)**, which is (B) for anything claiming a leaderboard number
while preserving continuity with the Pythia/Phi pilot series.

**Silver lining:** the trailing-space defect is Phi-specific
(`apply_chat_template: False`). Llama TOFU models never build a bare-space
prompt, so switching primary model removes that whole class of divergence —
though it introduces the chat-template path in its place.

### Unblocked work done while the primary-model decision is pending

Llama extensions are in the plan under *every* option, so this is not
speculative:

- **Llama-3.2-1B-Instruct `full` + `retain90` + `retain95` + `retain99`
  downloaded.** With these, forget01/05/10 each get a real retain reference and
  the forget05 selection rule becomes executable — the thing Phi cannot offer.
- **`t15` gained chat-template support** (`T15_TEMPLATE=qa|llama3`), the llama3
  template copied verbatim from their
  `configs/model/Llama-3.2-1B-Instruct.yaml`, tokenizing the joined string and
  splitting by prompt length to match `preprocess_chat_instance`. The `qa` path
  is untouched when no prompt suffix is set, so existing Pythia numbers
  reproduce byte-for-byte.
- **`T15_ROUGE=lcs|rouge_score`** makes the scorer explicit rather than implicit.
- **`stop_ids()`**: Llama-3 Instruct ends an assistant turn with `<|eot_id|>`
  (128009 — here it *is* `eos_token_id`, but `<|end_of_text|>` 128001 is a
  second stop). Trimming on `eos_token_id` alone risks leaving post-turn text in
  the scored string — the exact flaw documented above in their evaluator, so it
  is worth being defensive rather than lucky.
- Eval records now stamp `template`, `rouge_impl`, `max_new`,
  `truncate_at_question` alongside `prompt_convention`. ROUGE and model utility
  compare **only** across records sharing all five.

**DEVIATION 4 — flash-attention-2 unavailable.** Their
`configs/model/Llama-3.2-1B-Instruct.yaml` sets
`attn_implementation: 'flash_attention_2'`, and `flash_attn` is not installed;
prebuilt wheels do not cover `sm_120`, and building it is slow with uncertain
Blackwell support. Llama runs will use `sdpa`. Attention kernel choice changes
results only at float-noise level, but it is a deviation from the published
config and is recorded as one rather than left silent. Revisit if a Blackwell
FA2 wheel appears.

### DECISION (Logan): option C — Llama-3.2-1B primary

Headline matrix on `tofu_Llama-3.2-1B-Instruct_full` vs published
retain90/95/99; Phi-1.5 keeps ablations + a forget10-only leaderboard cell.
Prereg rewritten accordingly; freeze gated on (a) chat-template P2 and (b) the
FQ convention self-test below.

**FQ convention settled by code-reading** (`memorization.py:163-171`): their
stored per-example `score` is **wrong/correct** — the exact reciprocal of our
R = correct/wrong, with identical aggregation on both sides (probs are exp of
mean logprobs; "correct" = paraphrased answer, mean over perturbed answers). So
`theirs = 1/ours`, deterministic. The full-model self-test (our TRs vs their
published full-model log, same model, expect KS p ~ 1) will confirm empirically.
Published logs cover **forget01, forget05 and forget10** for full and all three
retain references — everything the matrix needs.

**Getting their evaluator running on Llama — two more environment items:**

1. Their `tokenizer_args` points at gated `meta-llama/Llama-3.2-1B-Instruct`
   (403). The open-unlearning checkpoint ships identical tokenizer files, so
   the fix is `model.tokenizer_args.pretrained_model_name_or_path=
   open-unlearning/tofu_Llama-3.2-1B-Instruct_full` — no license acceptance
   needed, exactly as LOCAL's gating sweep predicted.
2. `TypeError: Got unsupported ScalarType BFloat16` in `evaluate_probability`
   (`utils.py:98`) — their config evals in bf16 and a `.numpy()` on bf16
   breaks under our torch. Overrode `model.model_args.torch_dtype=float32`
   rather than patching their code. **DEVIATION 5:** published eval dtype is
   bf16, ours runs fp32 — noise-level, also makes the comparison like-for-like
   with our fp32 evaluator, but recorded.

**Their evaluator validated against their published numbers.** My run
(sdpa + fp32 + checkpoint tokenizer) vs `open-unlearning/eval` published
full-model forget01:

    forget_Q_A_Prob   0.9020 vs 0.9012   (0.09%)
    model_utility     0.5981 vs 0.5992   (0.18%)
    forget_Q_A_ROUGE  0.8537 vs 0.8731   (2.2% -- greedy generation is
                      bf16-vs-fp32 sensitive; logprob metrics are not)

So deviations 4/5 (sdpa, fp32) do not distort the reference. Also useful:
published full-model FQ on forget01 is **0.0068** — correctly *failing*, since
the full model knows the forget set.

**Template bug found by the diff, fixed.** First chat-template run gave forget
prob 0.8547 vs their 0.9020. Cause: my hand-rolled Llama-3 tags omitted the
date header (`Cutting Knowledge Date: ... / Today Date: 10 Apr 2025`) that the
tokenizer's Jinja template inserts — one missing header line, 5% prob shift.
t15 now renders through `tokenizer.apply_chat_template` with their
`date_string`, and the answer span mirrors `preprocess_chat_instance` exactly
(closing `<|eot_id|>` included — required for the FQ self-test to be
like-for-like). Also fixed a transformers 4.x/5.x API difference where
`apply_chat_template(tokenize=True)` returns a BatchEncoding in 5.x and
slicing it yielded an empty answer span.

---

## CHAT-TEMPLATE P2 — CLOSED. FQ SELF-TEST — PASS.

v2 (canonical `apply_chat_template` + date header) vs their evaluator, both on
`tofu_Llama-3.2-1B-Instruct_full` / forget01:

    metric                   ours   theirs    rel%
    forget truth ratio     0.4726   0.4731     0.1
    model utility          0.5979   0.5981     0.0
    forget Q-A prob        0.8953   0.9020     0.7
    forget Q-A ROUGE       0.8320   0.8537     2.5   (bf16/fp32 generation
                                                      band; their own rerun
                                                      differs from their
                                                      published number by 2.2%)

**FQ convention self-test: KS p = 1.000000** (stat 0.05, n=40 both sides,
means 0.5070 vs 0.5029). Our full-model truth ratios are distributionally
identical to their published log under the `theirs = 1/ours` transform — the
transform and the scored span are right, so forget quality computed against
the published retain logs (`t21_fq_published.py fq`) is trustworthy. The
selftest exits nonzero on failure, so runners gate on it mechanically.

Tooling landed for the matrix: `t20_llama_ours.py` (our method on Llama-1B,
t14 losses verbatim, chat-template batches, steps 150/750/1500 per split at
batch 4) and `t21_fq_published.py` (FQ vs published logs + mandatory
selftest).

**In flight:** Llama leakage floor — retain95 evaluated on forget05 under the
frozen headline protocol (llama3 template, 64 new tokens, LCS scorer). Its
forget-set gen-ROUGE is the forget05 floor; retain99/retain90 get the same
treatment before forget01/10 are scored.

**forget05 floor measured** (`reports/remote/floor_retain95_f05.json`):
retain95 on forget05 under the frozen headline protocol -> forget gen-ROUGE
**0.3505**, forget prob 0.1296, TR-med 1.26 (~1 = never-knew, as it must be),
utility 0.521. Strikingly close to the Pythia floor (0.364) despite different
model family, template and scorer.

## PRE-REGISTRATION FROZEN — commit `96f8fec`

Both gates passed first (chat-template P2: TR 0.1%/utility 0.0%; FQ self-test
KS p=1.000). The freeze commit predates every forget05 selection datum and any
forget01/10 scoring, and is publicly timestamped on the repo.

**SWEEP RUNNING** (`experiments/t20_run_sweep.sh`): 2 scopes x 4 gammas x
3 seeds = 24 cells on forget05, sequential + resumable, per cell:
train (750 steps) -> eval (frozen protocol) -> FQ vs published retain95 log ->
HF push -> summary row to `reports/remote/t20_forget05_sweep.jsonl` + git push.
First cell reached step 350/750 within minutes; forget_acc already 0.00 at
gamma 0.5 (margins driven negative), retain anchors active.

**Next:** sweep completes -> apply the frozen selection rule -> baselines
(GA/NPO/SimNPO/RMU at open-unlearning configs, effective batch 32) ->
forget01/10 scoring.

## 2026-08-12 (cont.) — step calibration

**Deep grid is too deep everywhere for all-token.** First snapshot
(all-token gamma2, step **150** = 3 epochs over forget05, post-eot-fix):
forget prob 0.0073, gen-ROUGE 0.0246, FQ p=0.000649 FAIL, mean TR (their
convention) 2.41 vs reference 0.97. Already far past the reference at the
shallowest planned depth — **Llama-1B forgets roughly an order of magnitude
faster than Pythia-410m under the same protocol.** The FQ crossing (TR moving
0.47 -> ~0.97) happens somewhere in steps 0-150.

Shallow calibration launched (all-token gamma2, snapshots {25,50,75,100,125},
150 total), running concurrently with the deep-snapshot evals. The deep
snapshots (300-750) are still being evaluated for the record — the
"deeper is monotonically worse" claim gets verified, not assumed; my P2 track
record says assumptions about this pipeline lose.

**Bookkeeping caution:** the shallow run reuses the final dir
`t20_forget05_all_g2_s0`, whose 750-step contents were preserved first as
`_step750`. The pending loop-1 eval tagged `calib_t20_forget05_all_g2_s0`
will therefore actually measure the 150-step model — that tag is
**depth-ambiguous and excluded** from the calibration table; `_step750`
stands in for the deep end if needed.

### FQ-vs-depth curve, and a threshold question I am NOT resolving by amendment

Scorer-independent (forget quality is logprob-based), all-token gamma2:

    step   mean TR (ours)      KS p      ref mean 0.9741
      25          0.6219    0.000000
      50          0.6748    0.000118
     150          2.4075    0.000649
     300     8841355.16     0.000000
     450        1429.35     0.000649

Mean TR crosses the reference between steps 50 and 150; snapshots at 75/100/125
are evaluating. Deep steps blow up (TR 1e6+) — over-forgetting drives correct-
answer probability to ~0, so wrong/correct diverges. Deeper is confirmed worse,
as suspected but now measured.

**Context for the p-values.** Published FQ for the *full* model on forget05 is
**1.43e-12**; our step-50 point is **1.18e-4**, about eight orders of magnitude
better. So the method moves forget quality enormously — it just does not reach
p > 0.05.

**The threshold may be miscalibrated, and that is a trap I am declining to walk
into.** LOCAL's Pythia result "all-token gamma in [0.5,2] passes FQ" was on
**forget01, n=40** — they explicitly called it "the n=40 ceiling". KS power
scales with n, so p > 0.05 at **n=200** (forget05) is a far harder bar, and the
0.05 admissibility rule in the prereg inherited its plausibility from the n=40
regime. open-unlearning's leaderboard for this architecture lists only
endpoints (Finetuned 3.91e-22, Retain 1.0) and **no method rows**, so there is
no published evidence about what is achievable at n=200 either way.

It would be easy, and wrong, to amend the selection rule now — I have seen FQ
at exactly one grid point (all-token, gamma2) and changing the admissibility
criterion on that basis is precisely the post-hoc rule-fitting the
pre-registration exists to prevent. So:

- **The selection rule stands unchanged.** p > 0.05 admissibility, then minimum
  leakage, as frozen.
- **Step count is chosen by max FQ**, which amendment 1 already declared as a
  calibration knob on the selection split. That is a free choice inside the
  frozen procedure, not a change to it.
- If the grid yields no admissible cell, the prereg's own instruction applies:
  **report that and stop** — and additionally report the full FQ/utility/leakage
  frontier, which is the informative result either way. A null on the
  admissibility gate with an 8-order-of-magnitude FQ improvement is a finding,
  not a failure, and it is the reviewers' call whether p > 0.05 at n=200 is the
  right bar.

### Amendment 2 independently validated by the published utility

Re-measuring retain95 under `rouge_score` reproduces open-unlearning's
**published** model utility for that checkpoint:

    scorer        our retain95 utility   published   error
    rouge_score                 0.5961      0.5991    0.5%
    our LCS                     0.5210      0.5991     13%

The short-reference sets move exactly as the punctuation diagnosis predicted:
real_authors ROUGE 0.26 -> 0.8545, world_facts 0.33 -> 0.8462. This is
independent confirmation that the LCS scorer was the defect and `rouge_score`
is correct — the published number was never used to tune anything, so
reproducing it to 0.5% is a genuine out-of-sample check.

**Headline forget05 leakage floor is therefore 0.3950** (retain95, rouge_score,
llama3 template, 64 new tokens). The LCS-measured 0.3505 is void for headline
use, as amendment 2 declared.

### FQ curve, continued — the threshold looks reachable after all

    step   mean TR      KS p        (ref 0.9741)
      25    0.6219   0.000000
      50    0.6748   0.000118
      75    0.7387   0.011843
     150    2.4075   0.000649

p rises steeply as mean TR approaches the reference **from below**, and steps
100/125 sit exactly where the crossing should be. So p > 0.05 at n=200 may well
be achievable for this method, and the decision last heartbeat not to amend the
admissibility rule on one grid point looks correct — had I relaxed the
threshold then, I would have permanently weakened the headline claim to work
around a problem that was about to solve itself two calibration points later.

### all-token step calibration complete — interior optimum at step 100

    step   mean TR     FQ p       (reference 0.9741)
      25    0.6219   0.000000
      50    0.6748   0.000118
      75    0.7387   0.011843
     100    0.8536   0.022092   <- selected (argmax FQ)
     125    1.0472   0.000184
     150    2.4075   0.000649
     300    8.84e+06 0.000000
     450    1429.35  0.000649
     600     68.1017 0.000000

The optimum is **interior**, so the scan brackets it properly rather than
running into the edge of what was tried.

**A result worth stating carefully:** at step 125 the mean truth ratio (1.0472)
is *closer* to the reference (0.9741) than step 100's 0.8536, yet FQ collapses
from 0.0221 to 0.000184. The KS test is rejecting on **distribution shape**, not
location — so "move the mean onto the reference" is not the same objective as
"pass forget quality", and any tuning that targets the mean will miss. This also
means the admissibility bar is not simply a matter of training longer or
shorter: at n=200 the whole TR distribution has to match, and our unlearned
distribution has a different shape from the retain model's regardless of where
its centre sits.

Best all-token FQ is therefore **0.0221 — below the 0.05 threshold.** Per the
prereg that is reported, not engineered around. Whether any (gamma, scope) cell
clears 0.05 is now an empirical question the grid answers.

### min-token step calibration — a ceiling, not a slope

    min step   mean TR     FQ p     (reference 0.9741)
        150     0.6348   0.000000
        300     0.6863   0.000431
        450     0.7045   0.008539   <- selected (argmax FQ, interior)
        600     0.6916   0.002083

**min-token plateaus at mean TR ~0.70 and never approaches the reference**, even
at 4x the depth where all-token had already overshot to 2.41. This is not slow
convergence; it is a ceiling. Pinning only the weakest token per sequence leaves
the remaining answer tokens near their original probabilities, so the truth
ratio cannot be driven onto the retain distribution however long training runs.
Depth is not the free parameter for min-token that it is for all-token.

That makes amendment 3 (per-scope step counts) necessary rather than merely
fair: with a shared count, min-token's result would have been an artifact of
depth. With per-scope counts, min-token gets its best achievable FQ and the
comparison is about **scope**, which is the question.

### GRID LAUNCHED — 24 cells

    all-token cells: 100 steps    min-token cells: 450 steps
    gamma in {0.5, 1, 2, 4} x scope in {all, min} x seed in {0, 1, 2}

Per cell: train -> eval (llama3 template, 64 new tokens, rouge_score,
truncate) -> FQ vs the published retain95 log -> HF push -> summary row +
git push. Resumable and skip-if-done; a rental recycle costs at most one cell.

Calibration cost for the record: ~7 GPU-hours across 2 scopes x 9 depths, plus
the rescoring forced by amendment 2. That is more than budgeted, and all of it
bought corrections that would otherwise have silently biased the matrix.

### Two grid-runner defects caught in the first cells

**1. Stale-eval contamination (data integrity).** The runner correctly
*retrained* `all_g0.5_s0`, but its eval skip-if-done matched only the tag —
so the pre-amendment record (`rouge_impl: lcs`, from the pre-eot-fix 750-step
model) was reused as the metrics for the freshly retrained 100-step
checkpoint. Stale numbers silently attached to a new model, in a summary row
that looked entirely normal. I had verified training was not reusing stale
checkpoints and did not think to check the eval path.

Fixed: purged the stale record and its truth-ratio file, deduped the summary,
and the guard now requires `"rouge_impl": "rouge_score"` as well as the tag,
so no future protocol change can be inherited silently. The cell re-evaluates
on the next resumable pass.

**2. GPU idle during checkpoint upload (throughput).** The HF push ran
synchronously inside the cell loop: GPU measured at **0% / 2 MiB** for the
10-15 minutes each 2.4 GB upload took. Across 24 cells that is ~4-6 GPU-hours
of rented hardware spent waiting on network. Pushes now run in a background
subshell under `flock` — training proceeds immediately, uploads queue among
themselves rather than running 24-wide, and a final `wait` lets in-flight
transfers land before the run reports complete. Verified: GPU at **97%** with
an upload running concurrently.

Also seen once: `git pull` failed with `server certificate verification
failed` (transient TLS); the subsequent push succeeded and local/origin were
verified identical. Worth knowing the rental's network is not perfectly
reliable — the per-cell git push is the mechanism that makes results survive a
recycle, so a persistent failure there matters more than it looks.

### Grid: seed-0 all-token complete (5/24 cells, zero failures)

    cell            FQ p      leak     util    (floor 0.3950 / util 0.5961)
    all g0.5     0.000967   0.2520   0.4159
    all g1       0.002083   0.1847   0.4388
    all g2       0.022092   0.0735   0.4803   <- dominates g0.5/g1 on ALL three
    all g4       0.002083   0.0466   0.4772
    min g0.5     0.003010   0.3816   0.4397

Three observations at n=1 seed (to be confirmed across seeds, not concluded):

1. **gamma2 dominates gamma0.5 and gamma1 on every metric simultaneously** —
   better FQ, lower leakage, higher utility. If that ordering survives seeds,
   the "milder is safer" intuition is wrong on Llama: too-shallow margins leave
   the model in a distributionally weirder state than a deeper, cleaner push.
2. FQ peaks interior in gamma (0.0221 at gamma2, falling to 0.0021 at gamma4),
   echoing the interior peak in depth. Leakage, by contrast, is monotone in
   gamma. The two metrics genuinely decouple.
3. **min-token gamma0.5 leakage (0.3816) sits essentially AT the floor
   (0.3950)** — min-token barely suppresses generation below never-knew level,
   while all-token goes far below. Combined with min's TR ceiling (~0.70), the
   scope comparison is shaping up to be: all-token = deep suppression on both
   axes; min-token = mild on both, capped.

No cell clears FQ 0.05 yet. gamma2's 0.0221 matches the calibration cell
exactly (same config), which is a good reproducibility sign for the harness.

Pace: all-token ~7 min/cell, min-token ~16 min/cell (450 steps), zero FATALs,
uploads running concurrently. Remaining ~19 cells ~= 3.9 h.

### Seed 0 complete (8/8): first admissible cell, and the metrics agree with each other

    cell          FQ p       leak     util    admissible
    all g0.5    0.000967   0.2520   0.4159
    all g1      0.002083   0.1847   0.4388
    all g2      0.022092   0.0735   0.4803
    all g4      0.002083   0.0466   0.4772
    min g0.5    0.003010   0.3816   0.4397
    min g1      0.000184   0.3726   0.4404
    min g2      0.008539   0.3614   0.4329
    min g4      0.177934   0.3844   0.4517   YES     (floor 0.3950 / util 0.5961)

**min-token gamma4 clears the bar at p=0.178** — the only cell that does.

The coherence check: the one cell whose truth-ratio distribution is
statistically indistinguishable from the retain reference (FQ pass) is also
the one whose generation leakage (0.3844) sits essentially AT the reference's
own leakage (0.3950). Both metrics independently say the same thing: min_g4
*resembles a never-knew model* rather than an aggressively-suppressed one.
All-token cells drive leakage far below the floor — visible over-suppression —
and fail FQ; min-token's per-sequence ceiling turns out to be a feature at
high gamma: enough depth to match the reference, structurally prevented from
overshooting it.

Notable inversion vs Pythia (LOCAL t17): there, all-token gamma in [0.5,2]
passed (at n=40) and gamma8 over-forgot; here at n=200 on Llama, all-token
never passes and min-token gamma4 does. The scope story is model- and
n-dependent — which is itself a finding for the writeup.

Also: the gamma2->gamma4 jump inside min scope (0.0085 -> 0.178) is large. The
450-step calibration was done at gamma2 per amendment 3's declared procedure;
it transferred well to gamma4. Luck or robustness, but the procedure was fixed
in advance either way.

Seeds 1-2 decide: admissibility is on MEAN FQ across 3 seeds. If min_g4 holds,
the frozen rule selects it outright and the INADMISSIBLE_FALLBACK path is
never taken. ~2.5 h remain; baselines chained behind.

### min_g4 admissibility is mathematically locked; the SELECTION is not

min_g4 seeds: s0 = 0.1779, s1 = 0.0163 (the 10x drop confirms the seed-noise
finding). Sum = 0.1942 > 0.15, so its **3-seed mean exceeds 0.05 whatever seed
2 produces** — admissibility is decided two-thirds of the way through.

The *selection* is not decided. Every other cell could still go admissible on
a lucky seed 2 (all_g1 needs s2 > 0.126, all_g2 > 0.128 — both within the
observed noise range), and the frozen rule picks the admissible cell with
**minimum leakage**. all_g2's leakage (0.0497) is 7x below min_g4's (0.366),
so if all_g2 sneaks over the bar it wins selection outright and the story
changes from "resembles-the-reference" to "deep-suppressor". The rule is
frozen; whichever way seed 2 falls, it executes as written.

Worth pre-writing the interpretation fork:
- min_g4 alone admissible -> selected config *matches* the never-knew
  reference on both metrics; leakage ~ floor is the honest reading.
- all_g2 also admissible -> the rule prefers the deep suppressor; the paper
  must then be explicit that "minimum leakage among admissible" optimizes
  suppression *given* statistical indistinguishability, and the min_g4 point
  remains the distribution-matching exhibit.

### Grid 23/24 — selection effectively decided

min_g1_s2 (0.0002) and min_g2_s2 (0.0043) landed low: both configs are now
final-inadmissible. **min_g4 is the sole admissible config** (mean locked
> 0.05 since seed 1); its seed-2 cell is the last one training. t24 executes
the formal selection on the complete grid via the armed chain, then baselines
launch automatically.

Final grid means (3 seeds except min_g4 pending its s2):

    all-token: FQ 0.002-0.013 (all inadmissible), leakage 0.03-0.25 (deep
               suppression), utility 0.42-0.48
    min-token: FQ ~0.001-0.005 for gamma<=2 (inadmissible), leakage ~0.37-0.39
               (at floor), utility 0.42-0.46
    min_g4:    FQ mean >= 0.065 whatever s2 brings; leakage ~0.37; utility ~0.45

The final story survived seed averaging: the admissible cell is the one that
RESEMBLES the never-knew reference (leakage at floor) rather than maximally
suppressing; every deep suppressor fails distributional indistinguishability.

## GRID COMPLETE (24/24) — SELECTION: min-token gamma4

Frozen rule (96f8fec + amendments), executed mechanically by t24:

    scope gamma   FQ mean+-rng      leak          util        admissible
    all   0.5    0.0021+-0.0033   0.2431        0.4357        no
    all   1      0.0088+-0.0200   0.1663        0.4272        no
    all   2      0.0128+-0.0221   0.0520        0.4458        no
    all   4      0.0021+-0.0043   0.0349        0.4690        no
    min   0.5    0.0012+-0.0028   0.3944        0.4330        no
    min   1      0.0006+-0.0012   0.3749        0.4384        no
    min   2      0.0046+-0.0076   0.3767        0.4469        no
    min   4      0.0662+-0.1736   0.3448        0.4458        YES  <- SELECTED

Reference: floor 0.3950, retain95 utility 0.5961. Full-model FQ: 1.43e-12.

Honest caveats, recorded with the selection:
- min_g4's FQ mean rides on one high seed (0.178/0.016/0.004): the +-range is
  2.6x the mean. Its admissibility is a *mean over 3 seeds* clearing the bar,
  not a robust per-seed property. The seed-noise finding applies to our own
  headline number and the paper must say so.
- Its seed-2 FQ (0.0043) equals min_g2_s2's exactly -- KS p-values are discrete
  in the test statistic at n=200, so identical values are expected behaviour,
  not a data error (verified distinct leak/util for those cells).

End-of-grid mechanical note: the runner died at the very tail with a bash
lazy-read syntax error -- caused by my own mid-run edits to the script (the
exact hazard avoided for the janitor). All 24 cells' data were already
recorded; only the final HF push + completion marker were lost, both replayed
by hand. Lesson standing: never edit a running bash script, ever, including
"just the tail".

Chain state: marker written -> baselines launch; then t28 headline (min g4,
steps 90/900 for f01/f10 per amendment 4) -> t25 relearn curves.

### Baselines running; their evaluator kept as free cross-validation

GA seed-0 trained (60 steps, their config: 10 epochs at effective batch 32).
Their pipeline then runs open-unlearning's own TOFU evaluator regardless of
`trainer.args.do_eval=False` (that flag governs the HF Trainer loop only) --
the GPU idling at 0% during its CPU ROUGE passes briefly looked like a hang.

Decision: keep their eval phase rather than kill/edit the running runner.
It writes their evaluator's TOFU_EVAL.json alongside every baseline
checkpoint, i.e. a free evaluator-agreement check on all 12 baseline cells
(and editing a running bash script already burned us once tonight). Cost
~2 h across the stage; revised baseline ETA ~5 h.

### Baseline 1: GradAscent — admissible FQ by lobotomy

GA seed-0 (their published config, 10 epochs, effective batch 32):

    FQ p = 0.112  ADMISSIBLE     leakage 0.207     utility 0.034

GA passes forget quality at n=200 — by destroying the model outright
(retain prob 0.0073, final loss -46, grad norm 2064). The KS test cannot
distinguish "matches the never-knew reference" from "outputs garbage whose
truth ratios happen to spread like the reference's". This makes the utility
axis, not FQ alone, the discriminating comparison: ours-selected holds utility
0.446 at admissible FQ; GA holds 0.034.

**Evaluator-disagreement false alarm, resolved by timestamps.** Their in-run
TOFU_SUMMARY (util 0.5952, forget prob 0.82) was written at 06:47-06:49 —
per-epoch evals that fire on `eval_strategy: epoch` regardless of
`do_eval=False` — while the final model saved at 07:06. Their summary
describes an early-epoch model. Consequence: **the in-run eval jsons are
NOT final-model numbers and are useless as cross-validation** — the planned
evaluator-agreement check on baseline cells is dead (P2 already validated
agreement three independent ways, so nothing is actually lost).

### NPO seed-0: the two-axis split emerges

    method (s0)   FQ            util     reading
    GA            0.112  pass   0.034    admissible by lobotomy
    NPO           1.1e-5 fail   0.460    functional, but distinguishable
    ours min_g4   0.066  pass   0.446    both (3-seed mean)

NPO preserves utility at ours' level but its truth-ratio distribution stays
far from the reference. If this holds across seeds and for SimNPO/RMU, the
matrix headline is: **ours is the only method simultaneously admissible and
functional** — GA achieves FQ by destruction, NPO/... achieve utility without
distributional indistinguishability. Exactly the frontier plot's job.

(Seed-0 only for baselines so far; 10 cells remain.)

### SimNPO seed-0: near-no-op at their published config

    fq 0.0 | leak 0.7341 (floor 0.395 -- still reciting) | util 0.595 (= full)
    forget prob 0.848 vs full model's 0.895; mean TR 0.50 vs full's 0.503

At open-unlearning's shipped SimNPO config (beta 4.5, npo_coeff gamma=0.125,
retain NLL) the model is essentially unchanged after 10 epochs. Training loss
plateaued ~1.0. Their npo_coeff of 0.125 scales the forget term to 1/8th --
combined with lr 1e-5 this is plausibly just too gentle for Llama-1B/forget05.

Recorded as-is per the prereg (baselines run at published configs, full stop):
the honest row is "SimNPO at its published TOFU config does not unlearn this
model". Same reporting class as LOCAL's RMU-at-WMDP-defaults finding. No
retuning on our side -- that would be *our* tuning masquerading as their
method. LOCAL's t17 caveat ("check their tuned config before drawing
conclusions") applies verbatim and goes in the paper text.

RMU seed-0 training.

### Seed-0 baselines complete — four methods, four distinct modes

    method   FQ            leak     util    mode
    GA       0.112  PASS   0.207    0.034   lobotomy pass
    NPO      1e-05  fail   0.315    0.460   functional, distinguishable
    SimNPO   0.0    fail   0.734    0.595   did not unlearn (their config)
    RMU      0.0    fail   0.433    0.555   best baseline: real forgetting,
                                            good utility, still p=0
    ours     0.066  PASS   0.345    0.446   admissible + functional
    (ref)    --            0.395    0.596

RMU at their TOFU config is the serious competitor: forget prob 0.895 -> 0.403,
utility within 7% of reference, leakage near floor -- yet its truth-ratio
distribution is unambiguously distinguishable (p = 0.0). The utility-vs-ours
comparison (0.555 vs 0.446) will be the fair-minded reviewer's question; our
answer is the admissibility column, and the honest caveat is finding #3 (our
pass is a noisy-mean pass). Seeds 1-2 now running for all four baselines.

### Seeds 1-2 underway; one transient push failure retried

GA seed-1 training. RMU seed-0's HF push failed on a transient network error
(xet write-token request) -- retried with backoff. Note the janitor's deletion
criterion (API-verified safetensors on HF) is exactly what makes a failed push
safe: the local weights stay until the retry lands, by construction.

## HEADLINE — forget01 complete: admissible on every seed

    seed   FQ p      leak     util     (floor 0.4136, ref util 0.5972)
    s0     0.5786    0.3759   0.5005
    s1     0.0541    0.3036   0.4777
    s2     0.0971    0.4064   0.4964
    mean   0.2433 -- ADMISSIBLE, and per-seed admissible too

Nothing was selected on forget01: gamma/scope came from the frozen forget05
rule, steps from amendment 4's constant-epoch transfer. This is out-of-split
generalization of the pre-registered config -- the strongest form of the
claim the prereg machinery was built to support. Consistent with finding #4,
the n=40 regime is kinder to FQ (mean 0.243 vs forget05's 0.066), and per-seed
passes appear exactly where power drops.

forget10 (n=400, the harshest regime) is the remaining test: floor from
retain90, then 3 cells at 900 steps.

### forget10 seed-0: hard fail (2e-06) — and the mechanism is the ceiling

    forget10 floor (retain90): leak 0.3849, util 0.5905
    s0: FQ 2e-06 | leak 0.2999 | util 0.434 | mean TR 0.7323 vs ref 0.9488

Cross-split FQ for the same pre-registered config, monotone in n:

    forget01 (n=40):   mean 0.243   per-seed pass
    forget05 (n=200):  mean 0.066   noisy-mean pass
    forget10 (n=400):  s0 2e-06     (seeds 1-2 pending, deep fails are stable)

Two forces compound: (1) KS power grows with n, demanding a closer
distributional match; (2) min-token's depth ceiling (~0.73 mean TR here) sits
BELOW forget10's reference (0.9488) — the same ceiling that prevented
overshoot at forget05 becomes the binding constraint at forget10. The ceiling
is a property of the method+model; the reference level varies by split. So the
method's admissibility window is real but bounded: it passes where the
reference is reachable from below at its plateau, and fails where n pushes the
bar past its ceiling.

Honest headline shape: admissible at forget01 and forget05, fails at forget10
with an identified mechanism. That is a *boundary*, reported as one — not
hidden, and considerably more informative than a uniform pass would have been.

### forget10 s1 confirms the boundary (5e-05); final matrix cell training

forget10 seeds so far: 2e-06 / 5e-05 -- the ceiling-vs-reference mechanism
holds across seeds, as deep fails have all campaign. Seed 2 is the last
matrix cell of the campaign; relearn curves chain after it.

## HEADLINE STAGE COMPLETE — the full pre-registered matrix exists

forget10 final: 2e-06 / 5e-05 / 4e-06 — boundary confirmed on all three seeds.

Cross-split, one frozen config (min-token gamma4, constant-epoch steps):

    split      n     FQ per seed              verdict
    forget01   40    0.579 / 0.054 / 0.097    admissible, every seed
    forget05   200   0.178 / 0.016 / 0.004    admissible (mean 0.066)
    forget10   400   2e-6  / 5e-5  / 4e-6     fails; ceiling < reference at n=400

Matrix totals: 24 grid + 12 baseline + 6 headline = 42 cells, 3 floors,
zero training failures. Every checkpoint on HF (one retry pending for the
last cell), every summary row on GitHub.

Relearn curves (selected vs never-knew control, lr 1e-5 and 5e-5) are the
last GPU stage; t25 fetched the selected checkpoint from HF and is running.

### HF private storage limit hit; pruned reproducible checkpoints

The final headline checkpoint push failed with "Private repository storage
limit reached" -- ~42 x 2.4 GB ~= 100 GB, the private-tier cap. My earlier
retry masked the real error behind an echo's exit code (fixed in the retry
loop: branch on the push command itself, never on a trailing echo).

Mitigation under the autonomy grant: **pruned the 21 non-selected ours-grid
cells from HF** (all-token all gammas, min-token gamma<=2, x3 seeds, ~50 GB).
Every pruned cell is deterministically re-trainable (script + config + seed in
git) and all its metrics are recorded in the tracked jsonls. Kept: all 12
baseline cells, the selected config (min_g4) on every split and seed, and the
Phi GA pilot. Final push retrying against the freed space.

**FOR-LOGAN:** if you want the full 42-cell checkpoint archive on HF, the
options are upgrading the HF plan or flipping the repo public (public repos
have effectively no such cap) -- your call, it's money-or-visibility. The
campaign's conclusions do not depend on the pruned checkpoints.

### HF storage: squash GC appears async; final checkpoint parked locally

Pruned 21 cells, squashed history — pushes still hit the cap, consistent with
asynchronous garbage collection on HF's side. The s2 checkpoint is safe
locally (janitor deletes only after verified upload) and its metrics are all
recorded; will retry the push later rather than hammer the API. Standing HF
lesson recorded: **deleting files frees nothing until super_squash_history,
and even then reclamation is not immediate.**

Relearn curves flowing: selected@lr1e-5 at step 20, forget prob 0.32->0.38.
Control curve next; resistance claims wait for the control comparison.

### Relearn lr=1e-5: no resistance — residual knowledge accelerates recovery

    step        selected prob    control prob   (leak similar pattern)
       0            0.197           0.130
      40            0.460           0.307
      80            0.596           0.414
     160            0.782           (running)

The selected config relearns FASTER than the never-knew control at every
matched depth. Distributional indistinguishability (FQ pass) does not imply
the knowledge is gone -- it is latently present and reconstructs more easily
than learning from scratch. This is the honest completion of the story:
admissible + functional, but NOT relearn-resistant, consistent with LOCAL's
t18 (relearning is the hardest axis; first-order defenses fail under AdamW).
For the paper: FQ measures indistinguishability at rest, not under attack --
a third axis, and no tested method holds it.

## CAMPAIGN GPU PROGRAM COMPLETE

Relearn, both lrs, final (forget prob at matched steps, selected vs control):

    lr 1e-5:  step 80: 0.60 vs 0.41   step 160: 0.78 vs 0.67
    lr 5e-5:  step 80: 0.59 vs 0.54   step 160: 0.75 vs 0.70

Selected relearns faster than never-knew at BOTH lrs. The gap is larger at the
lower lr (residual structure's head start matters most when learning is slow);
at 5e-5 everything learns fast and the curves converge -- t18's lr-fragility
reproduced on Llama. No relearn resistance; FQ is an at-rest property.

Totals: 42 matrix cells + 3 floors + 4 relearn curves + 2 calibration series,
zero unrecovered failures. Artifacts: RESULTS.md, FINDINGS.md, PREREGISTRATION
(+4 amendments), fig_frontier, fig_relearn, per-cell jsonls, ~21 checkpoints
on HF (one parked local pending HF storage reclamation).

### Post-campaign: babysit mode

GPU idle, all stages complete. Heartbeat throttled from 23 min to hourly
(mailbox babysit + parked-push retry only; no new experiments). Parked push
still capped after the GC window — genuinely FOR-LOGAN now (plan upgrade /
public repo / accept local-only).

**Cost note for Logan:** the rental is now idle but billing. `vastai stop
instance` halts GPU charges (storage still billed; /workspace is NOT a volume,
so a later recycle/destroy loses only what isn't on git/HF — which at this
point is nothing scientific except the one parked checkpoint). Stopping is
Logan's call, not mine — the 3B/8B extension decision determines whether this
box is still needed.

## POST-HOC PHASE (labeled; separate from the pre-registered matrix)

Logan's questions after reading the figures drove three follow-ups:

1. **Utility decomposition** (fig_utility_components): our deficit is entirely
   retain-set collateral. Cause: the t14-inherited `[:400]` retain anchor cap
   -- designed as 10:1 retain:forget coverage for Pythia/forget01, silently
   degraded to 2:1 / ~10% coverage at forget05. Finding: the pin's suppression
   generalizes across the TOFU distribution, so the retain bundle needs
   coverage of the interference domain, not just examples. Full-retain is
   parity, not cheating -- the baselines all train against the full split.
2. **ours-v2 full-retain** (3 seeds, running): tests the diagnosis. Reported
   post-hoc; FQ/leakage re-verified, never assumed.
3. **Pareto sweeps** (queued behind v2): every method gets its
   utility<->forgetting knob traced with the same eval battery --
   NPO lr {1e-5*,2e-5,5e-5}, GA epochs {2,5,10*}, RMU steering {2*,5,20},
   SimNPO forget-scaling {0.125*,1.0}, ours gamma {0.5..4} (* = published).
   3 seeds each. Pareto checkpoints are metrics-only (weights deleted after
   eval; HF capped, cells deterministic).

Framing note for the paper, from Logan's over-forgetting question: TOFU's FQ
is auditor-indistinguishability, under which over-forgetting is a (Streisand)
failure; under a content-removal threat model over-forgetting is acceptable
and all-token gamma4 (leak 0.035, util 0.469) is the matrix's best cell. The
Pareto figure should carry both frontiers.

### v2 s0 result: full-retain-same-budget is WORSE — coverage without pressure fails

    v2 s0 (full retain):  util 0.414  retain_prob 0.252  FQ 0.003
    v1 s0 ([:400]):       util 0.452  retain_prob ~0.33  FQ 0.178

Spreading the same 450x4 retain samples over 3,800 rows instead of 400 drops
per-row anchoring from ~4.5 visits to ~0.5 -- broad but too shallow to hold
against the pin. (Possible second factor: the hinge target cap = median margin
over the full split, which may sit lower than the 400-row median, weakening
targets.) The interference diagnosis stands; the repair needs coverage AND
pressure -- larger retain batch, higher retain weight, or proportionally more
steps. Seeds 1-2 will confirm; if they do, the v3 shape is retain batch 4->16
at unchanged forget batch. Babysit turn: logged, not launched.

### v2 confirmed worse; tuned NPO s0 may flip the leaderboard

v2 (full retain, same budget), all seeds: FQ {0.003, 0.221, 0.022} mean 0.082
(admissible) but utility 0.416 and retain_prob 0.256 -- strictly worse than
v1's 0.446/0.327. Coverage-without-pressure is settled: worse on 3/3 seeds.

**Tuned NPO lr 2e-5, seed 0: FQ 0.7126, utility 0.533, retain_prob 0.611,
leakage 0.299.** A deep FQ pass at utility well above ours. If seeds 1-2
confirm, NPO with a fair tuning budget DOMINATES our selected config on every
axis, and the paper's comparative claim must be rewritten accordingly: the
honest headline becomes the mechanism findings (decoupling, ceiling,
KS-shape, seed-noise, lobotomy-pass) rather than a leaderboard win. Logged
before the confirming seeds land so the reading is not fitted to hope.

## LEADERBOARD FLIP CONFIRMED — tuned NPO dominates

NPO at lr 2e-5 (2x published), 3/3 seeds:

    fq {0.7126, 0.3935, 0.7934} mean 0.633 -- deep pass on EVERY seed
    utility 0.538 (ours 0.446)   leakage ~0.29 (floor 0.395)

The published-config NPO (lr 1e-5, fq ~1.5e-5) was simply under-trained on
this model; one doubling of lr moves it from hopeless to the best cell on the
board. Implications, recorded plainly:

1. The pre-registered matrix's comparative claim ("ours is the only
   admissible-and-functional method") was TRUE AT PUBLISHED CONFIGS and is
   FALSE under an equal tuning budget. Both statements go in the paper, in
   that order, with this table.
2. The methodological findings are untouched and are now clearly the paper's
   core: lobotomy-pass, KS shape-vs-location, threshold-local seed noise,
   n-scaling, the min-token ceiling, prob/gen decoupling, no relearn
   resistance, and the evaluator/protocol results.
3. It also sharpens a benchmark critique: leaderboard configs are so
   lr-sensitive that "method X beats method Y" claims at fixed configs are
   nearly meaningless -- the fair object of comparison is the tuning-budgeted
   Pareto frontier, which is what the running sweep produces.
4. Open for the writeup: tuned-NPO's relearn resistance (expect none) and
   cross-split behavior (n=400 may still be hard for it -- its TR
   distribution shape is untested at forget10).

Full Pareto figure lands when GA/RMU/SimNPO points finish.

### NPO 5e-5 arm also passes — the tuning curve is single-peaked

    lr 1e-5 (published): fq ~1.5e-5, util 0.461   -- hopeless
    lr 2e-5:             fq 0.633,   util 0.538   -- optimum so far
    lr 5e-5 (2 seeds):   fq ~0.29,   util ~0.52   -- modest decline

The published config sits at the dead end of its own method's tuning curve.

### Post-hoc batch 1 complete; NPO curve final

    lr 1e-5*: mean fq 1.5e-5             util 0.461
    lr 2e-5 : mean fq 0.633 (per-seed!)  util 0.538   <- optimum
    lr 5e-5 : mean fq 0.194 {.11,.47,.004} util 0.517

Note 5e-5's seed triple: threshold-local seed noise reproduces in a tuned
baseline exactly as it did in ours -- further evidence the phenomenon is a
property of the KS-at-n=200 metric, not of any method. GA/RMU/SimNPO Pareto
sweep released by the completion marker.

## 2026-08-12 20:56 UTC — HF storage cap resolved: repo flipped PUBLIC per Logan

Logan green-lit making the checkpoint repo public. Flipped
`Elriggs/memorization-unlearning` via `update_repo_settings(private=False)`;
the parked `t28_forget10_min_g4.0_s2` push then succeeded on first retry.
**Every campaign checkpoint is now on HF.** Hourly retry becomes a no-op
(PUSHED.json skip guard). FOR-LOGAN storage item closed.

Pareto sweep progress at this point: seed-0 block 4/5 done (ga_2ep, ga_5ep,
rmu_sc5, rmu_sc20 all exit=0), simnpo_g1_s0 just started. ~25–30 min/cell.

## 2026-08-13 02:3x UTC — PARETO SWEEP COMPLETE: no tuning rescues GA/RMU/SimNPO

POSTHOC2 marker written; 15/15 cells, 45 stage-exits, zero failures. The
runner captured exit codes only, so FQ p-values were recomputed from the
stored truth-ratio jsons (deterministic KS; sink: t23p_pareto_forget05.jsonl).

3-seed means (fq / util / leak):
- GA 2ep:      1e-13 / 0.587 / 0.668   (functional, still reciting)
- GA 5ep:      4e-12 / 0.545 / 0.472
- RMU sc5:     5e-11 / 0.542 / 0.420
- RMU sc20:    0.18{3e-4,3e-3,.55} / 0.250 / 0.261  (threshold noise again)
- SimNPO γ1:   1e-10 / 0.592 / 0.514   (still near-no-op)
- NPO 2e-5:    0.633 {.71,.39,.79} / 0.538 / 0.286  ← the only tuned point
  reaching the admissible+functional corner; dominates ours (0.066/0.446).

Conclusion: the corner discriminates. GA never matches the reference
distribution while functional (passes only via lobotomy at 10ep/util 0.02);
RMU collapses utility before reaching admissibility; SimNPO barely moves.
Only NPO and ours get there under equal budget, and NPO wins.

Artifacts: t30_pareto.py → fig_pareto_forget05.png/svg (two panels: auditor
FQ-vs-utility + content-removal leakage-vs-utility with never-knew floor),
PARETO.md (full table with per-seed FQ). FINDINGS.md: headline rewritten to
the two-claim ordering; findings 13–15 added.

## 2026-08-13 15:0x UTC — t31: FQ anatomy + utility decomposition with tuned NPO (Logan Qs)

Analysis only, no training. t31_fq_anatomy.py renders from stored evals/TRs.

1. **v2 full-retain hurt, quantified per component:** utility 0.447→0.417
   (worse 3/3 seeds); retain/prob 0.327→0.256, retain/rouge 0.454→0.389.
   Coverage without pressure: 450 steps over 3,800 rows = ~0.5 visits/row.
2. **Where tuned NPO keeps utility ours loses:** almost entirely retain/prob
   (0.62 vs our 0.33; ref 0.87) and retain/rouge (0.45 vs 0.45→ ours v1 equal
   there) — see fig_utility_components_v2. real_authors/world_facts are
   comparable across all functional methods.
3. **FQ anatomy (fig_fq_anatomy):** KS D vs retain95 reference, per seed:
   ours {0.110,0.155,0.175} D̄=0.147; NPO 2e-5 {0.070,0.090,0.065} D̄=0.075;
   GA 10ep {0.120,0.125,0.135} D̄=0.127. n=200 admissibility bar ≈ D 0.136.
   Ours straddles the bar (hence threshold seed-noise); NPO is well inside.
   Mechanism visible in the ECDF: min-token pin drives every sequence to a
   similar depth → under-dispersed TR distribution missing the reference's
   BOTH tails (late start ~0.25, early saturation ~1.3). NPO's distribution
   matches including tails. GA's "pass" is a destruction coincidence — KS
   sees only forget-set TRs; the utility axis catches the dead model.
4. **Published-config discrepancy noted for the paper:** their repro table
   reports NPO forget05 FQ 0.14 (single seed, 2xL40s DeepSpeed); our 3-seed
   single-GPU rerun of the same config: p≈1.6e-5. Consistent with their own
   caveat ("results may vary... single GPU") and finding 3. Their docs also
   flag the NPO-implementation inconsistency carried into SimNPO's codebase
   (their note 3) — relevant to the "NPO is bad" literature narrative.

## 2026-08-13 16:3x UTC — t32: retain damage is GLOBAL, not a coverage problem

Logan Qs continued. t32_retain_examples.py (inference only; selected min-γ4
seed0 re-fetched from HF) vs full model, TOFU prob metric on retain95:
  anchored rows [0:400]:    0.867 -> 0.315
  unanchored rows [400:800]: 0.885 -> 0.283
Anchoring bought only ~0.03 locally — the collateral spreads through shared
parameters. This kills the "we only anchored 400 of 3,800" coverage
hypothesis properly (v2's failure already hinted): the binding problem is
the retain OBJECTIVE, not retain coverage. The margin hinge restores rank
margins and the KL preserves distribution shape; neither pushes absolute
gold-answer probability mass back. NPO's retain term is plain CE (NLL) —
it directly maximizes the exact quantity that is utility's weakest
component. Mechanical explanation of the 0.62-vs-0.33 retain/prob gap, and
the obvious v3: add a retain CE (or absolute-logprob pin) term.

Qualitative (in t32 log): our retain generations keep the GIST (genres,
motifs, award classes) but lose verbatim phrasing and confabulate entity
specifics (book titles, award names). "Forgetting retain facts" = verbatim/
confidence stripping + edge confabulation, not blank-out.

Compute (trainer_state + log mtimes, forget05): baselines 60 opt steps x
eff.batch 32 = 1,920 forget visits (~10 epochs), 20-22 min. Ours min: 450 x
4 = 1,800 visits (~9 epochs), ~40 min (batch 4, grad-ckpt, 3 fwd/step —
implementation, not method). Ours all: 400 visits (~2 ep), ~9 min. GA-2ep:
384 visits, ~4 min. Sample budgets between ours-min and 10-epoch baselines
are essentially matched.

## 2026-08-13 17:3x UTC — OVERNIGHT CHAIN LAUNCHED: v3 objectives, RRS metric, realistic-benchmark design

Logan approved all three follow-ups and went to sleep ("go ahead with your
plans"). Chain (t33_run.sh, background, marker RELEARN2 COMPLETE):
- Phase A: t33 v3 variants on the selected config (min g4, 450 steps,
  cap 400): "ce" (NPO's retain CE bolted on) and "lppin" (margin hinge
  replaced by per-token absolute log-prob restoration vs reference —
  restores prob mass without CE's overshoot incentive). 2x3 seeds,
  eval+fq+push per cell, summary reports/remote/t33_v3_forget05.jsonl.
- Phase B: retrain t20 all_g4_s0 (weights were lost to the storage-cap HF
  prune + local janitor; 100 steps deterministic) + re-eval to verify
  against the recorded cell.
- Phase C: relearn curves (t25, lr {1e-5,5e-5}) for all_g4_s0 (does deep
  suppression resist attack?), npo_lr2e-05_s0 (does the new champion?),
  and the v3 winner (admissible-max-utility rule, fallback max-fq).
- Phase D: t34_rrs.py — RRS := mean_t[control_rouge(t) - subject_rouge(t)],
  headline = min over lrs; RRS.md + fig_relearn_all + t34_rrs.json.
Smoke-tested both t33 variants at 3 steps before launch (exit 0 both).
Also wrote REALISTIC.md: RWKU-pilot design for real-world-knowledge
unlearning with paraphrase + neighbor probes (Logan's third request);
non-blocking defaults declared, pilot NOT launched (next session's call).

## 2026-08-13 04:00 UTC — CORRECTION to the compute comparison (t32 entry)

The "ours min ~40 min wall-clock" figure was wrong — it came from mtime gaps
in the interleaved grid, which bundled other cells' evals into the gap.
Ground truth from the t33 run (identical trainer): 450 steps ≈ 2 min
(safetensors written 110 s after launch). So: ours-min ≈ 2-3 min train,
ours-all ≈ 30 s, baselines ≈ 20 min (60 steps × 8 accum fwd/bwd at batch 4
under their HF trainer). Sample budgets remain matched (~1,800 vs 1,920
forget visits); wall-clock ours is ~10x cheaper, not slower.

## 2026-08-13 05:0x UTC — RWKU PILOT QUEUED (Logan: "do queue up the #4 realistic benchmark")

t35_rwku.py + t35_run.sh launched in background; blocks on the overnight
chain's RELEARN2 COMPLETE marker, then runs:
  basecheck: rank all 200 RWKU targets by base Llama-3.2-1B knowledge
    (level-2 QA ROUGE, 10 probes each) -> pilot = top-10 known targets
  baseeval x10: base-model reference rows (full probe suite, <=40/level)
  {ga, npo, ours} x 10 targets: unlearn on the target's wiki passages
    (2 epochs, batch 4; ga lr1e-5 ascent / npo lr2e-5 beta0.1 / ours
    all-token pin gamma2 + KL anchor on non-pilot targets' passages —
    extra-data use DECLARED), eval forget l1/l2/l3 + neighbor l1/l2,
    then delete weights (30 x 2.4GB would fill the disk).
Base model: unsloth/Llama-3.2-1B-Instruct mirror (meta-llama gated, same
weights) — NOT the TOFU fine-tune; this is pretrained knowledge.
Protocol stamps: greedy max_new 32, rouge_score, probes capped at 40/level.
Pilot config is DECLARED-NOT-TUNED; findings will note lr sensitivity per
the t23 lesson. Summary: results/t35_rwku.jsonl -> reports mirror.
Marker: RWKU PILOT COMPLETE in results/t20_logs/rwku.log.

## 2026-08-13 09:0x UTC — OVERNIGHT CHAIN COMPLETE: v3 verdict + RRS verdict

Zero failed exits across all phases. RELEARN2 COMPLETE written; RWKU pilot
auto-released.

**v3 (retain objective fix), 3-seed means:**
- lppin: utility 0.578 (v1: 0.447; reference 0.596), retain_prob 0.857
  (reference 0.87) — THE UTILITY DEFICIT IS SOLVED by absolute-logprob
  restoration. FQ mean 0.0055 {2.8e-4, 0.016, 1.8e-5}.
- ce: utility 0.522, retain_prob 0.614, FQ mean 0.0098.
- v1 comparison: FQ mean 0.066 was carried by one seed (0.178); all six v3
  seeds sit below 0.017. Reading: restoring absolute prob mass pulls the
  model toward the FULL model's distribution, and the forget-set TR
  distribution moves with it — utility and FQ trade off through the same
  KL/lp anchor. The frontier point moves right and down.

**RRS (relearn resistance, headline = min over lrs, ROUGE):**
  NPO 2e-5     -0.013   (least bad; +0.004 at lr 1e-5, ~neutral)
  selected v1  -0.043
  v3-ce        -0.048
  all-token g4 -0.067   (WORST)
Every subject negative: nothing tested resists relearning. And the ranking
INVERTS the suppression-depth hypothesis: the deepest at-rest suppressor
(all-token, leak 0.047) relearns FASTEST — 0.04 -> 0.89 ROUGE in 160 steps
at lr 1e-5, ending 0.22 ABOVE the never-knew control. Margin-pinned weights
apparently sit in a reflection of the original basin: one gradient step
back. At lr 5e-5 all curves converge onto the control (attack saturates).
Finding: at-rest suppression depth ANTI-correlates with relearn resistance.

all_g4_s0 retrain reproduced the recorded cell (leak 0.047 vs 0.035, util
0.477 vs 0.469, fq 0.0021 vs 0.0021 — nondeterminism-level drift, same
character). Artifacts: t34_rrs.json, RRS.md, fig_relearn_all.

## 2026-08-13 13:0x UTC — RWKU PILOT COMPLETE: methods keep their personalities on real knowledge

71/71 stage-exits zero. 10 targets (Morrison, FDR, Aristotle, Beyoncé, ...),
base-normalized means (t35_rwku.jsonl; fig_rwku_pilot):
            forget-removed%  neighbor-kept%  adv-probe-kept%
  GA              93.5             7.5             6.9    (lobotomy, again)
  NPO             59.9            75.1            46.6    (gentlest)
  ours            86.5            55.8            22.2    (deepest functional)
Probe-level: NPO's forgetting deficit is concentrated in CLOZE probes (83%
of base retained — sequence-level DPO loss barely touches fill-in-the-blank
memorization); ours removes cloze to 9%. Ours' neighbor collateral is also
concentrated in cloze (21% kept) while neighbor QA survives (67%).
Interpretation: token-level margin pinning transfers to completion-style
extraction; sequence-preference losses do not. Under content-removal with
paraphrase/jailbreak attackers, ours is the best functional point on real
knowledge; under collateral-minimization, NPO. Same personalities as TOFU.
Caveats: single declared config per method (no tuning grids), 1 seed, 10
targets, 1B model. fig_rwku_pilot.png/svg, t36_rwku_figure.py committed.

## 2026-08-13 22:0x UTC — t37 LAUNCHED: v1<->v3 Pareto dial + NPO hybrid (Logan)

Logan asked (a) whether the v1/v3 tradeoff traces a curve, (b) whether ours
composes with NPO. t37_llama_hybrid.py + t37_run.sh (background, marker T37
COMPLETE in t37.log):
- mix lam {0.25,0.5,0.75} x 3 seeds: retain = (1-lam)*margin_hinge +
  lam*logprob_pin + KL; forget side unchanged (min-token pin g4). Endpoints
  lam=0 (v1) and lam=1 (v3-lppin) already measured.
- npolp: NPO's reference-anchored forget loss (beta 0.1, sequence-sum) with
  OUR retain side (logprob pin + KL). Calibrate lr {1e-5, 2e-5} on seed 0,
  best by (admissible, utility), then seeds 1-2.
13 cells, ~11 min each; weights deleted after eval (re-derivable), summary
reports/remote/t37_hybrid_forget05.jsonl. Smoke-tested at 3 steps first.
Hypothesis: hybrid lands near (util 0.578, fq 0.63) — above both parents.
