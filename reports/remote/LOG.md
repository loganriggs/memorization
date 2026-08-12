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
