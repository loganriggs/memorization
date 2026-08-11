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

**Next:** finish our-side eval -> run their evaluator on the same checkpoint ->
per-factor + aggregate diff -> pre-register gamma/scope on forget05 -> matrix.
