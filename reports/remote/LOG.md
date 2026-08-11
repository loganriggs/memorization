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

**Next:** `verify_env.py` (matmul+backward on sm_120, bf16+grad-checkpoint,
5 min stress) -> P2 evaluator equivalence -> pre-register gamma/scope on
forget05 -> matrix.
