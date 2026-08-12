#!/bin/bash
# Pre-registered forget05 gamma/scope sweep (prereg frozen at 96f8fec,
# amendments 1-2 applied: eot excluded from training labels, step count
# selected on forget05, headline scorer = rouge_score).
# T20_GRID_STEPS must be set to the calibrated step count before running.
# 2 scopes x 4 gammas x 3 seeds = 24 cells, sequential and resumable:
# skip-if-done, per-cell logs, explicit exit codes, HF push + git push per cell.
#
# Per cell: train (t20) -> eval (t15, frozen headline protocol) -> FQ (t21 vs
# published retain95 log) -> hf_push -> append summary + push.
# Never pipe python through tail/grep; logs go to files, exit codes checked.
set -u
cd "$(dirname "$0")"
source /venv/main/bin/activate

SPLIT=forget05
STEPS="${T20_GRID_STEPS:?set T20_GRID_STEPS to the calibrated step count}"
LOGDIR=results/t20_logs
SUMMARY=../reports/remote/t20_forget05_sweep.jsonl
mkdir -p "$LOGDIR"

# Wait for the floor eval to release the GPU (matches its tag, not this script).

for seed in 0 1 2; do
  for scope in all min; do
    for gamma in 0.5 1 2 4; do
      tag="t20_${SPLIT}_${scope}_g${gamma}_s${seed}"
      ckpt="results/${tag}"
      efftag="${tag}_eval"

      # ---- train (skip if checkpoint already saved) ----
      if [ ! -f "$ckpt/config.json" ]; then
        for attempt in 1 2 3; do
          T20_STEPS="$STEPS" python t20_llama_ours.py train "$scope" "$gamma" "$seed" "$SPLIT" \
            > "$LOGDIR/${tag}_train.log" 2>&1
          rc=$?
          echo "train $tag attempt $attempt exit=$rc" >> "$LOGDIR/sweep.log"
          [ $rc -eq 0 ] && break
          sleep 10
        done
        [ -f "$ckpt/config.json" ] || { echo "FATAL train $tag" >> "$LOGDIR/sweep.log"; continue; }
      fi

      # ---- eval under the frozen headline protocol ----
      if ! grep -q "\"tag\": \"$efftag\"" results/t15_metrics.jsonl 2>/dev/null; then
        T15_TOK_ID="$ckpt" T15_FORGET_SPLIT=${SPLIT}_perturbed \
        T15_TEMPLATE=llama3 T15_ROUGE=rouge_score T15_MAX_NEW=64 T15_TRUNCATE=1 \
          python t15_tofu_metrics.py eval "$ckpt" "$efftag" \
          > "$LOGDIR/${tag}_eval.log" 2>&1
        echo "eval $tag exit=$?" >> "$LOGDIR/sweep.log"
      fi

      # ---- forget quality vs published retain95 log ----
      python t21_fq_published.py fq "results/t15_truthratios/${efftag}.json" \
        tofu_Llama-3.2-1B-Instruct_retain95 "$SPLIT" \
        > "$LOGDIR/${tag}_fq.log" 2>&1
      echo "fq $tag exit=$?" >> "$LOGDIR/sweep.log"

      # ---- summary row (eval record + fq p-value) ----
      python - "$tag" "$efftag" "$SUMMARY" <<'PYEOF' >> "$LOGDIR/sweep.log" 2>&1
import json, re, sys
tag, efftag, out = sys.argv[1:4]
recs = [json.loads(l) for l in open("results/t15_metrics.jsonl")]
ev = [r for r in recs if r.get("tag") == efftag][-1]
fq = open(f"results/t20_logs/{tag}_fq.log").read()
m = re.search(r"forget_quality p=([0-9.e-]+)", fq)
ev["fq_p_vs_retain95"] = float(m.group(1)) if m else None
ev["cell"] = tag
with open(out, "a") as f:
    f.write(json.dumps(ev) + "\n")
print(f"summary {tag} fq_p={ev['fq_p_vs_retain95']}")
PYEOF

      # ---- persist: HF checkpoint + git summary ----
      python hf_push.py "$ckpt" --model llama3.2-1b --split "$SPLIT" \
        --method "ours_${scope}_g${gamma}" --seed "$seed" \
        > "$LOGDIR/${tag}_push.log" 2>&1
      echo "hfpush $tag exit=$?" >> "$LOGDIR/sweep.log"
      ( cd .. && git add reports/remote/t20_forget05_sweep.jsonl \
          && git commit -q -m "sweep: $tag" && git pull --rebase -q origin main \
          && git push -q origin main ) >> "$LOGDIR/sweep.log" 2>&1
    done
  done
done
echo "SWEEP COMPLETE" >> "$LOGDIR/sweep.log"
