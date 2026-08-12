#!/bin/bash
# Pre-registered forget05 gamma/scope sweep (prereg frozen at 96f8fec,
# amendments 1-2 applied: eot excluded from training labels, step count
# selected on forget05, headline scorer = rouge_score).
# Step counts are calibrated per scope and baked in below.
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
# Calibrated per scope (prereg amendment 3, recorded 2026-08-12):
#   all-token step 100 (FQ 0.0221, interior peak)
#   min-token step 450 (FQ 0.0085, interior peak)
declare -A SCOPE_STEPS=( [all]="${T20_STEPS_ALL:-100}" [min]="${T20_STEPS_MIN:-450}" )
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
          T20_STEPS="${SCOPE_STEPS[$scope]}" python t20_llama_ours.py train "$scope" "$gamma" "$seed" "$SPLIT" \
            > "$LOGDIR/${tag}_train.log" 2>&1
          rc=$?
          echo "train $tag attempt $attempt exit=$rc" >> "$LOGDIR/sweep.log"
          [ $rc -eq 0 ] && break
          sleep 10
        done
        [ -f "$ckpt/config.json" ] || { echo "FATAL train $tag" >> "$LOGDIR/sweep.log"; continue; }
      fi

      # ---- eval under the frozen headline protocol ----
      # Skip only if a record exists for this tag AT THE CURRENT PROTOCOL.
      # Matching the tag alone let a pre-amendment record (rouge_impl=lcs) be
      # reused for a freshly retrained checkpoint -- stale metrics attached to
      # a new model, invisible in the summary.
      if ! grep -q "\"tag\": \"$efftag\".*\"rouge_impl\": \"rouge_score\"" results/t15_metrics.jsonl 2>/dev/null; then
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

      # ---- summary row (eval record + fq p-value; skip if already recorded) ----
      if grep -q "\"cell\": \"$tag\"" "$SUMMARY" 2>/dev/null; then
        echo "summary $tag already recorded" >> "$LOGDIR/sweep.log"
      else
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
      fi

      # ---- persist: HF checkpoint (async) + git summary ----
      # The upload is ~2.4 GB and takes 10-15 min, during which the GPU would
      # sit at 0% if we waited for it. Push in the background so the next cell
      # trains immediately; flock serializes uploads among themselves so we
      # never run N concurrent 2.4 GB transfers.
      (
        flock 9
        python hf_push.py "$ckpt" --model llama3.2-1b --split "$SPLIT" \
          --method "ours_${scope}_g${gamma}" --seed "$seed" \
          > "$LOGDIR/${tag}_push.log" 2>&1
        echo "hfpush $tag exit=$?" >> "$LOGDIR/sweep.log"
      ) 9>"$LOGDIR/.push.lock" &
      ( cd .. && git add reports/remote/t20_forget05_sweep.jsonl \
          && git commit -q -m "sweep: $tag" && git pull --rebase -q origin main \
          && git push -q origin main ) >> "$LOGDIR/sweep.log" 2>&1
    done
  done
done
wait   # let any in-flight uploads finish before declaring done
echo "SWEEP COMPLETE" >> "$LOGDIR/sweep.log"
