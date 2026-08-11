#!/bin/bash
# Sequential, resumable t15 pipeline. No output-hiding pipes: full logs
# per stage in results/t15_logs/. Skips eval tags already in the jsonl.
# Usage: bash experiments/t15_run_all.sh
set -u
ulimit -c 0   # no core dumps: a crashed 6GB process must not wedge the next one
cd "$(dirname "$0")/.."
PY=.venv/bin/python
mkdir -p results/t15_logs

evaluate () {  # evaluate <model_dir> <tag>
  if grep -q "\"tag\": \"$2\"" results/t15_metrics.jsonl 2>/dev/null; then
    echo "skip $2 (done)"; return
  fi
  for attempt in 1 2 3; do
    echo "eval $2 (attempt $attempt) ..."
    $PY -u experiments/t15_tofu_metrics.py eval "$1" "$2" \
        >> "results/t15_logs/eval_$2.log" 2>&1
    rc=$?
    echo "eval $2 exit=$rc"
    [ $rc -eq 0 ] && return
  done
}

evaluate results/t11_tofu_base        base
evaluate results/t11_tofu_ga          ga
evaluate results/t11_tofu_npo         npo
evaluate results/t11_tofu_ours        ours_min
evaluate results/t11_tofu_ours_alltok ours_alltok
evaluate results/t13_min_g0.5         min_g0.5
evaluate results/t13_min_g2.0         min_g2
evaluate results/t13_min_g8.0         min_g8
evaluate results/t13_all_g0.5         all_g0.5
evaluate results/t13_all_g2.0         all_g2
evaluate results/t13_all_g8.0         all_g8

# retain-only reference model (after evals: GPU to itself)
if [ ! -d results/t15_retain_ref ]; then
  echo "training retain reference ..."
  $PY -u experiments/t15_tofu_metrics.py train_retain \
      > results/t15_logs/train_retain.log 2>&1
  echo "train_retain exit=$?"
fi

evaluate results/t15_retain_ref retain_ref

# forget quality: KS vs retain_ref for every unlearned checkpoint + base
for tag in base ga npo ours_min ours_alltok \
           min_g0.5 min_g2 min_g8 all_g0.5 all_g2 all_g8; do
  $PY experiments/t15_tofu_metrics.py ks "$tag" retain_ref \
      >> results/t15_logs/ks.log 2>&1
done
echo "pipeline done"
