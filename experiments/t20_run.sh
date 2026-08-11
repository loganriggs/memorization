#!/bin/bash
# T20 batch: flat_l1, SAM baselines, reoccupation; evals+KS; relearn
# matrix incl. adjacent-data (jog) and paraphrase attacks; weight
# distance analysis. Sequential, resumable, pinned off core 4.
set -u
ulimit -c 0
cd "$(dirname "$0")/.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY="taskset -c 0,2-19 .venv/bin/python"
mkdir -p results/t20_logs

for m in flat_l1 npo_sam pin_sam reoccupy; do
  if [ ! -d "results/t20_$m" ]; then
    echo "train $m ..."
    $PY -u experiments/t20_methods.py train $m \
        > "results/t20_logs/train_$m.log" 2>&1
    echo "train $m exit=$?"
  fi
done

for m in flat_l1 npo_sam pin_sam reoccupy; do
  if ! grep -q "\"tag\": \"$m\"" results/t15_metrics.jsonl 2>/dev/null; then
    for attempt in 1 2 3; do
      $PY -u experiments/t15_tofu_metrics.py eval "results/t20_$m" "$m" \
          >> "results/t20_logs/eval_$m.log" 2>&1 && break
    done
    $PY experiments/t15_tofu_metrics.py ks "$m" retain_ref \
        >> results/t20_logs/ks.log 2>&1 || true
  fi
done

relearn () {  # relearn <dir> <tag> <lr> <src>
  if grep -q "\"tag\": \"$2@$3/$4\"" results/t20_methods.jsonl 2>/dev/null; then
    echo "skip relearn $2@$3/$4"; return
  fi
  $PY -u experiments/t20_methods.py relearn "$1" "$2" "$3" "$4" \
      > "results/t20_logs/relearn_$2_$3_$4.log" 2>&1
  echo "relearn $2@$3/$4 exit=$?"
}

for m in flat_l1 npo_sam pin_sam reoccupy; do
  relearn "results/t20_$m" $m 1e-5 forget
  relearn "results/t20_$m" $m 5e-5 forget
done
# adjacent-data "jog" attack + paraphrase familiarity control
for spec in "results/t13_all_g2.0 all_g2" "results/t13_all_g8.0 all_g8" \
            "results/t11_tofu_npo npo" "results/t20_flat_l1 flat_l1" \
            "results/t20_reoccupy reoccupy" \
            "results/t15_retain_ref retain_ref"; do
  set -- $spec
  relearn "$1" "$2" 1e-5 adjacent
done
relearn results/t13_all_g8.0   all_g8     1e-5 para
relearn results/t15_retain_ref retain_ref 1e-5 para

$PY -u experiments/t20_methods.py wdist \
    > results/t20_logs/wdist.log 2>&1
echo "wdist exit=$?"
echo "t20 batch done"
