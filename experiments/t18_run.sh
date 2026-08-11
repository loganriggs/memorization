#!/bin/bash
# T18 batch: S2 gradient-flattening + clean decoy, evals, and the
# two-lr relearn matrix (red-team #6). Sequential, resumable, pinned.
set -u
ulimit -c 0
cd "$(dirname "$0")/.."
PY="taskset -c 0,2-19 .venv/bin/python"
mkdir -p results/t18_logs

train () {
  local ckpt=$1 logf=$2; shift 2
  if [ -d "$ckpt" ]; then echo "skip train $ckpt"; return; fi
  echo "train -> $ckpt ..."
  $PY -u "$@" > "results/t18_logs/$logf" 2>&1
  echo "train $ckpt exit=$?"
}
train results/t17_flatten flatten.log experiments/t17_methods.py train flatten
train results/t17_decoy2  decoy2.log  experiments/t17_methods.py train decoy2

evaluate () {
  if grep -q "\"tag\": \"$2\"" results/t15_metrics.jsonl 2>/dev/null; then
    echo "skip eval $2"; return
  fi
  for attempt in 1 2 3; do
    echo "eval $2 (attempt $attempt) ..."
    $PY -u experiments/t15_tofu_metrics.py eval "$1" "$2" \
        >> "results/t18_logs/eval_$2.log" 2>&1
    rc=$?; echo "eval $2 exit=$rc"; [ $rc -eq 0 ] && break
  done
  $PY experiments/t15_tofu_metrics.py ks "$2" retain_ref \
      >> results/t18_logs/ks.log 2>&1 || true
}
evaluate results/t17_flatten flatten
evaluate results/t17_decoy2  decoy2

relearn () {  # relearn <model_dir> <tag> <lr>
  if grep -q "\"tag\": \"$2@$3\"" results/t17_methods.jsonl 2>/dev/null; then
    echo "skip relearn $2@$3"; return
  fi
  echo "relearn $2 lr=$3 ..."
  $PY -u experiments/t17_methods.py relearn "$1" "$2" "$3" \
      > "results/t18_logs/relearn_$2_$3.log" 2>&1
  echo "relearn $2@$3 exit=$?"
}
relearn results/t17_flatten    flatten    1e-5
relearn results/t17_decoy2     decoy2     1e-5
relearn results/t15_retain_ref retain_ref 5e-5
relearn results/t11_tofu_npo   npo        5e-5
relearn results/t13_all_g2.0   all_g2     5e-5
relearn results/t13_all_g8.0   all_g8     5e-5
relearn results/t17_flatten    flatten    5e-5
relearn results/t17_decoy2     decoy2     5e-5
echo "t18 batch done"
