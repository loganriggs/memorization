#!/bin/bash
# T19: flatten2 curriculum (pin-only 100 steps, then pin + grad-norm
# penalty). Train, eval+KS, relearn at both lrs. Pinned, resumable.
set -u
ulimit -c 0
cd "$(dirname "$0")/.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY="taskset -c 0,2-19 .venv/bin/python"
mkdir -p results/t19_logs

if [ ! -d results/t17_flatten2 ]; then
  echo "train flatten2 ..."
  $PY -u experiments/t17_methods.py train flatten2 \
      > results/t19_logs/flatten2.log 2>&1
  echo "train exit=$?"
fi
if ! grep -q '"tag": "flatten2"' results/t15_metrics.jsonl 2>/dev/null; then
  for attempt in 1 2 3; do
    $PY -u experiments/t15_tofu_metrics.py eval results/t17_flatten2 flatten2 \
        >> results/t19_logs/eval.log 2>&1 && break
  done
  $PY experiments/t15_tofu_metrics.py ks flatten2 retain_ref \
      >> results/t19_logs/ks.log 2>&1 || true
fi
for lr in 1e-5 5e-5; do
  if ! grep -q "\"tag\": \"flatten2@$lr\"" results/t17_methods.jsonl 2>/dev/null; then
    $PY -u experiments/t17_methods.py relearn results/t17_flatten2 flatten2 $lr \
        > "results/t19_logs/relearn_$lr.log" 2>&1
    echo "relearn @$lr exit=$?"
  fi
done
echo "t19 done"
