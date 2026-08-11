#!/bin/bash
# T21: seed replication for relearn/jog/para claims. Trainings at
# seeds 1,2 for pin_g8 + npo, then the relearn matrix incl. relearn-
# seed variance on existing seed-0 checkpoints and the control.
set -u
ulimit -c 0
cd "$(dirname "$0")/.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY="taskset -c 0,2-19 .venv/bin/python"
mkdir -p results/t21_logs

for m in pin_g8 npo; do for s in 1 2; do
  if [ ! -d "results/t21_${m}_s${s}" ]; then
    echo "train $m s$s ..."
    $PY -u experiments/t21_seeds.py train $m $s \
        > "results/t21_logs/train_${m}_s${s}.log" 2>&1
    echo "train $m s$s exit=$?"
  fi
done; done

relearn () {  # relearn <dir> <tag> <lr> <src> <rseed>
  local suffix=""
  [ "$5" != "0" ] && suffix="/r$5"
  if grep -q "\"tag\": \"$2@$3/$4$suffix\"" results/t20_methods.jsonl 2>/dev/null; then
    echo "skip relearn $2@$3/$4$suffix"; return
  fi
  $PY -u experiments/t20_methods.py relearn "$1" "$2" "$3" "$4" "$5" \
      > "results/t21_logs/relearn_$2_$4_r$5.log" 2>&1
  echo "relearn $2@$3/$4$suffix exit=$?"
}

# new-seed checkpoints: direct relearn + jog (+ para for pin_g8)
for s in 1 2; do
  relearn "results/t21_pin_g8_s$s" "pin_g8_s$s" 1e-5 forget 0
  relearn "results/t21_pin_g8_s$s" "pin_g8_s$s" 1e-5 adjacent 0
  relearn "results/t21_pin_g8_s$s" "pin_g8_s$s" 1e-5 para 0
  relearn "results/t21_npo_s$s"    "npo_s$s"    1e-5 forget 0
  relearn "results/t21_npo_s$s"    "npo_s$s"    1e-5 adjacent 0
done
# relearn-seed variance on existing seed-0 checkpoints + control
for r in 1 2; do
  relearn results/t13_all_g8.0   all_g8     1e-5 forget $r
  relearn results/t15_retain_ref retain_ref 1e-5 forget $r
  relearn results/t15_retain_ref retain_ref 1e-5 para   $r
done
echo "t21 batch done"
