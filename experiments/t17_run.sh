#!/bin/bash
# T17 campaign batch: RMU + fairness factorial + SimNPO + decoy pilot,
# then official metrics + KS, then relearn curves (incl. the never-knew
# control). Sequential, resumable, pinned off unstable core 4.
set -u
ulimit -c 0
cd "$(dirname "$0")/.."
PY="taskset -c 0,2-19 .venv/bin/python"
mkdir -p results/t17_logs

train () {  # train <script args...> <ckpt_dir> <log>
  local ckpt=$1 logf=$2; shift 2
  if [ -d "$ckpt" ]; then echo "skip train $ckpt (exists)"; return; fi
  echo "train -> $ckpt ..."
  $PY -u "$@" > "results/t17_logs/$logf" 2>&1
  echo "train $ckpt exit=$?"
}

train results/t16_rmu       rmu.log         experiments/t16_rmu.py unlearn
train results/t17_npo_klhinge npo_klhinge.log experiments/t17_methods.py train npo_klhinge
train results/t17_pin_ce    pin_ce.log      experiments/t17_methods.py train pin_ce
train results/t17_simnpo    simnpo.log      experiments/t17_methods.py train simnpo
train results/t17_decoy     decoy.log       experiments/t17_methods.py train decoy

evaluate () {  # evaluate <model_dir> <tag>
  if grep -q "\"tag\": \"$2\"" results/t15_metrics.jsonl 2>/dev/null; then
    echo "skip eval $2 (done)"; return
  fi
  for attempt in 1 2 3; do
    echo "eval $2 (attempt $attempt) ..."
    $PY -u experiments/t15_tofu_metrics.py eval "$1" "$2" \
        >> "results/t17_logs/eval_$2.log" 2>&1
    rc=$?; echo "eval $2 exit=$rc"
    [ $rc -eq 0 ] && break
  done
  $PY experiments/t15_tofu_metrics.py ks "$2" retain_ref \
      >> results/t17_logs/ks.log 2>&1 || true
}

evaluate results/t16_rmu         rmu
evaluate results/t17_npo_klhinge npo_klhinge
evaluate results/t17_pin_ce     pin_ce
evaluate results/t17_simnpo     simnpo
evaluate results/t17_decoy      decoy

relearn () {  # relearn <model_dir> <tag>
  if grep -q "\"t17_relearn\", \"tag\": \"$2\"" results/t17_methods.jsonl 2>/dev/null; then
    echo "skip relearn $2 (done)"; return
  fi
  echo "relearn $2 ..."
  $PY -u experiments/t17_methods.py relearn "$1" "$2" \
      > "results/t17_logs/relearn_$2.log" 2>&1
  echo "relearn $2 exit=$?"
}

relearn results/t15_retain_ref  retain_ref_control
relearn results/t11_tofu_npo    npo
relearn results/t13_all_g2.0    all_g2
relearn results/t13_all_g8.0    all_g8
relearn results/t16_rmu         rmu
relearn results/t17_npo_klhinge npo_klhinge
relearn results/t17_pin_ce     pin_ce
relearn results/t17_simnpo     simnpo
relearn results/t17_decoy      decoy
echo "t17 batch done"
