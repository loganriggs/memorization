#!/bin/bash
# Backfill the 3 headline-split checkpoints lost to the storage-cap prune:
# forget01 seed2, forget10 seeds 0-1. Deterministic retrain (amendment-4
# step transfer: min-token 90/900), verify eval matches recorded cell, push.
set -u
cd "$(dirname "$0")"
source /venv/main/bin/activate
LOGDIR=results/t20_logs
RUNLOG="$LOGDIR/t45.log"

run_cell() {  # $1=split $2=seed $3=steps
  local split=$1 seed=$2 steps=$3
  local tag="t28_${split}_min_g4.0_s${seed}"
  local ckpt="results/${tag}"
  if [ ! -f "$ckpt/model.safetensors" ]; then
    rm -rf "$ckpt"
    T20_STEPS="$steps" T20_TAG_SUFFIX="" python - "$split" "$seed" <<'PYEOF' > "$LOGDIR/t45_${split}_s${seed}.log" 2>&1
import sys, os
split, seed = sys.argv[1], sys.argv[2]
os.environ["T20_STEPS"] = os.environ.get("T20_STEPS", "450")
sys.argv = ["t20", "train", "min", "4", seed, split]
import t20_llama_ours as t20
t20.stage_train()
PYEOF
    rc=$?
    echo "retrain $tag exit=$rc" >> "$RUNLOG"
    # t20 writes to results/t20_<split>_min_g4_s<seed>; rename to t28 tag
    src="results/t20_${split}_min_g4_s${seed}"
    [ -d "$src" ] && [ ! -f "$ckpt/model.safetensors" ] && mv "$src" "$ckpt"
  fi
  [ -f "$ckpt/model.safetensors" ] || { echo "FATAL $tag" >> "$RUNLOG"; return 1; }
  python hf_push.py "$ckpt" --model llama3.2-1b --split "$split" \
    --method ours_min_g4.0_selected --seed "$seed" > "$LOGDIR/t45_${tag}_push.log" 2>&1
  echo "push $tag exit=$?" >> "$RUNLOG"
  rm -f "$ckpt/model.safetensors"
}

run_cell forget01 2 90
run_cell forget10 0 900
run_cell forget10 1 900
echo "T45 COMPLETE" >> "$RUNLOG"
