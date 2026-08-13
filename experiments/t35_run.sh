#!/bin/bash
# RWKU pilot runner (Logan-approved queueing 2026-08-13 "do queue up the #4
# realistic benchmark"). WAITS for the overnight chain's RELEARN2 COMPLETE
# marker, then:
#   basecheck (rank 200 targets by base 1B knowledge)
#   baseeval x10 (base-model reference rows)
#   {ga, npo, ours} x 10 targets: train -> eval -> delete weights (disk!)
# Summary rows accumulate in results/t35_rwku.jsonl (mirrored to reports).
set -u
cd "$(dirname "$0")"
source /venv/main/bin/activate

LOGDIR=results/t20_logs
RUNLOG="$LOGDIR/rwku.log"
mkdir -p "$LOGDIR"

until grep -q "RELEARN2 COMPLETE" "$LOGDIR/v3.log" 2>/dev/null; do sleep 120; done
echo "chain marker seen, starting RWKU pilot" >> "$RUNLOG"

if [ ! -f results/t35_targets.json ]; then
  python t35_rwku.py basecheck > "$LOGDIR/t35_basecheck.log" 2>&1
  echo "basecheck exit=$?" >> "$RUNLOG"
fi
[ -f results/t35_targets.json ] || { echo "FATAL no targets" >> "$RUNLOG"; exit 1; }

for k in 0 1 2 3 4 5 6 7 8 9; do
  if ! grep -q "\"tag\": \"t35_base_t${k}\"" results/t35_rwku.jsonl 2>/dev/null; then
    python t35_rwku.py baseeval "$k" > "$LOGDIR/t35_base_t${k}.log" 2>&1
    echo "baseeval t${k} exit=$?" >> "$RUNLOG"
  fi
done

for k in 0 1 2 3 4 5 6 7 8 9; do
  for method in ga npo ours; do
    tag="t35_${method}_t${k}"
    ckpt="results/${tag}"
    if ! grep -q "\"tag\": \"${tag}_eval\"" results/t35_rwku.jsonl 2>/dev/null; then
      if [ ! -f "$ckpt/config.json" ]; then
        python t35_rwku.py train "$method" "$k" > "$LOGDIR/${tag}_train.log" 2>&1
        echo "train $tag exit=$?" >> "$RUNLOG"
      fi
      [ -f "$ckpt/config.json" ] || { echo "FATAL train $tag" >> "$RUNLOG"; continue; }
      python t35_rwku.py eval "$ckpt" "$k" "${tag}_eval" > "$LOGDIR/${tag}_eval.log" 2>&1
      rc=$?
      echo "eval $tag exit=$rc" >> "$RUNLOG"
      # pilot checkpoints are disposable; 30 x 2.4GB would fill the disk
      [ $rc -eq 0 ] && rm -f "$ckpt"/*.safetensors
    fi
  done
  cp results/t35_rwku.jsonl ../reports/remote/t35_rwku.jsonl
  ( cd .. && git add reports/remote/t35_rwku.jsonl \
      && git commit -q -m "rwku pilot: target $k done" \
      && git pull --rebase -q origin main && git push -q origin main ) >> "$RUNLOG" 2>&1
done
cp results/t35_rwku.jsonl ../reports/remote/t35_rwku.jsonl
echo "RWKU PILOT COMPLETE" >> "$RUNLOG"
