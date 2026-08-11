#!/bin/bash
cd "$(dirname "$0")/.."
rm -f results/t22_lp_phase2.jsonl
for n in 350 250 150 100; do
  echo "=== N=$n ===" 
  T22_N=$n taskset -c 0,2-19 .venv/bin/python -u experiments/t22_lp_phase2.py
  rc=$?
  echo "N=$n exit=$rc"
  [ $rc -eq 0 ] && break   # first regime with interior completes fully
done
echo scan done
