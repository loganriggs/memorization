#!/bin/bash
# Overnight chain (Logan-approved 2026-08-13): v3 retain-objective variants,
# then relearn-resistance curves for the third metric, then RRS computation.
#
# Phase A: t33 v3 {ce, lppin} x 3 seeds on forget05 (train -> eval -> fq ->
#          summary -> async HF push). Marker: V3 COMPLETE
# Phase B: retrain t20 all-token g4 s0 (weights lost to storage prune+janitor;
#          100 steps, deterministic seed) + verify eval.
# Phase C: relearn curves (t25, both lrs) for: all_g4_s0, npo_lr2e-05_s0,
#          v3 winner s0 (winner = admissible w/ max utility, else max fq).
# Phase D: t34 RRS metric + figures. Marker: RELEARN2 COMPLETE
set -u
cd "$(dirname "$0")"
source /venv/main/bin/activate

LOGDIR=results/t20_logs
SUMMARY=../reports/remote/t33_v3_forget05.jsonl
RUNLOG="$LOGDIR/v3.log"
mkdir -p "$LOGDIR"

run_eval_fq_summary() {  # $1=tag  $2=ckpt  $3=cellname  $4=summary_file
  local tag=$1 ckpt=$2 cell=$3 summ=$4 efftag="$1_eval"
  if ! grep -q "\"tag\": \"$efftag\".*\"rouge_impl\": \"rouge_score\"" results/t15_metrics.jsonl 2>/dev/null; then
    T15_TOK_ID="$ckpt" T15_FORGET_SPLIT=forget05_perturbed \
    T15_TEMPLATE=llama3 T15_ROUGE=rouge_score T15_MAX_NEW=64 T15_TRUNCATE=1 \
      python t15_tofu_metrics.py eval "$ckpt" "$efftag" \
      > "$LOGDIR/${tag}_eval.log" 2>&1
    echo "eval $tag exit=$?" >> "$RUNLOG"
  fi
  python t21_fq_published.py fq "results/t15_truthratios/${efftag}.json" \
    tofu_Llama-3.2-1B-Instruct_retain95 forget05 \
    > "$LOGDIR/${tag}_fq.log" 2>&1
  echo "fq $tag exit=$?" >> "$RUNLOG"
  if ! grep -q "\"cell\": \"$cell\"" "$summ" 2>/dev/null; then
    python - "$tag" "$efftag" "$cell" "$summ" <<'PYEOF' >> "$RUNLOG" 2>&1
import json, re, sys
tag, efftag, cell, out = sys.argv[1:5]
recs = [json.loads(l) for l in open("results/t15_metrics.jsonl")]
ev = [r for r in recs if r.get("tag") == efftag][-1]
m = re.search(r"forget_quality p=([0-9.e-]+)",
              open(f"results/t20_logs/{tag}_fq.log").read())
ev["fq_p_vs_retain95"] = float(m.group(1)) if m else None
ev["cell"] = cell
with open(out, "a") as f:
    f.write(json.dumps(ev) + "\n")
print(f"summary {cell} fq_p={ev['fq_p_vs_retain95']}")
PYEOF
  fi
}

# ---------------- Phase A: v3 variants ----------------
for variant in ce lppin; do
  for seed in 0 1 2; do
    tag="t33_forget05_min_g4_s${seed}_${variant}"
    ckpt="results/${tag}"
    if [ ! -f "$ckpt/config.json" ]; then
      for attempt in 1 2 3; do
        python t33_llama_v3.py train "$variant" "$seed" forget05 \
          > "$LOGDIR/${tag}_train.log" 2>&1
        rc=$?
        echo "train $tag attempt $attempt exit=$rc" >> "$RUNLOG"
        [ $rc -eq 0 ] && break
        sleep 10
      done
      [ -f "$ckpt/config.json" ] || { echo "FATAL train $tag" >> "$RUNLOG"; continue; }
    fi
    run_eval_fq_summary "$tag" "$ckpt" "$tag" "$SUMMARY"
    (
      flock 9
      python hf_push.py "$ckpt" --model llama3.2-1b --split forget05 \
        --method "ours_min_g4.0_v3${variant}" --seed "$seed" \
        > "$LOGDIR/${tag}_push.log" 2>&1
      echo "hfpush $tag exit=$?" >> "$RUNLOG"
    ) 9>"$LOGDIR/.push.lock" &
    ( cd .. && git add reports/remote/t33_v3_forget05.jsonl \
        && git commit -q -m "v3: $tag" && git pull --rebase -q origin main \
        && git push -q origin main ) >> "$RUNLOG" 2>&1
  done
done
echo "V3 COMPLETE" >> "$RUNLOG"

# ---------------- Phase B: restore all-token g4 s0 ----------------
AG="results/t20_forget05_all_g4_s0"
if [ ! -f "$AG/model.safetensors" ]; then
  rm -rf "$AG"   # config-only husk left by janitor; retrain deterministically
  T20_STEPS=100 python t20_llama_ours.py train all 4 0 forget05 \
    > "$LOGDIR/t20_forget05_all_g4_s0_retrain.log" 2>&1
  echo "retrain all_g4_s0 exit=$?" >> "$RUNLOG"
  # verify it reproduces the recorded cell (leak ~0.035, util ~0.469)
  run_eval_fq_summary "t20_forget05_all_g4_s0_retrain" "$AG" \
    "t20_forget05_all_g4_s0_retrain" "../reports/remote/t33_v3_forget05.jsonl"
fi

# ---------------- Phase C: relearn curves ----------------
# v3 winner: admissible (fq>0.05) with max utility; fallback max fq. Seed 0.
WINNER_VARIANT=$(python - <<'PYEOF'
import json
rows = [json.loads(l) for l in open("../reports/remote/t33_v3_forget05.jsonl")
        if '"t33_' in l]
from collections import defaultdict
by = defaultdict(list)
for r in rows:
    by[r["cell"].rsplit("_", 1)[1]].append(r)
def key(v):
    rs = by[v]
    fq = sum(x["fq_p_vs_retain95"] or 0 for x in rs) / len(rs)
    ut = sum(x["model_utility"] for x in rs) / len(rs)
    return (fq > 0.05, ut if fq > 0.05 else fq)
print(max(by, key=key) if by else "ce")
PYEOF
)
echo "v3 winner variant: $WINNER_VARIANT" >> "$RUNLOG"

for subj in "results/t20_forget05_all_g4_s0|t25_allg4_s0" \
            "results/t23_forget05_npo_lr2e-05_s0|t25_npo2e5_s0" \
            "results/t33_forget05_min_g4_s0_${WINNER_VARIANT}|t25_v3${WINNER_VARIANT}_s0"; do
  ckpt="${subj%|*}"; rtag="${subj#*|}"
  [ -f "$ckpt/model.safetensors" ] || { echo "SKIP relearn $rtag (no weights)" >> "$RUNLOG"; continue; }
  for lr in 1e-5 5e-5; do
    if grep -q "\"tag\": \"${rtag}_lr${lr}\"" results/t25_relearn.jsonl 2>/dev/null; then
      echo "relearn $rtag lr=$lr already recorded" >> "$RUNLOG"; continue
    fi
    python t25_relearn.py run "$ckpt" "${rtag}_lr${lr}" "$lr" forget05 \
      > "$LOGDIR/${rtag}_lr${lr}_relearn.log" 2>&1
    echo "relearn $rtag lr=$lr exit=$?" >> "$RUNLOG"
  done
done

# ---------------- Phase D: RRS metric + figures ----------------
python t34_rrs.py > "$LOGDIR/t34_rrs.log" 2>&1
echo "rrs exit=$?" >> "$RUNLOG"
( cd .. && git add -A && git commit -q -m "overnight: v3 cells, relearn curves, RRS metric" \
    && git pull --rebase -q origin main && git push -q origin main ) >> "$RUNLOG" 2>&1
wait
echo "RELEARN2 COMPLETE" >> "$RUNLOG"
