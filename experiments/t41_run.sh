#!/bin/bash
# t41: pin-on-AltPO — substitution + pin (Logan 2026-08-14: does burying the
# true answer under a REINFORCED SUBSTITUTE block the relearn path better
# than pin-on-NPO?). all-token pin g4 100 steps on the kept AltPO seed-0
# checkpoint; eval+fq; relearn both lrs; t34 RRS rerun.
# Marker: T41 COMPLETE in t39.log
set -u
cd "$(dirname "$0")"
source /venv/main/bin/activate
LOGDIR=results/t20_logs
RUNLOG="$LOGDIR/t39.log"
SUMMARY=../reports/remote/t37_hybrid_forget05.jsonl
TAG="t41_altpo_pin_all_g4_s0"
CKPT="results/${TAG}"
INIT="results/t39_forget05_altpo_lr1e-5_s0"

[ -f "$INIT/model.safetensors" ] || { echo "FATAL t41 no init ckpt" >> "$RUNLOG"; exit 1; }
if [ ! -f "$CKPT/model.safetensors" ]; then
  T37B_INIT="$INIT" T37B_TAG="$TAG" \
    python t37b_seqpin.py train all 100 0 > "$LOGDIR/${TAG}_train.log" 2>&1
  echo "train $TAG exit=$?" >> "$RUNLOG"
fi
[ -f "$CKPT/model.safetensors" ] || { echo "FATAL t41 train" >> "$RUNLOG"; exit 1; }

efftag="${TAG}_eval"
if ! grep -q "\"tag\": \"$efftag\".*\"rouge_impl\": \"rouge_score\"" results/t15_metrics.jsonl 2>/dev/null; then
  T15_TOK_ID="$CKPT" T15_FORGET_SPLIT=forget05_perturbed \
  T15_TEMPLATE=llama3 T15_ROUGE=rouge_score T15_MAX_NEW=64 T15_TRUNCATE=1 \
    python t15_tofu_metrics.py eval "$CKPT" "$efftag" > "$LOGDIR/${TAG}_eval.log" 2>&1
  echo "eval $TAG exit=$?" >> "$RUNLOG"
fi
python t21_fq_published.py fq "results/t15_truthratios/${efftag}.json" \
  tofu_Llama-3.2-1B-Instruct_retain95 forget05 > "$LOGDIR/${TAG}_fq.log" 2>&1
echo "fq $TAG exit=$?" >> "$RUNLOG"
if ! grep -q "\"cell\": \"$TAG\"" "$SUMMARY" 2>/dev/null; then
  python - "$TAG" "$efftag" "$SUMMARY" <<'PYEOF' >> "$RUNLOG" 2>&1
import json, re, sys
tag, efftag, out = sys.argv[1:4]
recs = [json.loads(l) for l in open("results/t15_metrics.jsonl")]
ev = [r for r in recs if r.get("tag") == efftag][-1]
m = re.search(r"forget_quality p=([0-9.e-]+)",
              open(f"results/t20_logs/{tag}_fq.log").read())
ev["fq_p_vs_retain95"] = float(m.group(1)) if m else None
ev["cell"] = tag
open(out, "a").write(json.dumps(ev) + "\n")
print(f"summary {tag} fq_p={ev['fq_p_vs_retain95']}")
PYEOF
fi

for lr in 1e-5 5e-5; do
  if ! grep -q "\"tag\": \"t25_altpo_pin_s0_lr${lr}\"" results/t25_relearn.jsonl 2>/dev/null; then
    python t25_relearn.py run "$CKPT" "t25_altpo_pin_s0_lr${lr}" "$lr" forget05 \
      > "$LOGDIR/t25_altpo_pin_lr${lr}.log" 2>&1
    echo "relearn altpo_pin lr=$lr exit=$?" >> "$RUNLOG"
  fi
done
python t34_rrs.py > "$LOGDIR/t34_rrs4.log" 2>&1
echo "rrs exit=$?" >> "$RUNLOG"
( cd .. && git add -A && git commit -q -m "t41: pin-on-AltPO cell + relearn + RRS" \
    && git pull --rebase -q origin main && git push -q origin main ) >> "$RUNLOG" 2>&1
echo "T41 COMPLETE" >> "$RUNLOG"
