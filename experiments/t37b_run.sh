#!/bin/bash
# t37b: sequential pin-on-NPO (waits for T37 COMPLETE). 6 cells:
# {min g4 200 steps, all g4 100 steps} x 3 seeds, init from tuned NPO seed k.
# Marker: T37B COMPLETE in results/t20_logs/t37.log
set -u
cd "$(dirname "$0")"
source /venv/main/bin/activate

LOGDIR=results/t20_logs
SUMMARY=../reports/remote/t37_hybrid_forget05.jsonl
RUNLOG="$LOGDIR/t37.log"

until grep -q "T37 COMPLETE" "$RUNLOG" 2>/dev/null; do sleep 120; done
echo "t37 marker seen, starting seqpin" >> "$RUNLOG"

cell() {  # $1=tag $2=scope $3=steps $4=seed
  local tag=$1 ckpt="results/$1" efftag="$1_eval"
  if [ ! -f "$ckpt/config.json" ]; then
    python t37b_seqpin.py train "$2" "$3" "$4" > "$LOGDIR/${tag}_train.log" 2>&1
    echo "train $tag exit=$?" >> "$RUNLOG"
    [ -f "$ckpt/config.json" ] || { echo "FATAL train $tag" >> "$RUNLOG"; return 1; }
  fi
  if ! grep -q "\"tag\": \"$efftag\".*\"rouge_impl\": \"rouge_score\"" results/t15_metrics.jsonl 2>/dev/null; then
    T15_TOK_ID="$ckpt" T15_FORGET_SPLIT=forget05_perturbed \
    T15_TEMPLATE=llama3 T15_ROUGE=rouge_score T15_MAX_NEW=64 T15_TRUNCATE=1 \
      python t15_tofu_metrics.py eval "$ckpt" "$efftag" > "$LOGDIR/${tag}_eval.log" 2>&1
    echo "eval $tag exit=$?" >> "$RUNLOG"
  fi
  python t21_fq_published.py fq "results/t15_truthratios/${efftag}.json" \
    tofu_Llama-3.2-1B-Instruct_retain95 forget05 > "$LOGDIR/${tag}_fq.log" 2>&1
  echo "fq $tag exit=$?" >> "$RUNLOG"
  if ! grep -q "\"cell\": \"$tag\"" "$SUMMARY" 2>/dev/null; then
    python - "$tag" "$efftag" "$SUMMARY" <<'PYEOF' >> "$RUNLOG" 2>&1
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
  rm -f "$ckpt"/*.safetensors
}

for seed in 0 1 2; do
  cell "t37s_forget05_min_g4_s${seed}" min 200 "$seed"
  cell "t37s_forget05_all_g4_s${seed}" all 100 "$seed"
done

( cd .. && git add reports/remote/t37_hybrid_forget05.jsonl \
    && git commit -q -m "t37b: sequential pin-on-NPO cells" \
    && git pull --rebase -q origin main && git push -q origin main ) >> "$RUNLOG" 2>&1
echo "T37B COMPLETE" >> "$RUNLOG"
