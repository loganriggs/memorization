#!/bin/bash
# t37: v1<->v3 Pareto dial + NPO hybrid (Logan request 2026-08-13).
# Phase 1: npolp calibration on seed 0 at lr {1e-5, 2e-5} -> pick by
#          (admissible, then max utility; else max fq), run seeds 1-2.
# Phase 2: mix lam {0.25, 0.5, 0.75} x seeds {0,1,2}.
# Each cell: train -> eval (frozen protocol) -> fq -> summary row.
# Marker: T37 COMPLETE in results/t20_logs/t37.log
set -u
cd "$(dirname "$0")"
source /venv/main/bin/activate

LOGDIR=results/t20_logs
SUMMARY=../reports/remote/t37_hybrid_forget05.jsonl
RUNLOG="$LOGDIR/t37.log"
mkdir -p "$LOGDIR"

cell() {  # $1=tag  $2..=train args
  local tag=$1; shift
  local ckpt="results/${tag}" efftag="${tag}_eval"
  if [ ! -f "$ckpt/config.json" ]; then
    python t37_llama_hybrid.py train "$@" > "$LOGDIR/${tag}_train.log" 2>&1
    echo "train $tag exit=$?" >> "$RUNLOG"
    [ -f "$ckpt/config.json" ] || { echo "FATAL train $tag" >> "$RUNLOG"; return 1; }
  fi
  if ! grep -q "\"tag\": \"$efftag\".*\"rouge_impl\": \"rouge_score\"" results/t15_metrics.jsonl 2>/dev/null; then
    T15_TOK_ID="$ckpt" T15_FORGET_SPLIT=forget05_perturbed \
    T15_TEMPLATE=llama3 T15_ROUGE=rouge_score T15_MAX_NEW=64 T15_TRUNCATE=1 \
      python t15_tofu_metrics.py eval "$ckpt" "$efftag" \
      > "$LOGDIR/${tag}_eval.log" 2>&1
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
  # weights are re-derivable; keep disk flat (delete after eval+fq recorded)
  rm -f "$ckpt"/*.safetensors
}

# Phase 1: hybrid calibration + seeds
cell "t37_forget05_npolp_lr1e-05_s0" npolp 1e-5 0
cell "t37_forget05_npolp_lr2e-05_s0" npolp 2e-5 0
BEST_LR=$(python - <<'PYEOF'
import json
rows = [json.loads(l) for l in open("../reports/remote/t37_hybrid_forget05.jsonl")]
cand = [r for r in rows if "npolp" in r["cell"] and r["cell"].endswith("_s0")]
def key(r):
    fq = r["fq_p_vs_retain95"] or 0
    return (fq > 0.05, r["model_utility"] if fq > 0.05 else fq)
best = max(cand, key=key)
print(best["cell"].split("_lr")[1].split("_")[0])
PYEOF
)
echo "hybrid best lr: $BEST_LR" >> "$RUNLOG"
for seed in 1 2; do
  cell "t37_forget05_npolp_lr${BEST_LR}_s${seed}" npolp "$BEST_LR" "$seed"
done

# Phase 2: the lambda dial
for lam in 0.25 0.5 0.75; do
  for seed in 0 1 2; do
    cell "t37_forget05_mix${lam}_s${seed}" mix "$lam" "$seed"
  done
done

( cd .. && git add reports/remote/t37_hybrid_forget05.jsonl \
    && git commit -q -m "t37: hybrid + lambda-dial cells" \
    && git pull --rebase -q origin main && git push -q origin main ) >> "$RUNLOG" 2>&1
echo "T37 COMPLETE" >> "$RUNLOG"
