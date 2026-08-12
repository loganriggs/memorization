#!/bin/bash
# Headline splits (forget01, forget10) for the SELECTED config, 3 seeds each.
# Refuses to run unless reports/remote/t20_selection.json exists and was
# produced by t24 on the complete grid -- the prereg's ordering requirement.
# Steps per amendment 4: constant sample-epochs (scale with split rows).
# Floors are measured from the split's retain reference before its cells score.
set -u
cd "$(dirname "$0")"
source /venv/main/bin/activate

SEL=../reports/remote/t20_selection.json
[ -f "$SEL" ] || { echo "no selection file -- run t24 on the complete grid first"; exit 2; }
SCOPE=$(python -c "import json;print(json.load(open('$SEL'))['selected']['scope'])")
GAMMA=$(python -c "import json;print(json.load(open('$SEL'))['selected']['gamma'])")
STATUS=$(python -c "import json;print(json.load(open('$SEL'))['status'])")
LOGDIR=results/t28_logs
SUMMARY=../reports/remote/t28_headline.jsonl
mkdir -p "$LOGDIR"
echo "selected: $SCOPE gamma=$GAMMA status=$STATUS" >> "$LOGDIR/headline.log"

# amendment-4 step table
declare -A STEPS_F01=( [all]=20 [min]=90 )
declare -A STEPS_F10=( [all]=200 [min]=900 )

for SPLIT in forget01 forget10; do
  case $SPLIT in
    forget01) RETAIN=retain99; STEPS=${STEPS_F01[$SCOPE]}; HOLD=holdout01;;
    forget10) RETAIN=retain90; STEPS=${STEPS_F10[$SCOPE]}; HOLD=holdout10;;
  esac

  # ---- measure this split's floor first (retain reference, frozen protocol) ----
  ftag="floorRS_${RETAIN}_${SPLIT}"
  if ! grep -q "\"tag\": \"$ftag\".*\"rouge_impl\": \"rouge_score\"" results/t15_metrics.jsonl 2>/dev/null; then
    T15_TOK_ID="open-unlearning/tofu_Llama-3.2-1B-Instruct_${RETAIN}" \
    T15_FORGET_SPLIT=${SPLIT}_perturbed T15_TEMPLATE=llama3 \
    T15_ROUGE=rouge_score T15_MAX_NEW=64 T15_TRUNCATE=1 \
      python t15_tofu_metrics.py eval \
      "open-unlearning/tofu_Llama-3.2-1B-Instruct_${RETAIN}" "$ftag" \
      > "$LOGDIR/${ftag}.log" 2>&1
    echo "floor $ftag exit=$?" >> "$LOGDIR/headline.log"
  fi

  for seed in 0 1 2; do
    tag="t28_${SPLIT}_${SCOPE}_g${GAMMA}_s${seed}"
    ckpt="results/${tag}"
    efftag="${tag}_eval"

    if [ ! -f "$ckpt/config.json" ]; then
      for attempt in 1 2 3; do
        T20_STEPS="$STEPS" python t20_llama_ours.py train "$SCOPE" "$GAMMA" "$seed" "$SPLIT" \
          > "$LOGDIR/${tag}_train.log" 2>&1
        rc=$?
        echo "train $tag attempt $attempt exit=$rc" >> "$LOGDIR/headline.log"
        [ $rc -eq 0 ] && break
        sleep 10
      done
      # t20 writes to results/t20_<split>_<scope>_g<gamma>_s<seed>; move to t28 name
      src="results/t20_${SPLIT}_${SCOPE}_g$(python -c "print(f'{float('$GAMMA'):g}')")_s${seed}"
      [ -d "$src" ] && [ ! -d "$ckpt" ] && mv "$src" "$ckpt"
      [ -f "$ckpt/config.json" ] || { echo "FATAL train $tag" >> "$LOGDIR/headline.log"; continue; }
    fi

    if ! grep -q "\"tag\": \"$efftag\".*\"rouge_impl\": \"rouge_score\"" results/t15_metrics.jsonl 2>/dev/null; then
      T15_TOK_ID="$ckpt" T15_FORGET_SPLIT=${SPLIT}_perturbed T15_TEMPLATE=llama3 \
      T15_ROUGE=rouge_score T15_MAX_NEW=64 T15_TRUNCATE=1 \
        python t15_tofu_metrics.py eval "$ckpt" "$efftag" \
        > "$LOGDIR/${tag}_eval.log" 2>&1
      echo "eval $tag exit=$?" >> "$LOGDIR/headline.log"
    fi

    python t21_fq_published.py fq "results/t15_truthratios/${efftag}.json" \
      "tofu_Llama-3.2-1B-Instruct_${RETAIN}" "$SPLIT" \
      > "$LOGDIR/${tag}_fq.log" 2>&1
    echo "fq $tag exit=$?" >> "$LOGDIR/headline.log"

    if ! grep -q "\"cell\": \"$tag\"" "$SUMMARY" 2>/dev/null; then
    python - "$tag" "$efftag" "$SUMMARY" "$STATUS" <<'PYEOF' >> "$LOGDIR/headline.log" 2>&1
import json, re, sys
tag, efftag, out, status = sys.argv[1:5]
recs = [json.loads(l) for l in open("results/t15_metrics.jsonl")]
ev = [r for r in recs if r.get("tag") == efftag][-1]
fq = open(f"results/t28_logs/{tag}_fq.log").read()
m = re.search(r"forget_quality p=([0-9.e-]+)", fq)
ev["fq_p"] = float(m.group(1)) if m else None
ev["cell"] = tag
ev["selection_status"] = status
with open(out, "a") as f:
    f.write(json.dumps(ev) + "\n")
print(f"summary {tag} fq_p={ev['fq_p']}")
PYEOF
    fi

    (
      flock 9
      python hf_push.py "$ckpt" --model llama3.2-1b --split "$SPLIT" \
        --method "ours_${SCOPE}_g${GAMMA}_selected" --seed "$seed" \
        > "$LOGDIR/${tag}_push.log" 2>&1
      echo "hfpush $tag exit=$?" >> "$LOGDIR/headline.log"
    ) 9>"$LOGDIR/.push.lock" &
    ( cd .. && git add reports/remote/t28_headline.jsonl \
        && git commit -q -m "headline: $tag" && git pull --rebase -q origin main \
        && git push -q origin main ) >> "$LOGDIR/headline.log" 2>&1
  done
done
wait
echo "HEADLINE COMPLETE" >> "$LOGDIR/headline.log"
