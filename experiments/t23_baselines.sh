#!/bin/bash
# Baselines at open-unlearning's published per-method TOFU configs on
# Llama-3.2-1B, forget05, seeds 0-2. 4 methods x 3 seeds = 12 cells.
#
# Per prereg: effective batch 32 = per_device 4 x accum 8 (their per_device=8
# does not fit 32GB); sdpa (deviation 4, no sm_120 flash-attn); tokenizer from
# the ungated checkpoint (their tokenizer_args points at gated meta-llama).
# Training stays at their bf16 recipe -- the bf16 issue was eval-only.
# Eval: our t15 under the frozen headline protocol + FQ vs published retain95
# log. Same protocol stamps as the t20 grid, so rows are directly comparable.
set -u
cd "$(dirname "$0")"

SPLIT=forget05
RETAIN=retain95
LOGDIR=results/t23_logs
SUMMARY=../reports/remote/t23_baselines_forget05.jsonl
OU=/workspace/memorization/ref_repo/open-unlearning
mkdir -p "$LOGDIR"

for seed in 0 1 2; do
  for method in GradAscent NPO SimNPO RMU; do
    mtag=$(echo "$method" | tr 'A-Z' 'a-z')
    tag="t23_${SPLIT}_${mtag}_s${seed}"
    ckpt="$(pwd)/results/${tag}"
    efftag="${tag}_eval"

    # ---- train at their published config (skip if saved) ----
    if [ ! -f "$ckpt/config.json" ]; then
      for attempt in 1 2 3; do
        ( cd "$OU" && source /venv/oueval/bin/activate && \
          PYTHONPATH=src HF_DATASETS_CACHE=/workspace/.hf_home/datasets_ou \
          python src/train.py --config-name=unlearn.yaml \
            experiment=unlearn/tofu/default \
            trainer="$method" \
            model=Llama-3.2-1B-Instruct \
            model.model_args.attn_implementation=sdpa \
            model.tokenizer_args.pretrained_model_name_or_path=open-unlearning/tofu_Llama-3.2-1B-Instruct_full \
            forget_split="$SPLIT" retain_split="$RETAIN" holdout_split=holdout05 \
            trainer.args.per_device_train_batch_size=4 \
            trainer.args.gradient_accumulation_steps=8 \
            trainer.args.seed="$seed" \
            trainer.args.do_eval=False trainer.args.eval_on_start=False \
            task_name="$tag" \
            paths.output_dir="$ckpt" ) \
          > "$LOGDIR/${tag}_train.log" 2>&1
        rc=$?
        echo "train $tag attempt $attempt exit=$rc" >> "$LOGDIR/baselines.log"
        [ $rc -eq 0 ] && break
        sleep 10
      done
      [ -f "$ckpt/config.json" ] || { echo "FATAL train $tag" >> "$LOGDIR/baselines.log"; continue; }
    fi

    source /venv/main/bin/activate
    # ---- eval, frozen headline protocol (skip only at current protocol) ----
    if ! grep -q "\"tag\": \"$efftag\".*\"rouge_impl\": \"rouge_score\"" results/t15_metrics.jsonl 2>/dev/null; then
      T15_TOK_ID="$ckpt" T15_FORGET_SPLIT=${SPLIT}_perturbed \
      T15_TEMPLATE=llama3 T15_ROUGE=rouge_score T15_MAX_NEW=64 T15_TRUNCATE=1 \
        python t15_tofu_metrics.py eval "$ckpt" "$efftag" \
        > "$LOGDIR/${tag}_eval.log" 2>&1
      echo "eval $tag exit=$?" >> "$LOGDIR/baselines.log"
    fi

    # ---- forget quality vs published retain95 log ----
    python t21_fq_published.py fq "results/t15_truthratios/${efftag}.json" \
      tofu_Llama-3.2-1B-Instruct_${RETAIN} "$SPLIT" \
      > "$LOGDIR/${tag}_fq.log" 2>&1
    echo "fq $tag exit=$?" >> "$LOGDIR/baselines.log"

    # ---- summary row (idempotent) ----
    if grep -q "\"cell\": \"$tag\"" "$SUMMARY" 2>/dev/null; then
      echo "summary $tag already recorded" >> "$LOGDIR/baselines.log"
    else
    python - "$tag" "$efftag" "$SUMMARY" <<'PYEOF' >> "$LOGDIR/baselines.log" 2>&1
import json, re, sys
tag, efftag, out = sys.argv[1:4]
recs = [json.loads(l) for l in open("results/t15_metrics.jsonl")]
ev = [r for r in recs if r.get("tag") == efftag][-1]
fq = open(f"results/t23_logs/{tag}_fq.log").read()
m = re.search(r"forget_quality p=([0-9.e-]+)", fq)
ev["fq_p_vs_retain95"] = float(m.group(1)) if m else None
ev["cell"] = tag
with open(out, "a") as f:
    f.write(json.dumps(ev) + "\n")
print(f"summary {tag} fq_p={ev['fq_p_vs_retain95']}")
PYEOF
    fi

    # ---- persist: HF (async, serialized) + git ----
    (
      flock 9
      python hf_push.py "$ckpt" --model llama3.2-1b --split "$SPLIT" \
        --method "$mtag" --seed "$seed" \
        > "$LOGDIR/${tag}_push.log" 2>&1
      echo "hfpush $tag exit=$?" >> "$LOGDIR/baselines.log"
    ) 9>"$LOGDIR/.push.lock" &
    ( cd .. && git add reports/remote/t23_baselines_forget05.jsonl \
        && git commit -q -m "baseline: $tag" && git pull --rebase -q origin main \
        && git push -q origin main ) >> "$LOGDIR/baselines.log" 2>&1
  done
done
wait
echo "BASELINES COMPLETE" >> "$LOGDIR/baselines.log"
