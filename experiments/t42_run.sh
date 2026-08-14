#!/bin/bash
# t42: coherent vs far substitution (Logan's hypothesis: a consistent
# counter-world resists peeling; contrast = maximally dissimilar answers).
# Waits for T41 COMPLETE. Per arm: AltPO training recipe with our generated
# alternates -> eval+fq -> relearn both lrs -> RRS rerun at the end.
# Marker: T42 COMPLETE in t39.log
set -u
cd "$(dirname "$0")"
source /venv/main/bin/activate
LOGDIR=results/t20_logs
RUNLOG="$LOGDIR/t39.log"
OU=/workspace/memorization/ref_repo/open-unlearning
SUMMARY=../reports/remote/t37_hybrid_forget05.jsonl

until grep -q "T41 COMPLETE" "$RUNLOG" 2>/dev/null; do sleep 180; done
echo "t42 substitution variants starting" >> "$RUNLOG"

if [ ! -f results/t42_coherent.json ]; then
  python t42_gen.py > "$LOGDIR/t42_gen.log" 2>&1
  echo "t42 gen exit=$?" >> "$RUNLOG"
fi
[ -f results/t42_coherent.json ] || { echo "FATAL t42 gen" >> "$RUNLOG"; exit 1; }

for arm in coherent far; do
  TAG="t42_${arm}_s0"
  CKPT="$(pwd)/results/${TAG}"
  ALT="$(pwd)/results/t42_${arm}.json"
  if [ ! -f "$CKPT/config.json" ]; then
    ( cd "$OU" && source /venv/oueval/bin/activate && \
      PYTHONPATH=src HF_DATASETS_CACHE=/workspace/.hf_home/datasets_ou \
      python src/train.py --config-name=unlearn.yaml \
        experiment=unlearn/tofu/default trainer=DPO \
        model=Llama-3.2-1B-Instruct \
        model.model_args.attn_implementation=sdpa \
        model.tokenizer_args.pretrained_model_name_or_path=open-unlearning/tofu_Llama-3.2-1B-Instruct_full \
        forget_split=forget05 retain_split=retain95 holdout_split=holdout05 \
        trainer.args.per_device_train_batch_size=4 \
        trainer.args.gradient_accumulation_steps=8 \
        trainer.args.do_eval=False trainer.args.eval_on_start=False \
        trainer.args.seed=0 trainer.args.learning_rate=1e-5 \
        data.forget.TOFU_QA_forget.handler=QAwithAlternateDataset \
        ~data.forget.TOFU_QA_forget.args.hf_args.name \
        data.forget.TOFU_QA_forget.args.hf_args.path=json \
        +data.forget.TOFU_QA_forget.args.hf_args.data_files="$ALT" \
        data.forget.TOFU_QA_forget.args.hf_args.split=train \
        +data.forget.TOFU_QA_forget.args.alternate_key=alternate \
        task_name="$TAG" paths.output_dir="$CKPT" ) \
      > "$LOGDIR/${TAG}_train.log" 2>&1
    echo "train $TAG exit=$?" >> "$RUNLOG"
    source /venv/main/bin/activate
  fi
  [ -f "$CKPT/config.json" ] || { echo "FATAL train $TAG" >> "$RUNLOG"; continue; }

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
    if ! grep -q "\"tag\": \"t25_${arm}_s0_lr${lr}\"" results/t25_relearn.jsonl 2>/dev/null; then
      python t25_relearn.py run "$CKPT" "t25_${arm}_s0_lr${lr}" "$lr" forget05 \
        > "$LOGDIR/t25_${arm}_lr${lr}.log" 2>&1
      echo "relearn $arm lr=$lr exit=$?" >> "$RUNLOG"
    fi
  done
done
python t34_rrs.py > "$LOGDIR/t34_rrs5.log" 2>&1
echo "rrs exit=$?" >> "$RUNLOG"
( cd .. && git add -A && git commit -q -m "t42: coherent vs far substitution cells + relearn + RRS" \
    && git pull --rebase -q origin main && git push -q origin main ) >> "$RUNLOG" 2>&1
echo "T42 COMPLETE" >> "$RUNLOG"
