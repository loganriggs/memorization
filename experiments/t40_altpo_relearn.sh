#!/bin/bash
# t40: AltPO relearn/RRS — does overwrite-based unlearning resist relearning?
# Waits for the PDU supplement, retrains altpo_lr1e-5_s0 (weights were deleted
# after eval; deterministic), runs t25 relearn at both lrs, reruns t34.
# Marker: T40 COMPLETE in t39.log
set -u
cd "$(dirname "$0")"
source /venv/main/bin/activate
LOGDIR=results/t20_logs
RUNLOG="$LOGDIR/t39.log"
OU=/workspace/memorization/ref_repo/open-unlearning

until grep -q "T39PDU COMPLETE" "$RUNLOG" 2>/dev/null; do sleep 180; done
echo "t40 altpo relearn starting" >> "$RUNLOG"

tag="t39_forget05_altpo_lr1e-5_s0"
ckpt="$(pwd)/results/${tag}"
ALT_JSON="$OU/community/methods/AltPO/data/tofu_Llama-3.2-1B-Instruct_full/forget05/alt5_seed_0.json"
if [ ! -f "$ckpt/model.safetensors" ]; then
  rm -rf "$ckpt"
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
      +data.forget.TOFU_QA_forget.args.hf_args.data_files="$ALT_JSON" \
      data.forget.TOFU_QA_forget.args.hf_args.split=train \
      +data.forget.TOFU_QA_forget.args.alternate_key=alternate \
      task_name="$tag" paths.output_dir="$ckpt" ) \
    > "$LOGDIR/${tag}_retrain.log" 2>&1
  echo "retrain $tag exit=$?" >> "$RUNLOG"
  source /venv/main/bin/activate
fi
[ -f "$ckpt/model.safetensors" ] || { echo "FATAL t40 retrain" >> "$RUNLOG"; exit 1; }

for lr in 1e-5 5e-5; do
  if ! grep -q "\"tag\": \"t25_altpo_s0_lr${lr}\"" results/t25_relearn.jsonl 2>/dev/null; then
    python t25_relearn.py run "$ckpt" "t25_altpo_s0_lr${lr}" "$lr" forget05 \
      > "$LOGDIR/t25_altpo_lr${lr}.log" 2>&1
    echo "relearn altpo lr=$lr exit=$?" >> "$RUNLOG"
  fi
done
python t34_rrs.py > "$LOGDIR/t34_rrs3.log" 2>&1
echo "rrs exit=$?" >> "$RUNLOG"
( cd .. && git add -A && git commit -q -m "t40: AltPO relearn curves + RRS update" \
    && git pull --rebase -q origin main && git push -q origin main ) >> "$RUNLOG" 2>&1
echo "T40 COMPLETE" >> "$RUNLOG"
