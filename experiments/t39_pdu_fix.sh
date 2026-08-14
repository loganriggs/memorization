#!/bin/bash
# PDU supplement: its shipped trainer config has retain_loss_eps: ??? (mandatory,
# no default) so the phase-B cells FATALed. Their community run.sh sets 0.3 for
# TOFU — rerun the 3 seeds with that value after the main chain frees the GPU.
# Marker: T39PDU COMPLETE in t39.log
set -u
cd "$(dirname "$0")"
source /venv/main/bin/activate
LOGDIR=results/t20_logs
RUNLOG="$LOGDIR/t39.log"
OU=/workspace/memorization/ref_repo/open-unlearning
SUMMARY=../reports/remote/t39_newbaselines.jsonl

until grep -q "T39 COMPLETE" "$RUNLOG" 2>/dev/null; do sleep 180; done
echo "pdu supplement starting (retain_loss_eps=0.3 per their run.sh)" >> "$RUNLOG"

for seed in 0 1 2; do
  tag="t39_forget05_pdu_s${seed}"
  ckpt="$(pwd)/results/${tag}"
  if [ ! -f "$ckpt/config.json" ]; then
    ( cd "$OU" && source /venv/oueval/bin/activate && \
      PYTHONPATH=src HF_DATASETS_CACHE=/workspace/.hf_home/datasets_ou \
      python src/train.py --config-name=unlearn.yaml \
        experiment=unlearn/tofu/default trainer=PDU \
        model=Llama-3.2-1B-Instruct \
        model.model_args.attn_implementation=sdpa \
        model.tokenizer_args.pretrained_model_name_or_path=open-unlearning/tofu_Llama-3.2-1B-Instruct_full \
        forget_split=forget05 retain_split=retain95 holdout_split=holdout05 \
        trainer.args.per_device_train_batch_size=4 \
        trainer.args.gradient_accumulation_steps=8 \
        trainer.args.do_eval=False trainer.args.eval_on_start=False \
        trainer.method_args.retain_loss_eps=0.3 \
        trainer.args.seed="$seed" \
        task_name="$tag" paths.output_dir="$ckpt" ) \
      > "$LOGDIR/${tag}_train.log" 2>&1
    echo "train $tag exit=$?" >> "$RUNLOG"
    source /venv/main/bin/activate
  fi
  [ -f "$ckpt/config.json" ] || { echo "FATAL train $tag" >> "$RUNLOG"; continue; }
  efftag="${tag}_eval"
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
  rm -f "$ckpt"/*.safetensors "$ckpt"/checkpoint-*/*.safetensors 2>/dev/null
done
( cd .. && git add reports/remote/t39_newbaselines.jsonl \
    && git commit -q -m "t39: PDU cells (retain_loss_eps=0.3 per their run.sh)" \
    && git pull --rebase -q origin main && git push -q origin main ) >> "$RUNLOG" 2>&1
echo "T39PDU COMPLETE" >> "$RUNLOG"
