#!/bin/bash
# t39 master chain (Logan 2026-08-14: "do all three" + baseline completeness).
# Phase R: rebuild the 9 champion checkpoints deterministically, verify by
#          re-eval (tag *_r2), KEEP weights, push all to HF.  Marker: T39R
# Phase B: run every open-unlearning-shipped method we skipped, at shipped
#          defaults, 3 seeds, forget05: GradDiff WGA SatImp UNDIAL CEU PDU;
#          AltPO (DPO trainer + generated alternate answers) at lr {1e-5,2e-5}
#          (same 2x budget NPO got).                        Marker: T39B
# Phase C: champion relearn curves (t25, both lrs) + t34 RRS rerun. Marker: T39C
# Phase D: RWKU pilot, hybrid method, 10 targets + t36 figure. Marker: T39 COMPLETE
set -u
cd "$(dirname "$0")"
source /venv/main/bin/activate

LOGDIR=results/t20_logs
RUNLOG="$LOGDIR/t39.log"
OU=/workspace/memorization/ref_repo/open-unlearning
SUMMARY=../reports/remote/t39_newbaselines.jsonl
mkdir -p "$LOGDIR"

evalcell() {  # $1=tag $2=ckpt $3=efftag $4=summary(or "-") $5=cellname
  local tag=$1 ckpt=$2 efftag=$3 summ=$4 cell=$5
  if ! grep -q "\"tag\": \"$efftag\".*\"rouge_impl\": \"rouge_score\"" results/t15_metrics.jsonl 2>/dev/null; then
    T15_TOK_ID="$ckpt" T15_FORGET_SPLIT=forget05_perturbed \
    T15_TEMPLATE=llama3 T15_ROUGE=rouge_score T15_MAX_NEW=64 T15_TRUNCATE=1 \
      python t15_tofu_metrics.py eval "$ckpt" "$efftag" > "$LOGDIR/${tag}_eval.log" 2>&1
    echo "eval $tag exit=$?" >> "$RUNLOG"
  fi
  python t21_fq_published.py fq "results/t15_truthratios/${efftag}.json" \
    tofu_Llama-3.2-1B-Instruct_retain95 forget05 > "$LOGDIR/${tag}_fq.log" 2>&1
  echo "fq $tag exit=$?" >> "$RUNLOG"
  if [ "$summ" != "-" ] && ! grep -q "\"cell\": \"$cell\"" "$summ" 2>/dev/null; then
    python - "$tag" "$efftag" "$cell" "$summ" <<'PYEOF' >> "$RUNLOG" 2>&1
import json, re, sys
tag, efftag, cell, out = sys.argv[1:5]
recs = [json.loads(l) for l in open("results/t15_metrics.jsonl")]
ev = [r for r in recs if r.get("tag") == efftag][-1]
m = re.search(r"forget_quality p=([0-9.e-]+)",
              open(f"results/t20_logs/{tag}_fq.log").read())
ev["fq_p_vs_retain95"] = float(m.group(1)) if m else None
ev["cell"] = cell
open(out, "a").write(json.dumps(ev) + "\n")
print(f"summary {cell} fq_p={ev['fq_p_vs_retain95']}")
PYEOF
  fi
}

# ================= Phase R: champions, rebuilt + kept + pushed =================
for seed in 0 1 2; do
  tag="t37_forget05_npolp_lr1e-05_s${seed}"
  ckpt="results/${tag}"
  if [ ! -f "$ckpt/model.safetensors" ]; then
    rm -rf "$ckpt"
    python t37_llama_hybrid.py train npolp 1e-5 "$seed" > "$LOGDIR/${tag}_retrain.log" 2>&1
    echo "retrain $tag exit=$?" >> "$RUNLOG"
  fi
  evalcell "$tag" "$ckpt" "${tag}_r2_eval" "-" "-"
  ( flock 9
    python hf_push.py "$ckpt" --model llama3.2-1b --split forget05 \
      --method hybrid_joint --seed "$seed" > "$LOGDIR/${tag}_push.log" 2>&1
    echo "hfpush $tag exit=$?" >> "$RUNLOG" ) 9>"$LOGDIR/.push.lock" &
done
for seed in 0 1 2; do
  for sc in "min 200" "all 100"; do
    scope="${sc% *}"; steps="${sc#* }"
    tag="t37s_forget05_${scope}_g4_s${seed}"
    ckpt="results/${tag}"
    if [ ! -f "$ckpt/model.safetensors" ]; then
      rm -rf "$ckpt"
      python t37b_seqpin.py train "$scope" "$steps" "$seed" > "$LOGDIR/${tag}_retrain.log" 2>&1
      echo "retrain $tag exit=$?" >> "$RUNLOG"
    fi
    evalcell "$tag" "$ckpt" "${tag}_r2_eval" "-" "-"
    ( flock 9
      python hf_push.py "$ckpt" --model llama3.2-1b --split forget05 \
        --method "hybrid_seq_${scope}" --seed "$seed" > "$LOGDIR/${tag}_push.log" 2>&1
      echo "hfpush $tag exit=$?" >> "$RUNLOG" ) 9>"$LOGDIR/.push.lock" &
  done
done
# reproduction check: r2 evals vs original records
python - <<'PYEOF' >> "$RUNLOG" 2>&1
import json
recs = [json.loads(l) for l in open("results/t15_metrics.jsonl")]
by = {r["tag"]: r for r in recs if r.get("stage") == "eval"}
for base in ([f"t37_forget05_npolp_lr1e-05_s{s}" for s in range(3)] +
             [f"t37s_forget05_{sc}_g4_s{s}" for sc in ("min", "all") for s in range(3)]):
    a, b = by.get(f"{base}_eval"), by.get(f"{base}_r2_eval")
    if a and b:
        print(f"REPRO {base}: util {a['model_utility']:.4f}->{b['model_utility']:.4f} "
              f"leak {a['forget_rouge']:.4f}->{b['forget_rouge']:.4f}")
PYEOF
echo "T39R COMPLETE" >> "$RUNLOG"

# ================= Phase B: skipped shipped baselines =================
run_ou() {  # $1=tag $2=trainer $3... extra hydra overrides
  local tag=$1 trainer=$2; shift 2
  local ckpt="$(pwd)/results/${tag}"
  if [ ! -f "$ckpt/config.json" ]; then
    ( cd "$OU" && source /venv/oueval/bin/activate && \
      PYTHONPATH=src HF_DATASETS_CACHE=/workspace/.hf_home/datasets_ou \
      python src/train.py --config-name=unlearn.yaml \
        experiment=unlearn/tofu/default \
        trainer="$trainer" \
        model=Llama-3.2-1B-Instruct \
        model.model_args.attn_implementation=sdpa \
        model.tokenizer_args.pretrained_model_name_or_path=open-unlearning/tofu_Llama-3.2-1B-Instruct_full \
        forget_split=forget05 retain_split=retain95 holdout_split=holdout05 \
        trainer.args.per_device_train_batch_size=4 \
        trainer.args.gradient_accumulation_steps=8 \
        trainer.args.do_eval=False trainer.args.eval_on_start=False \
        task_name="$tag" paths.output_dir="$ckpt" "$@" ) \
      > "$LOGDIR/${tag}_train.log" 2>&1
    echo "train $tag exit=$?" >> "$RUNLOG"
    source /venv/main/bin/activate
  fi
  [ -f "$ckpt/config.json" ] || { echo "FATAL train $tag" >> "$RUNLOG"; return 1; }
  evalcell "$tag" "$ckpt" "${tag}_eval" "$SUMMARY" "$tag"
  rm -f "$ckpt"/*.safetensors "$ckpt"/checkpoint-*/*.safetensors 2>/dev/null
}

for seed in 0 1 2; do
  for method in GradDiff WGA SatImp UNDIAL CEU PDU; do
    mtag=$(echo "$method" | tr 'A-Z' 'a-z')
    run_ou "t39_forget05_${mtag}_s${seed}" "$method" trainer.args.seed="$seed"
  done
done

# AltPO: generate alternate answers once, then DPO+alt at lr {1e-5, 2e-5}
ALT_JSON="$OU/community/methods/AltPO/data/tofu_Llama-3.2-1B-Instruct_full/forget05/alt5_seed_0.json"
if [ ! -f "$ALT_JSON" ]; then
  ( cd "$OU/community/methods/AltPO" && source /venv/oueval/bin/activate && \
    PYTHONPATH="$OU/src" python generate.py dataset_config.dataset_kwargs.name=forget05 ) \
    > "$LOGDIR/t39_altpo_generate.log" 2>&1
  echo "altpo generate exit=$?" >> "$RUNLOG"
  source /venv/main/bin/activate
fi
if [ -f "$ALT_JSON" ]; then
  for lr in 1e-5 2e-5; do
    for seed in 0 1 2; do
      run_ou "t39_forget05_altpo_lr${lr}_s${seed}" DPO \
        trainer.args.seed="$seed" trainer.args.learning_rate="$lr" \
        data.forget.TOFU_QA_forget.handler=QAwithAlternateDataset \
        ~data.forget.TOFU_QA_forget.args.hf_args.name \
        data.forget.TOFU_QA_forget.args.hf_args.path=json \
        +data.forget.TOFU_QA_forget.args.hf_args.data_files="$ALT_JSON" \
        data.forget.TOFU_QA_forget.args.hf_args.split=train \
        +data.forget.TOFU_QA_forget.args.alternate_key=alternate
    done
  done
else
  echo "SKIP altpo (generation failed)" >> "$RUNLOG"
fi
( cd .. && git add reports/remote/t39_newbaselines.jsonl \
    && git commit -q -m "t39: skipped-baseline sweep rows" \
    && git pull --rebase -q origin main && git push -q origin main ) >> "$RUNLOG" 2>&1
echo "T39B COMPLETE" >> "$RUNLOG"

# ================= Phase C: champion relearn + RRS =================
for sub in "results/t37_forget05_npolp_lr1e-05_s0|t25_hybrid_joint_s0" \
           "results/t37s_forget05_all_g4_s0|t25_hybrid_seqall_s0" \
           "results/t37s_forget05_min_g4_s0|t25_hybrid_seqmin_s0"; do
  ckpt="${sub%|*}"; rtag="${sub#*|}"
  [ -f "$ckpt/model.safetensors" ] || { echo "SKIP relearn $rtag" >> "$RUNLOG"; continue; }
  for lr in 1e-5 5e-5; do
    if ! grep -q "\"tag\": \"${rtag}_lr${lr}\"" results/t25_relearn.jsonl 2>/dev/null; then
      python t25_relearn.py run "$ckpt" "${rtag}_lr${lr}" "$lr" forget05 \
        > "$LOGDIR/${rtag}_lr${lr}_relearn.log" 2>&1
      echo "relearn $rtag lr=$lr exit=$?" >> "$RUNLOG"
    fi
  done
done
python t34_rrs.py > "$LOGDIR/t34_rrs2.log" 2>&1
echo "rrs exit=$?" >> "$RUNLOG"
echo "T39C COMPLETE" >> "$RUNLOG"

# ================= Phase D: RWKU hybrid =================
for k in 0 1 2 3 4 5 6 7 8 9; do
  tag="t35_hybrid_t${k}"
  ckpt="results/${tag}"
  if ! grep -q "\"tag\": \"${tag}_eval\"" results/t35_rwku.jsonl 2>/dev/null; then
    if [ ! -f "$ckpt/config.json" ]; then
      python t35_rwku.py train hybrid "$k" > "$LOGDIR/${tag}_train.log" 2>&1
      echo "train $tag exit=$?" >> "$RUNLOG"
    fi
    [ -f "$ckpt/config.json" ] || { echo "FATAL train $tag" >> "$RUNLOG"; continue; }
    python t35_rwku.py eval "$ckpt" "$k" "${tag}_eval" > "$LOGDIR/${tag}_eval.log" 2>&1
    rc=$?
    echo "eval $tag exit=$rc" >> "$RUNLOG"
    [ $rc -eq 0 ] && rm -f "$ckpt"/*.safetensors
  fi
done
cp results/t35_rwku.jsonl ../reports/remote/t35_rwku.jsonl
python t36_rwku_figure.py > "$LOGDIR/t36_2.log" 2>&1
echo "rwku fig exit=$?" >> "$RUNLOG"
wait
( cd .. && git add -A && git commit -q -m "t39: champions repro+push, new baselines, champion RRS, RWKU hybrid" \
    && git pull --rebase -q origin main && git push -q origin main ) >> "$RUNLOG" 2>&1
echo "T39 COMPLETE" >> "$RUNLOG"
