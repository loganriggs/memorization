#!/bin/bash
# t44 master chain (Logan 2026-08-15: mechanism diagnostic + hardening items).
# M:  restore subjects (altpo retrain, seq-all fetch from HF), t43 mechanism.
# H1: seq-all RRS seeds 1-2 (2 lrs each) + t34 rerun.          Marker: T44H1
# H2: cross-splits forget01/forget10: NPO 2e-5, joint hybrid, seq-all x3
#     seeds each; eval+fq vs the split's retain reference.      Marker: T44H2
# H3: RWKU at 50 targets x {ga,npo,ours,hybrid} (+40 baseevals). T44 COMPLETE
set -u
cd "$(dirname "$0")"
source /venv/main/bin/activate
LOGDIR=results/t20_logs
RUNLOG="$LOGDIR/t44.log"
OU=/workspace/memorization/ref_repo/open-unlearning
CROSS=../reports/remote/t44_cross_splits.jsonl
mkdir -p "$LOGDIR"

# ---------------- Phase M ----------------
# seq-all weights: fetch all 3 seeds back from HF into their result dirs
python - <<'PYEOF' >> "$RUNLOG" 2>&1
import shutil, os
from huggingface_hub import hf_hub_download
for s in range(3):
    dst = f"results/t37s_forget05_all_g4_s{s}/model.safetensors"
    if not os.path.exists(dst):
        p = hf_hub_download("Elriggs/memorization-unlearning",
                            f"llama3.2-1b/forget05/hybrid_seq_all/seed{s}/model.safetensors")
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy(p, dst)
        print(f"fetched seq_all seed{s}")
PYEOF
echo "seqall fetch exit=$?" >> "$RUNLOG"

# altpo subject: deterministic retrain if pruned
AP="results/t39_forget05_altpo_lr1e-5_s0"
ALT_JSON="$OU/community/methods/AltPO/data/tofu_Llama-3.2-1B-Instruct_full/forget05/alt5_seed_0.json"
if [ ! -f "$AP/model.safetensors" ]; then
  rm -rf "$AP"
  ( cd "$OU" && source /venv/oueval/bin/activate && \
    PYTHONPATH=src HF_DATASETS_CACHE=/workspace/.hf_home/datasets_ou \
    python src/train.py --config-name=unlearn.yaml \
      experiment=unlearn/tofu/default trainer=DPO \
      model=Llama-3.2-1B-Instruct model.model_args.attn_implementation=sdpa \
      model.tokenizer_args.pretrained_model_name_or_path=open-unlearning/tofu_Llama-3.2-1B-Instruct_full \
      forget_split=forget05 retain_split=retain95 holdout_split=holdout05 \
      trainer.args.per_device_train_batch_size=4 trainer.args.gradient_accumulation_steps=8 \
      trainer.args.do_eval=False trainer.args.eval_on_start=False \
      trainer.args.seed=0 trainer.args.learning_rate=1e-5 \
      data.forget.TOFU_QA_forget.handler=QAwithAlternateDataset \
      ~data.forget.TOFU_QA_forget.args.hf_args.name \
      data.forget.TOFU_QA_forget.args.hf_args.path=json \
      +data.forget.TOFU_QA_forget.args.hf_args.data_files="$ALT_JSON" \
      data.forget.TOFU_QA_forget.args.hf_args.split=train \
      +data.forget.TOFU_QA_forget.args.alternate_key=alternate \
      task_name=t39_forget05_altpo_lr1e-5_s0 paths.output_dir="$(pwd)/$AP" ) \
    > "$LOGDIR/t44_altpo_retrain.log" 2>&1
  echo "altpo retrain exit=$?" >> "$RUNLOG"
  source /venv/main/bin/activate
fi

python t43_mech.py > "$LOGDIR/t43_mech.log" 2>&1
echo "t43 mech exit=$?" >> "$RUNLOG"
( cd .. && git add -A && git commit -q -m "t43: relearn-mechanism diagnostic" \
    && git pull --rebase -q origin main && git push -q origin main ) >> "$RUNLOG" 2>&1
echo "T44M COMPLETE" >> "$RUNLOG"

# ---------------- Phase H1: seq-all seeds 1-2 RRS ----------------
for seed in 1 2; do
  ckpt="results/t37s_forget05_all_g4_s${seed}"
  for lr in 1e-5 5e-5; do
    rtag="t25_hybrid_seqall_s${seed}_lr${lr}"
    if ! grep -q "\"tag\": \"$rtag\"" results/t25_relearn.jsonl 2>/dev/null; then
      python t25_relearn.py run "$ckpt" "$rtag" "$lr" forget05 \
        > "$LOGDIR/${rtag}_relearn.log" 2>&1
      echo "relearn seqall_s${seed} lr=$lr exit=$?" >> "$RUNLOG"
    fi
  done
done
python t34_rrs.py > "$LOGDIR/t34_rrs6.log" 2>&1
echo "rrs exit=$?" >> "$RUNLOG"
( cd .. && git add -A && git commit -q -m "t44 H1: seq-all RRS seeds 1-2" \
    && git pull --rebase -q origin main && git push -q origin main ) >> "$RUNLOG" 2>&1
echo "T44H1 COMPLETE" >> "$RUNLOG"

# ---------------- Phase H2: cross-splits ----------------
declare -A REF=( [forget01]="tofu_Llama-3.2-1B-Instruct_retain99" [forget10]="tofu_Llama-3.2-1B-Instruct_retain90" )
declare -A RETAIN=( [forget01]="retain99" [forget10]="retain90" )
declare -A HOLD=( [forget01]="holdout01" [forget10]="holdout10" )
declare -A JSTEPS=( [forget01]="100" [forget10]="1000" )
declare -A PSTEPS=( [forget01]="20" [forget10]="200" )

xeval() {  # $1=tag $2=ckpt $3=split
  local tag=$1 ckpt=$2 split=$3 efftag="$1_eval"
  if ! grep -q "\"tag\": \"$efftag\".*\"rouge_impl\": \"rouge_score\"" results/t15_metrics.jsonl 2>/dev/null; then
    T15_TOK_ID="$ckpt" T15_FORGET_SPLIT=${split}_perturbed \
    T15_TEMPLATE=llama3 T15_ROUGE=rouge_score T15_MAX_NEW=64 T15_TRUNCATE=1 \
      python t15_tofu_metrics.py eval "$ckpt" "$efftag" > "$LOGDIR/${tag}_eval.log" 2>&1
    echo "eval $tag exit=$?" >> "$RUNLOG"
  fi
  python t21_fq_published.py fq "results/t15_truthratios/${efftag}.json" \
    "${REF[$split]}" "$split" > "$LOGDIR/${tag}_fq.log" 2>&1
  echo "fq $tag exit=$?" >> "$RUNLOG"
  if ! grep -q "\"cell\": \"$tag\"" "$CROSS" 2>/dev/null; then
    python - "$tag" "$efftag" "$CROSS" <<'PYEOF' >> "$RUNLOG" 2>&1
import json, re, sys
tag, efftag, out = sys.argv[1:4]
recs = [json.loads(l) for l in open("results/t15_metrics.jsonl")]
ev = [r for r in recs if r.get("tag") == efftag][-1]
m = re.search(r"forget_quality p=([0-9.e-]+)",
              open(f"results/t20_logs/{tag}_fq.log").read())
ev["fq_p_vs_ref"] = float(m.group(1)) if m else None
ev["cell"] = tag
open(out, "a").write(json.dumps(ev) + "\n")
print(f"summary {tag} fq_p={ev['fq_p_vs_ref']}")
PYEOF
  fi
}

for split in forget01 forget10; do
  for seed in 0 1 2; do
    # NPO 2e-5 at the split
    ntag="t23_${split}_npo_lr2e-05_s${seed}"
    nckpt="$(pwd)/results/${ntag}"
    if [ ! -f "$nckpt/model.safetensors" ]; then
      ( cd "$OU" && source /venv/oueval/bin/activate && \
        PYTHONPATH=src HF_DATASETS_CACHE=/workspace/.hf_home/datasets_ou \
        python src/train.py --config-name=unlearn.yaml \
          experiment=unlearn/tofu/default trainer=NPO \
          model=Llama-3.2-1B-Instruct model.model_args.attn_implementation=sdpa \
          model.tokenizer_args.pretrained_model_name_or_path=open-unlearning/tofu_Llama-3.2-1B-Instruct_full \
          forget_split="$split" retain_split="${RETAIN[$split]}" holdout_split="${HOLD[$split]}" \
          trainer.args.per_device_train_batch_size=4 trainer.args.gradient_accumulation_steps=8 \
          trainer.args.do_eval=False trainer.args.eval_on_start=False \
          trainer.args.seed="$seed" trainer.args.learning_rate=2e-5 \
          task_name="$ntag" paths.output_dir="$nckpt" ) \
        > "$LOGDIR/${ntag}_train.log" 2>&1
      echo "train $ntag exit=$?" >> "$RUNLOG"
      source /venv/main/bin/activate
    fi
    [ -f "$nckpt/model.safetensors" ] || { echo "FATAL $ntag" >> "$RUNLOG"; continue; }
    xeval "$ntag" "$nckpt" "$split"

    # joint hybrid at the split (sample-budget-matched steps)
    jtag="t37_${split}_npolp_lr1e-05_s${seed}"
    if [ ! -f "results/${jtag}/config.json" ]; then
      T37_SPLIT="$split" T37_STEPS="${JSTEPS[$split]}" \
        python t37_llama_hybrid.py train npolp 1e-5 "$seed" \
        > "$LOGDIR/${jtag}_train.log" 2>&1
      echo "train $jtag exit=$?" >> "$RUNLOG"
    fi
    [ -f "results/${jtag}/config.json" ] && xeval "$jtag" "results/${jtag}" "$split"
    rm -f "results/${jtag}"/model.safetensors

    # seq-all pin on the split's NPO (amendment-4 step transfer)
    stag="t37s_${split}_all_g4_s${seed}"
    if [ ! -f "results/${stag}/config.json" ]; then
      T37B_SPLIT="$split" python t37b_seqpin.py train all "${PSTEPS[$split]}" "$seed" \
        > "$LOGDIR/${stag}_train.log" 2>&1
      echo "train $stag exit=$?" >> "$RUNLOG"
    fi
    [ -f "results/${stag}/config.json" ] && xeval "$stag" "results/${stag}" "$split"
    rm -f "results/${stag}"/model.safetensors "$nckpt"/model.safetensors \
          "$nckpt"/checkpoint-*/*.safetensors 2>/dev/null
  done
done
( cd .. && git add -A && git commit -q -m "t44 H2: cross-split NPO/joint/seq-all" \
    && git pull --rebase -q origin main && git push -q origin main ) >> "$RUNLOG" 2>&1
echo "T44H2 COMPLETE" >> "$RUNLOG"

# ---------------- Phase H3: RWKU at 50 targets ----------------
export T35_TOPN=50
for k in $(seq 10 49); do
  if ! grep -q "\"tag\": \"t35_base_t${k}\"" results/t35_rwku.jsonl 2>/dev/null; then
    python t35_rwku.py baseeval "$k" > "$LOGDIR/t35_base_t${k}.log" 2>&1
    echo "baseeval t${k} exit=$?" >> "$RUNLOG"
  fi
done
for k in $(seq 0 49); do
  for method in ga npo ours hybrid; do
    tag="t35_${method}_t${k}"
    ckpt="results/${tag}"
    if ! grep -q "\"tag\": \"${tag}_eval\"" results/t35_rwku.jsonl 2>/dev/null; then
      if [ ! -f "$ckpt/config.json" ]; then
        python t35_rwku.py train "$method" "$k" > "$LOGDIR/${tag}_train.log" 2>&1
        echo "train $tag exit=$?" >> "$RUNLOG"
      fi
      [ -f "$ckpt/config.json" ] || { echo "FATAL train $tag" >> "$RUNLOG"; continue; }
      python t35_rwku.py eval "$ckpt" "$k" "${tag}_eval" > "$LOGDIR/${tag}_eval.log" 2>&1
      rc=$?
      echo "eval $tag exit=$rc" >> "$RUNLOG"
      [ $rc -eq 0 ] && rm -f "$ckpt"/*.safetensors
    fi
  done
  if [ $((k % 10)) -eq 9 ]; then
    cp results/t35_rwku.jsonl ../reports/remote/t35_rwku.jsonl
    ( cd .. && git add reports/remote/t35_rwku.jsonl \
        && git commit -q -m "rwku scale: through target $k" \
        && git pull --rebase -q origin main && git push -q origin main ) >> "$RUNLOG" 2>&1
  fi
done
cp results/t35_rwku.jsonl ../reports/remote/t35_rwku.jsonl
python t36_rwku_figure.py > "$LOGDIR/t36_3.log" 2>&1
echo "rwku fig exit=$?" >> "$RUNLOG"
( cd .. && git add -A && git commit -q -m "t44 H3: RWKU 50-target scale complete" \
    && git pull --rebase -q origin main && git push -q origin main ) >> "$RUNLOG" 2>&1
echo "T44 COMPLETE" >> "$RUNLOG"
