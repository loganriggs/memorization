#!/bin/bash
# Relearn curves: selected config (seed 0) + retain95 never-knew control,
# both at lr 1e-5 and 5e-5 (t18: relearning is lr-fragile, one lr is not
# evidence). Selected checkpoint is re-fetched from HF because the janitor
# strips local weight shards after verified upload.
set -u
cd "$(dirname "$0")"
source /venv/main/bin/activate

SEL=../reports/remote/t20_selection.json
[ -f "$SEL" ] || { echo "no selection file"; exit 2; }
SCOPE=$(python -c "import json;print(json.load(open('$SEL'))['selected']['scope'])")
GAMMA=$(python -c "import json;g=json.load(open('$SEL'))['selected']['gamma'];print(f'{g:g}')")
LOGDIR=results/t25_logs
mkdir -p "$LOGDIR"

CKPT=results/relearn_selected
if [ ! -f "$CKPT/config.json" ]; then
  python - "$SCOPE" "$GAMMA" "$CKPT" <<'PYEOF' > "$LOGDIR/fetch.log" 2>&1
import sys
from huggingface_hub import snapshot_download
scope, gamma, dst = sys.argv[1:4]
sub = f"llama3.2-1b/forget05/ours_{scope}_g{gamma}/seed0"
p = snapshot_download("Elriggs/memorization-unlearning",
                      allow_patterns=[f"{sub}/*"])
import shutil, os
shutil.copytree(os.path.join(p, sub), dst, dirs_exist_ok=True)
print("fetched", sub, "->", dst)
PYEOF
  echo "fetch exit=$?" >> "$LOGDIR/relearn.log"
fi

for lr in 1e-5 5e-5; do
  for spec in "$CKPT:selected_${SCOPE}_g${GAMMA}" \
              "open-unlearning/tofu_Llama-3.2-1B-Instruct_retain95:control_retain95"; do
    src="${spec%%:*}"; name="${spec##*:}"
    tag="t25_${name}_lr${lr}"
    if grep -q "\"tag\": \"$tag\"" results/t25_relearn.jsonl 2>/dev/null; then
      echo "skip $tag (recorded)" >> "$LOGDIR/relearn.log"; continue
    fi
    python t25_relearn.py run "$src" "$tag" "$lr" forget05 \
      > "$LOGDIR/${tag}.log" 2>&1
    echo "relearn $tag exit=$?" >> "$LOGDIR/relearn.log"
    ( cd .. && git add reports/remote/t25_relearn.jsonl \
        && git commit -q -m "relearn: $tag" && git pull --rebase -q origin main \
        && git push -q origin main ) >> "$LOGDIR/relearn.log" 2>&1
  done
done
echo "RELEARN COMPLETE" >> "$LOGDIR/relearn.log"
