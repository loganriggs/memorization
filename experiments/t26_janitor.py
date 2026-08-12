"""T26: disk janitor — delete local checkpoints that are verified on HF.

The grid + baselines produce ~36 x 2.4 GB checkpoints against ~83 GB free, so
local copies must go once safely uploaded. A checkpoint is deleted only when
ALL of:
  1. its expected HF path lists a model safetensors file (verified via API,
     not inferred from the push log),
  2. the local dir has been quiet for >15 min (not the active training dir),
  3. it is not on the KEEP list (calibration depths needed for relearn).

Runs as a loop; safe to kill any time. Never touches anything outside
experiments/results/t20_forget05_* and t23_forget05_*.
"""
import os
import re
import time

from huggingface_hub import HfApi

REPO = "Elriggs/memorization-unlearning"
RESULTS = "/workspace/memorization/experiments/results"
KEEP = {"t20_forget05_all_g2_s0_step100", "t20_forget05_min_g2_s0_step450"}
QUIET_S = 15 * 60


def hf_path(dirname):
    m = re.match(r"t20_forget05_(all|min)_g([0-9.]+)_s(\d)$", dirname)
    if m:
        return f"llama3.2-1b/forget05/ours_{m.group(1)}_g{m.group(2)}/seed{m.group(3)}"
    m = re.match(r"t23_forget05_([a-z]+)_s(\d)$", dirname)
    if m:
        return f"llama3.2-1b/forget05/{m.group(1)}/seed{m.group(2)}"
    return None


def main():
    api = HfApi()
    while True:
        try:
            files = set(api.list_repo_files(REPO))
        except Exception as e:  # noqa: BLE001 — transient API failure, retry next pass
            print(f"HF list failed ({type(e).__name__}), retrying later", flush=True)
            time.sleep(600)
            continue
        for d in sorted(os.listdir(RESULTS)):
            full = os.path.join(RESULTS, d)
            if d in KEEP or not os.path.isdir(full):
                continue
            hp = hf_path(d)
            if hp is None:
                continue
            uploaded = any(f.startswith(hp + "/") and f.endswith(".safetensors")
                           for f in files)
            quiet = time.time() - os.path.getmtime(full) > QUIET_S
            if uploaded and quiet:
                # Delete only the weight shards. config.json/tokenizer/PUSHED
                # stay, so a restarted runner's skip-guards (train: config.json
                # exists; push: PUSHED marker) still hold and nothing retrains.
                sz = 0
                for r, _, fs in os.walk(full):
                    for f in fs:
                        if f.endswith(".safetensors"):
                            fp = os.path.join(r, f)
                            sz += os.path.getsize(fp)
                            os.remove(fp)
                if sz:
                    print(f"reclaimed {sz/1e9:.1f} GB: {d} (verified at {hp})",
                          flush=True)
        time.sleep(600)


if __name__ == "__main__":
    main()
