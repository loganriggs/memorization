"""Push a checkpoint to the single shared HF model repo.

All campaign checkpoints live in ONE repo (default loganriggs/memorization-unlearning,
override with HF_CKPT_REPO), separated by path:

    <model>/<split>/<method>/seed<k>/

e.g. phi-1.5/forget05/ours_alltok_g2/seed0/. Uploads by folder path, so nobody
ever has to clone the whole repo -- pull one cell with
    hf download <repo> --include 'phi-1.5/forget05/ours_alltok_g2/seed0/*'

Usage:
    python experiments/hf_push.py results/t17_ours_alltok \
        --model phi-1.5 --split forget05 --method ours_alltok_g2 --seed 0

Skips the upload if that path already carries the same commit marker, so it is
safe to call from a resumable runner. Requires HF_TOKEN with write scope.
"""
import argparse
import json
import os
import subprocess
import sys

def _default_repo():
    """<hf-username>/memorization-unlearning -- the HF account is Elriggs, not the
    GitHub handle, so resolve it from the token rather than hardcoding either."""
    from huggingface_hub import HfApi
    return f"{HfApi().whoami()['name']}/memorization-unlearning"

# Optimizer/scheduler state is large and never needed downstream; evals only load weights.
IGNORE = ["optimizer.pt", "scheduler.pt", "rng_state*", "*.lock", "trainer_state.json"]


def git_sha():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except subprocess.CalledProcessError:
        return "unknown"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("local_dir")
    p.add_argument("--model", required=True)
    p.add_argument("--split", required=True)
    p.add_argument("--method", required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--force", action="store_true")
    a = p.parse_args()

    from huggingface_hub import HfApi
    from huggingface_hub.utils import EntryNotFoundError

    if not os.path.isdir(a.local_dir):
        sys.exit(f"no such checkpoint dir: {a.local_dir}")

    path = f"{a.model}/{a.split}/{a.method}/seed{a.seed}"
    marker = {"src": a.local_dir, "repo_commit": git_sha(),
              "model": a.model, "split": a.split,
              "method": a.method, "seed": a.seed}

    repo = os.environ.get("HF_CKPT_REPO") or _default_repo()
    api = HfApi()
    # Private by default: publishing pre-submission checkpoints is irreversible
    # (scraped/mirrored), flipping private->public later is not. HF_CKPT_PUBLIC=1 opts out.
    api.create_repo(repo, repo_type="model", exist_ok=True,
                    private=os.environ.get("HF_CKPT_PUBLIC") != "1")

    if not a.force:
        try:
            prev = json.loads(open(api.hf_hub_download(
                repo, f"{path}/PUSHED.json"), encoding="utf-8").read())
            if prev.get("repo_commit") == marker["repo_commit"]:
                print(f"skip {path} (already pushed at {prev['repo_commit']})")
                return
        except (EntryNotFoundError, OSError):
            pass

    with open(os.path.join(a.local_dir, "PUSHED.json"), "w", encoding="utf-8") as f:
        json.dump(marker, f, indent=2)

    api.upload_folder(
        repo_id=repo, repo_type="model", folder_path=a.local_dir,
        path_in_repo=path, ignore_patterns=IGNORE,
        commit_message=f"{a.model} {a.split} {a.method} seed{a.seed} @ {marker['repo_commit']}",
    )
    print(f"pushed {a.local_dir} -> {repo}/{path}")


if __name__ == "__main__":
    main()
