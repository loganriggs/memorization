"""Stage 1: prove this rental can train before we spend GPU-days on it.

Three checks, each fatal:
  1. sm_120 kernels actually exist in this torch build. A cu124 wheel imports
     fine and reports cuda available, then dies on the first real op with
     "no kernel image is available" -- so we run a matmul AND a backward.
  2. bf16 + grad-checkpointing path works (what t14_phi.py uses).
  3. Sustained-load stability. The local box had a failing CPU core that cost a
     day; a rental can be just as sick and we'd rather find out in 5 minutes.

Writes results/env_check.json. Exit code is the signal -- do not pipe this
through tail/grep, it masks crashes.
"""
import json
import os
import sys
import time

import torch

OUT = "results/env_check.json"
STRESS_SECONDS = int(os.environ.get("STRESS_SECONDS", "300"))


def fail(msg):
    print(f"FAIL: {msg}", flush=True)
    sys.exit(1)


def main():
    os.makedirs("results", exist_ok=True)
    r = {"torch": torch.__version__, "cuda_build": torch.version.cuda}

    if not torch.cuda.is_available():
        fail("torch.cuda.is_available() is False")
    dev = torch.device("cuda")
    cap = torch.cuda.get_device_capability()
    r["gpu"] = torch.cuda.get_device_name(0)
    r["capability"] = f"{cap[0]}.{cap[1]}"
    r["vram_gb"] = round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1)
    print(f"{r['gpu']} cc{r['capability']} {r['vram_gb']}GB | torch {r['torch']} cu{r['cuda_build']}", flush=True)

    # 1. matmul + backward -- the actual arch test
    try:
        a = torch.randn(4096, 4096, device=dev, requires_grad=True)
        b = torch.randn(4096, 4096, device=dev)
        (a @ b).sum().backward()
        torch.cuda.synchronize()
        assert a.grad is not None and torch.isfinite(a.grad).all()
    except Exception as e:  # noqa: BLE001 -- any failure here is disqualifying
        fail(f"fp32 matmul+backward: {type(e).__name__}: {e}")
    r["matmul_backward"] = "ok"

    # 2. bf16 + grad checkpointing, the t14_phi.py training path
    try:
        from torch.utils.checkpoint import checkpoint
        lin = torch.nn.Linear(2048, 2048, device=dev, dtype=torch.bfloat16)
        x = torch.randn(64, 2048, device=dev, dtype=torch.bfloat16, requires_grad=True)
        checkpoint(lambda t: lin(t).relu(), x, use_reentrant=False).sum().backward()
        torch.cuda.synchronize()
    except Exception as e:  # noqa: BLE001
        fail(f"bf16 + grad checkpoint: {type(e).__name__}: {e}")
    r["bf16_checkpoint"] = "ok"

    # 3. sustained load -- looking for a crash or a nondeterministic result,
    #    not for throughput. Same inputs every iteration, so any drift is hardware.
    torch.manual_seed(0)
    x = torch.randn(2048, 2048, device=dev)
    w = torch.randn(2048, 2048, device=dev, requires_grad=True)
    ref, iters, t0 = None, 0, time.time()
    while time.time() - t0 < STRESS_SECONDS:
        for _ in range(50):
            out = (x @ w).tanh()
            loss = out.square().mean()
            w.grad = None
            loss.backward()
            iters += 1
        torch.cuda.synchronize()
        v = float(loss)
        if not (v == v):  # NaN
            fail(f"NaN loss under load at iter {iters}")
        if ref is None:
            ref = v
        elif abs(v - ref) > 1e-3 * max(1.0, abs(ref)):
            fail(f"nondeterministic result under load: {v} vs {ref} at iter {iters} "
                 "-- suspect bad GPU/RAM, do not train on this box")
        print(f"  stress {int(time.time()-t0)}s / {STRESS_SECONDS}s, {iters} iters, loss {v:.6f}", flush=True)
    r["stress_seconds"] = STRESS_SECONDS
    r["stress_iters"] = iters
    r["stress"] = "ok"

    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(r, f, indent=2)
    print(f"PASS -- wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
