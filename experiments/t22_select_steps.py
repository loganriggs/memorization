"""T22: pick the per-scope step count by max forget quality (prereg amendment 3).

Reads every calibration FQ log, groups by scope, and selects argmax FQ. Written
as code so the choice is reproducible and auditable rather than eyeballed — the
whole point of the pre-registration is that the selection is executed, not
judged.

Ties: prefer the SHALLOWER step count (less intervention, and less
over-forgetting risk on splits with more forget data).

    python t22_select_steps.py               # report + selection
    python t22_select_steps.py --json out.json
"""
import glob
import json
import os
import re
import sys

LOGDIR = "results/t20_logs"


def collect():
    rows = []
    for f in sorted(glob.glob(f"{LOGDIR}/calib*step*_fq.log")):
        b = os.path.basename(f)[:-len("_fq.log")]
        scope = "min" if "_min_" in b else "all"
        m_step = re.search(r"step(\d+)", b)
        txt = open(f).read()
        m_p = re.search(r"KS stat=[0-9.e-]+ p=([0-9.e-]+)", txt)
        m_tr = re.search(r"mean_ours=([0-9.e+-]+)", txt)
        m_ref = re.search(r"mean_theirs=([0-9.e+-]+)", txt)
        if not (m_step and m_p):
            continue
        rows.append({"scope": scope, "step": int(m_step.group(1)),
                     "fq_p": float(m_p.group(1)),
                     "mean_tr": float(m_tr.group(1)) if m_tr else None,
                     "ref_tr": float(m_ref.group(1)) if m_ref else None})
    # dedupe (scope, step), keeping the last log written
    seen = {}
    for r in rows:
        seen[(r["scope"], r["step"])] = r
    return sorted(seen.values(), key=lambda r: (r["scope"], r["step"]))


def main():
    rows = collect()
    print(f"{'scope':6}{'step':>6}{'mean_TR':>14}{'ref_TR':>9}{'FQ p':>12}")
    for r in rows:
        tr = f"{r['mean_tr']:.4f}" if r["mean_tr"] is not None and r["mean_tr"] < 1e6 else f"{r['mean_tr']:.3e}"
        print(f"{r['scope']:6}{r['step']:>6}{tr:>14}"
              f"{r['ref_tr'] if r['ref_tr'] else 0:>9.4f}{r['fq_p']:>12.6f}")

    sel = {}
    for scope in ("all", "min"):
        cand = [r for r in rows if r["scope"] == scope]
        if not cand:
            print(f"\n{scope}: NO calibration points — cannot select")
            continue
        best = max(cand, key=lambda r: (r["fq_p"], -r["step"]))
        sel[scope] = best
        print(f"\n{scope}: selected step {best['step']} "
              f"(FQ p={best['fq_p']:.6f}, mean TR {best['mean_tr']:.4f} "
              f"vs ref {best['ref_tr']:.4f})")
        if best["fq_p"] < 0.05:
            print(f"  NOTE: best FQ is below the 0.05 admissibility threshold. "
                  f"The grid may yield no admissible cell; per the prereg that "
                  f"is reported, not worked around.")
        edge = min(cand, key=lambda r: r["step"]), max(cand, key=lambda r: r["step"])
        if best["step"] in (edge[0]["step"], edge[1]["step"]):
            print(f"  NOTE: optimum sits at the edge of the scanned range "
                  f"[{edge[0]['step']}, {edge[1]['step']}] — the true optimum "
                  f"may lie outside it; extend before trusting this value.")

    if "--json" in sys.argv:
        out = sys.argv[sys.argv.index("--json") + 1]
        json.dump({"rows": rows, "selected": sel}, open(out, "w"), indent=2)
        print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
