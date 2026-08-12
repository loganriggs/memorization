"""T24: apply the frozen selection rule to the completed forget05 grid.

Mechanical execution of PREREGISTRATION.md section 2 (frozen at 96f8fec,
amendments 1-3). No judgment enters here; that is the point.

  1. Aggregate per (scope, gamma) across seeds: mean and range (max-min).
  2. Admissible iff mean FQ p > 0.05.
  3. Among admissible: argmin mean forget-leakage ROUGE.
  4. Tie-break (within one range-width of leakage): higher mean utility.
  5. Tie-break 2: smaller gamma.
  6. If none admissible: report the null. Per the declared fallback (mailbox
     2026-08-12 04:20, unobjected), also emit the argmax-FQ config labelled
     INADMISSIBLE_FALLBACK so forget01/10 can still be run and clearly marked.

Writes reports/remote/t20_selection.json and prints the aggregate table.
Refuses to run if any (scope, gamma) has fewer than 3 seeds, unless --partial.
"""
import json
import sys
from collections import defaultdict

SWEEP = "../reports/remote/t20_forget05_sweep.jsonl"
OUT = "../reports/remote/t20_selection.json"
SEEDS = 3


def main():
    rows = [json.loads(l) for l in open(SWEEP)]
    cells = defaultdict(list)
    for r in rows:
        c = r["cell"].replace("t20_forget05_", "")
        scope, gamma, _seed = c.split("_")
        cells[(scope, float(gamma[1:]))].append(r)

    agg = []
    incomplete = []
    for (scope, gamma), rs in sorted(cells.items()):
        if len(rs) < SEEDS:
            incomplete.append((scope, gamma, len(rs)))
        def stat(key):
            vs = [x[key] if key != "fq" else (x.get("fq_p_vs_retain95") or 0.0)
                  for x in rs]
            return sum(vs) / len(vs), max(vs) - min(vs)
        fq_m, fq_r = stat("fq")
        lk_m, lk_r = stat("forget_rouge")
        ut_m, ut_r = stat("model_utility")
        agg.append({"scope": scope, "gamma": gamma, "n_seeds": len(rs),
                    "fq_mean": fq_m, "fq_range": fq_r,
                    "leak_mean": lk_m, "leak_range": lk_r,
                    "util_mean": ut_m, "util_range": ut_r,
                    "admissible": fq_m > 0.05})

    print(f"{'scope':6}{'gamma':>6}{'n':>3}{'FQ mean±rng':>18}"
          f"{'leak mean±rng':>18}{'util mean±rng':>18}{'adm':>5}")
    for a in agg:
        print(f"{a['scope']:6}{a['gamma']:>6}{a['n_seeds']:>3}"
              f"{a['fq_mean']:>11.4f}±{a['fq_range']:<6.4f}"
              f"{a['leak_mean']:>11.4f}±{a['leak_range']:<6.4f}"
              f"{a['util_mean']:>11.4f}±{a['util_range']:<6.4f}"
              f"{'YES' if a['admissible'] else 'no':>5}")

    if incomplete and "--partial" not in sys.argv:
        print(f"\nINCOMPLETE: {incomplete} — refusing to select "
              f"(pass --partial to preview)")
        sys.exit(2)

    admissible = [a for a in agg if a["admissible"]]
    result = {"aggregate": agg, "rule": "prereg 96f8fec + amendments 1-3"}
    if admissible:
        best = min(admissible, key=lambda a: a["leak_mean"])
        near = [a for a in admissible
                if a["leak_mean"] - best["leak_mean"] <= best["leak_range"]]
        if len(near) > 1:
            best = max(near, key=lambda a: (a["util_mean"], -a["gamma"]))
        result["selected"] = best
        result["status"] = "ADMISSIBLE_SELECTION"
        print(f"\nSELECTED: {best['scope']} gamma={best['gamma']} "
              f"(admissible, min leakage)")
    else:
        fb = max(agg, key=lambda a: a["fq_mean"])
        result["selected"] = fb
        result["status"] = "INADMISSIBLE_FALLBACK"
        print(f"\nNO ADMISSIBLE CELL (best mean FQ "
              f"{max(a['fq_mean'] for a in agg):.4f} <= 0.05).")
        print(f"Fallback per declared rule: argmax-FQ = {fb['scope']} "
              f"gamma={fb['gamma']} — forget01/10 run with this config, "
              f"labelled INADMISSIBLE_FALLBACK; the null is the headline.")

    if "--partial" not in sys.argv:
        json.dump(result, open(OUT, "w"), indent=2)
        print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
