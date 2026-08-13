"""T31: post-campaign analysis — utility component breakdown with tuned NPO,
and the anatomy of forget quality (truth-ratio ECDFs vs the reference).

Answers three questions from Logan (2026-08-13):
  1. Did full-retain (v2, all 3,800 rows) help utility?  (no — shown per component)
  2. Where exactly does tuned NPO keep utility that ours loses?
  3. What is "perfect forgetting" under FQ, and how far from it is each method?

No training; renders entirely from stored eval records and truth-ratio dumps.
Writes reports/remote/fig_utility_components_v2.png/.svg and
reports/remote/fig_fq_anatomy.png/.svg.
"""
import glob
import json

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp

RR = "../reports/remote"
TRD = "results/t15_truthratios"
EVAL_DS = "/workspace/.hf_home/hub/datasets--open-unlearning--eval/snapshots/*"

EVALS = {r["tag"]: r for l in open("results/t15_metrics.jsonl")
         for r in [json.loads(l)] if r.get("stage") == "eval"}


def tr_component(vals):
    """TOFU utility's truth-ratio term: mean max(0, 1 - 1/R)."""
    r = np.asarray(vals, dtype=np.float64)
    return float(np.mean(np.maximum(0.0, 1.0 - 1.0 / (r + 1e-10))))


def components(tags):
    """9 utility components, averaged over the given eval tags (seeds)."""
    out = {}
    for s in ("retain", "real_authors", "world_facts"):
        probs, rouges, trs = [], [], []
        for t in tags:
            probs.append(EVALS[t][f"{s}_prob"])
            rouges.append(EVALS[t][f"{s}_rouge"])
            trs.append(tr_component(
                json.load(open(f"{TRD}/{t}.json"))[s]))
        out[f"{s}/prob"] = np.mean(probs)
        out[f"{s}/rouge"] = np.mean(rouges)
        out[f"{s}/tr"] = np.mean(trs)
    return out


def published_ref_components():
    return json.load(open(f"{RR}/utility_components.json"))["retain95_ref"]


GROUPS = {
    "retain95 reference": None,  # filled from utility_components.json
    "ours v1 (retain cap 400)": [f"t20_forget05_min_g4_s{s}_eval" for s in range(3)],
    "ours v2 (full 3,800 retain)": [f"t20_forget05_min_g4_s{s}_fullretain_eval" for s in range(3)],
    "NPO published (lr 1e-5)": [f"t23_forget05_npo_s{s}_eval" for s in range(3)],
    "NPO tuned (lr 2e-5)": [f"t23_forget05_npo_lr2e-05_s{s}_eval" for s in range(3)],
}
COLORS = ["#666666", "#d62728", "#f4a3a3", "#b8a2d8", "#9467bd"]


def fig_components():
    data = {}
    for name, tags in GROUPS.items():
        data[name] = published_ref_components() if tags is None else components(tags)
    comps = list(next(iter(data.values())))
    x = np.arange(len(comps))
    w = 0.16
    fig, ax = plt.subplots(figsize=(11.5, 4.8), dpi=150)
    for i, (name, d) in enumerate(data.items()):
        ax.bar(x + (i - 2) * w, [d[c] for c in comps], w, label=name,
               color=COLORS[i])
    ax.set_xticks(x)
    ax.set_xticklabels(comps, rotation=20, ha="right", fontsize=8.5)
    ax.set_ylabel("component value (utility = harmonic mean of all 9)")
    ax.set_title("Utility components, forget05 (3-seed means) — where each "
                 "method loses utility")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    for ext in ("png", "svg"):
        fig.savefig(f"{RR}/fig_utility_components_v2.{ext}")
    print("wrote fig_utility_components_v2")
    return data


def published_forget_trs(model_dirname, split="forget05"):
    pats = [f"{EVAL_DS}/{model_dirname}/evals_{split}/TOFU_EVAL.json",
            f"{EVAL_DS}/{model_dirname}/TOFU_EVAL.json"]
    fs = [f for pat in pats for f in glob.glob(pat)]
    j = json.load(open(fs[0]))
    return np.array([v["score"] for v in
                     j["forget_truth_ratio"]["value_by_index"].values()])


def fig_anatomy():
    ref = published_forget_trs("tofu_Llama-3.2-1B-Instruct_retain95")
    full = published_forget_trs("tofu_Llama-3.2-1B-Instruct_full")
    curves = {
        "retain95 reference (perfect forgetting, p=1)": ("#333333", "-", ref),
        "full model (never unlearned)": ("#999999", ":", full),
    }
    methods = {
        "ours min-token γ4": ("#d62728", "-",
            [f"t20_forget05_min_g4_s{s}_eval" for s in range(3)]),
        "NPO tuned lr 2e-5": ("#9467bd", "-",
            [f"t23_forget05_npo_lr2e-05_s{s}_eval" for s in range(3)]),
        "GA published 10ep (lobotomy)": ("#2ca02c", "--",
            [f"t23_forget05_gradascent_s{s}_eval" for s in range(3)]),
    }
    fig, (a, b) = plt.subplots(1, 2, figsize=(12.6, 4.9), dpi=150)
    for name, (c, ls, vals) in curves.items():
        xs = np.sort(vals)
        a.plot(xs, np.linspace(0, 1, len(xs)), color=c, ls=ls, lw=1.8,
               label=name)
    print("KS D (per seed) vs retain95 reference, and pooled:")
    for name, (c, ls, tags) in methods.items():
        pooled = []
        ds = []
        for t in tags:
            ours = 1.0 / (np.array(json.load(open(f"{TRD}/{t}.json"))["forget"])
                          + 1e-10)
            pooled.append(ours)
            ds.append(ks_2samp(ours, ref).statistic)
        pooled = np.concatenate(pooled)
        xs = np.sort(pooled)
        a.plot(xs, np.linspace(0, 1, len(xs)), color=c, ls=ls, lw=1.8,
               label=f"{name} (D̄={np.mean(ds):.3f})")
        print(f"  {name:32s} D = {['%.3f' % d for d in ds]}  "
              f"pooled n={len(pooled)}")
        b.hist(pooled, bins=np.linspace(0, 2.0, 41), density=True, alpha=0.45,
               color=c, label=name)
    b.hist(ref, bins=np.linspace(0, 2.0, 41), density=True, histtype="step",
           color="#333", lw=2.0, label="retain95 reference")
    a.set_xlim(0, 2.0)
    a.set_xlabel("truth ratio (their convention: wrong/correct; higher = "
                 "more 'never knew')")
    a.set_ylabel("ECDF over forget05")
    a.set_title("A — what the KS test compares: forget-set truth-ratio ECDFs\n"
                "(FQ = p-value of max vertical gap D vs the black curve)")
    a.legend(fontsize=7.5, loc="lower right")
    a.grid(alpha=0.25)
    b.set_xlabel("truth ratio (wrong/correct)")
    b.set_ylabel("density")
    b.set_title("B — same data as histograms (seeds pooled)")
    b.legend(fontsize=7.5)
    fig.tight_layout()
    for ext in ("png", "svg"):
        fig.savefig(f"{RR}/fig_fq_anatomy.{ext}")
    print("wrote fig_fq_anatomy")


if __name__ == "__main__":
    data = fig_components()
    for n, d in data.items():
        hm = len(d) / sum(1.0 / max(v, 1e-9) for v in d.values())
        print(f"{n:32s} harmonic mean = {hm:.4f}")
    fig_anatomy()
