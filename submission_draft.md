# Challenge entry draft: a hand-coded bilinear sequence memorizer at 0.70–0.73× trained capacity (and a criterion-gameability result)

*Response to Linsefors & Bushnaq, "Challenge: Hand coding weights for
efficient sequence memorisation." Uses a bilinear hidden layer rather than
the ReLU MLP (explicitly welcomed by the challenge rules: "if you want to go
for something slightly different … we'd still be interested"). All weights
are produced by eigensolves, closed-form solves, reweighting, and greedy
accept-if-better search — no gradient descent or generic black-box
optimizer.*

## Setup

Architecture (parameter-matched to the challenge's 5d² MLP):

```
x = [onehot(t1); onehot(t2)]          (length 2·V_in, V_in = 2d)
h = (Lx) ⊙ (Lx)                       L: (d × 2·V_in)   — symmetric bilinear
logits = D h                          D: (V_out × d), V_out = d
```

Facts are generated exactly as in the challenge (seed 42, labels split
evenly). Evaluation: max n_facts with argmax accuracy ≥ 0.9, binary search
to 2% precision — identical protocol to the trained baselines.

## The algorithm

**Stage 0 — choose the code.** Fix `D = −I`: label c's logit is *minus* its
neuron's squared activation, so the predicted label is the one whose neuron
is **quietest**. Facts are stored as approximate cancellations
`L[c, t1] ≈ −L[c, V_in + t2]` for each fact (t1,t2) of label c. (Why this
and not detectors — see "Why silence," below.)

**Stage 1 — spectral initialization (anti-Rayleigh).** For each label c,
with A_c = mean of xxᵀ over label-c facts and B_c = mean over all other
facts (+εI), take the *smallest* generalized eigenvector of (A_c, B_c),
normalized to vᵀB_c v = 1. This is the spectral relaxation of a 1-D
signed-graph embedding: own-fact edges attract token values toward
cancellation, all other edges repel.

**Stage 2 — fact reweighting.** Re-solve Stage 1 ~40 times, doubling the
weight of currently-misclassified facts in A/B each round (cap 1e6,
renormalize). Keep the best iterate. Pure eigensolves + reweighting.

**Stage 3 — hinge-margin greedy repair.** Margin of fact i:
`min_{c≠y} h_c − h_y`. Objective: Σ min(margin, τ) with τ = 0.3 (near-misses
matter; safe facts don't). Moves, accepted only if (correct-count, hinge)
improves lexicographically: (a) single-entry grid scans over L (25
candidates, ±row-max width); (b) paired moves `L[c,t1] += δ,
L[c,V_in+t2] −= δ` along own-fact edges — these preserve that fact's
response while shifting everything else, exactly the escape direction plain
coordinate moves lack. 12 passes, 8 restarts from Stage-2 ± small noise.
With incremental evaluation (changing one entry shifts one pre-activation
column) a full repair costs seconds up to d=16.

## Results

Max facts at acc ≥ 0.9 (trained baselines from the same harness; "ceiling"
= all 4d² possible facts):

| d | construction | trained (sym bilinear) | ratio |
|---|---|---|---|
| 2–5 | = ceiling | = ceiling | 1.00× |
| 6 | 135 | 144 (ceiling) | 0.94× |
| 8 | 248 (H16 pipeline) | 256 (ceiling) | 0.97× |
| 12 | 387 | 576 (ceiling) | 0.67× |
| **16** | **672** | **960** (< ceiling — uncensored) | **0.70×** |
| 32 | ≈2200 (bracketed: 0.959 @ 2048, 0.868 @ 2458) | 3008 | ≈0.73× |

The d=16 → d=32 ratios (0.70×, 0.73×) indicate the construction matches the
trained scaling exponent with a ≈0.7 prefactor, rather than falling behind
with scale (the failure mode of the original post's construction at
acc = 1.0).

The challenge post's own hand-coded MLP reached ≈0.1× trained at its scales
(larger d; not strictly comparable, but indicative). Accuracy when storing
the *full* ceiling: 1.0 / 1.0 / 1.0 / 0.91 / 0.89 / 0.84 / 0.70 / 0.63 for
d = 2…16.

![construction progression](results/fig_construction.png)

## Why silence (the mechanistic story)

The algorithm embodies an explanation obtained by reverse-engineering
trained models (L1-pruned to minimal readouts, gauge-canonicalized,
10 seeds × d = 2–8):

1. **Trained models choose silence.** Pruned readouts converge to D ≈ −I
   variants (5/6 minimal d=3 seeds): storing facts as *inactivity*, the
   bilinear analog of the challenge post's own hand-coded trick. The
   detector alternative (D=+I, Rayleigh directions) exists but is rare.
2. **Silence is the linear-algebra-friendly code.** "This fact's neuron is
   quiet" is one soft linear condition per fact; "loud" is not.
3. **But exact silence is impossible-by-uselessness.** Each label's own-fact
   graph (t1-tokens vs t2-tokens, edges = facts) is supercritical: its
   exact-null vector is constant-per-component and co-nulls 60–93% of
   *wrong* facts → mass ties. Hence approximate silence with margins, i.e.
   a signed-embedding *ordering* problem — which fixes the construction
   shape: spectral relaxation (Stage 1) + combinatorial rounding-repair
   (Stage 3).
4. **The readout is never the bottleneck.** Frozen random features, ridge
   readouts on constructed features, and alternating D-solves all fail to
   improve anything (three independent tests). Capacity lives entirely in L
   — matching the trained models, where the same asymmetry appears under
   pruning and quantization.
5. **No unique solution exists to copy.** Across seeds, trained sparse
   solutions share only family statistics (≈d/2 signed taps per label,
   occasional zero-tap "default" labels, balanced ±1 excitation/veto) —
   Hungarian-aligned weight similarity is near-chance at d=8. Hand-coding
   must target the family, not a matrix; ours picks the simplest member
   (pure −I silence).

## Bonus: the same recipe on the challenge's ReLU architecture

Porting the pipeline to the challenge's exact folded ReLU MLP (assign S=3
neurons per label as in the original post; `D = −assignment`; per-neuron
ridge initialization toward "≤0 on assigned labels' facts, >0 elsewhere";
same hinge repair with a ReLU forward):

| d=16, acc ≥ 0.9 | max facts |
|---|---|
| their hand-coded (published) | 80–92 |
| **this construction** | **280** |
| their hybrid (hand-coded emb + trained unembed) | ~256–268 |
| trained | 784 |

**3.0–3.5× the original hand-coded construction on its own architecture**
(0.36× trained vs their 0.11×), and on par with their *hybrid* despite
using no gradient descent anywhere. Notably the ReLU version is *harder*
for this pipeline than the bilinear (0.61 vs 0.84 ceiling accuracy at d=8):
one-sided ReLU silence requires all S of a label's neurons to be
simultaneously non-positive per fact, whereas bilinear cancellation is a
single signed condition — evidence that the bilinear layer is not just
equally capable but genuinely more *constructible*, which is part of why we
recommend it as the lens for understanding memorization.

## The certified-insertion variant: every step an explanation

A second construction trades a little capacity for full auditability. It
stores facts *one at a time*, each by an exact algebraic step:

- **Prior ledger.** The spectral init is treated as a batch of
  pre-insertions with certificates: each label's null-space part silences
  its own facts *identically* (verified to 0.0e0 — the own-fact response
  equals the residual's response), so every init-stored fact is "stored
  because residual ≈ 0 and rivals loud," and every init-unstored fact has
  one of two diagnosed causes: (A) own-row residual too loud, or (B) the
  H11 structural tie — the winning rival is exactly co-quiet because the
  fact's tokens lie in one component of the *rival's* own-fact graph
  (empirically ~70%/30% of failures; zero unexplained).
- **Certified insertion.** Along any update direction, every fact's
  rivalry condition is an exact quadratic in the step size; candidate
  steps come from its roots, and a move commits only if the global stored
  count increases. Directions are matched to the ledger cause: own-row
  moves for (A), and for (B) a rival-row least-squares direction
  ((X_ownᵀX_own + εI)⁻¹x — un-quiet the rival with minimal disturbance of
  its own facts). No gradient descent, no blind search.
- **Obstruction ledger.** Facts that cannot be inserted name their
  blockers. The blocker distribution is heavily concentrated: at d=8,
  ~6 "keystone" facts block most failed insertions; at d=16, eight facts
  each block 22-23 of the 273 remaining — the residual capacity gap is a
  small clique of contested token-columns, not diffuse interference.

Ceiling accuracies: d4 0.984, d8 0.859, **d16 0.733 — the best of all our
constructions at d16** (repair pipelines: 0.628-0.66), and above *gradient
descent itself* on the same frozen-D architecture (0.686). On the challenge
metric (max facts at acc ≥ 0.9), certified insertion reaches **785 at d16 —
0.82× trained** (repair pipelines: 672), and at d32 stores ≥2867 —
0.95× of trained (3008) — with the ratios *rising* with scale as the tie
supply grows. **These certified numbers are criterion-gameability results,
not robust-storage results** — see the criterion note: at any margin
floor they collapse, and the fair capacity headline is the repair
pipelines' 0.70–0.73×. The storage
of every fact — and the failure of every non-stored fact — has a
human-readable certificate. Three rendered ledger entries (d = 8):

> **Stored by init** — fact (10,10)→4: own-neuron response = residual
> response (null part contributes exactly 0); quietest rival: label 1.
> **Inserted** — fact (13,0)→7: CE-gradient direction, exact step chosen
> from the roots of 31 affected facts' rivalry quadratics; one stored fact
> sacrificed, net +1.
> **Obstructed** — fact (3,10)→7: every storing step breaks exactly one
> fact — (1,10)→7, the *same label sharing the t2 = 10 column*; the two
> compete for a single tie (blocker margin ~1e-30).

All certificate quantities live at tie scale (1e-15 … 1e-30) — see the
robustness caveat below; the certified construction wins by exact
tie-placement, not by margin engineering.

**Criterion note (integrity).** At the d=16 capacity point, 784 of 785
stored facts lie within ±1e-9 of exact ties: at *any* margin requirement
(even 1e-12) the certified capacity collapses to ~0. The number 785 is
valid under the challenge's argmax criterion as literally specified, but
it is a statement about favorable placement of float64 round-off, not
about robust storage (and may not reproduce bit-identically across BLAS
builds). We report it (a) because it is criterion-legal, and (b) because
it demonstrates something the challenge authors may want to know: **the
argmax-accuracy criterion without a margin floor is gameable by
tie-placement**, and a margin- or quantization-robustness requirement
(e.g. "accuracy must survive weight noise sigma, or rounding to k bits")
would separate trained solutions from tie-constructions sharply — trained
models pass such a test easily (6-bit robustness), tie-constructions not
at all. Our robust-construction numbers (the repair pipelines: 0.70-0.73x
trained, thin-but-real margins) are the fair comparison under a
margin-corrected criterion.

*Robustness caveat (important).* The methods form a three-tier
capacity-robustness hierarchy (d=8, fraction of facts retained when a
margin slack γ is demanded):

| method | acc (γ=0) | γ=0.01 | γ=0.1 | median margin |
|---|---|---|---|---|
| certified insertion | 0.855 | 0.000 | 0.000 | ~1e-30 (exact ties) |
| greedy repair (hinge) | 0.805 | 0.648 | 0.234 | 0.043 |
| GD, D=−I frozen | 0.898 | ~0.88 | 0.859 | 1.69 |

The certified method's capacity edge is bought entirely at the zero-slack
extreme — legal under the argmax criterion, destroyed by any weight
perturbation (trained solutions tolerate ~6-bit quantization). This is
intrinsic, not fixable by post-hoc scaling: a fact's margin is a min over
rivals, and the giant own-fact-graph components pin at least one rival to
an exact tie for essentially every fact (retrofit attempts refuted
empirically). Robust margins require globally breaking component
structure — which is precisely what gradient descent pays ~10% capacity
to do. Part of the certified method's edge over same-architecture GD is
therefore criterion arbitrage, not pure algorithmic superiority.

## Limitations and open problems

- The gap to 1.0× decomposes cleanly (d=16 capacities): free-D trained 960
  → **GD with D frozen to −I: 864** → this construction: 672. So the
  silence-code restriction itself costs only ~10% (the architecture holds
  90% of free capacity under GD — Stage 0 is nearly free), while the
  remaining ~22% is pure *optimizer*: gradient descent moves all weights
  against all facts simultaneously; greedy repair does not. Every variant
  we tried (shared-neuron tap graphs, solved readouts, trained-D-graph
  imports, asymmetric L≠R at 1.8× params) lands within ±0.05 of the same
  ceiling accuracy, and failure forensics show no localized structure to
  exploit. Closing the optimizer gap without GD is the interesting open
  problem — candidate directions: coupled block moves, message-passing on
  the fact graph, LP/SDP relaxations of the ordering constraints.
- Scaling exponent not yet measured beyond d=16 (runtime of repair grows
  ~d⁴; incremental evaluation mitigates but d=32+ needs more engineering).
- The construction is for the bilinear variant; porting the silence +
  signed-embedding story back to the ReLU MLP (where trained models
  *also* store facts in activation patterns with engineered zeros) is
  untested.

## Reproduction

`capacity.py` (data + protocol, exact copy of the challenge repo's),
`h12b_repair.py` (Stages 1–2), `h12c_fast.py` (Stage 3),
`h12c_capacity.py` (capacity search; results in
`results/handcoded_h12c.jsonl`). Full discovery trail: `research_log.md`;
supporting analyses in `results.md` and `tiny_models/`.
