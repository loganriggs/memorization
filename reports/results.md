# Bilinear layers on the sequence-memorization challenge

*Replication and extension of [Linsefors & Bushnaq, "Challenge: Hand coding weights for efficient sequence memorisation"](https://www.lesswrong.com/) (July 2026), using their code at [LindaLinsefors/Memory-Toy-Models](https://github.com/LindaLinsefors/Memory-Toy-Models) as the reference (cloned at `ref_repo/`).*

## TL;DR

1. **Their trained-MLP baseline replicates almost exactly** under their published protocol (same architecture, optimizer recipe, seeds-any criterion, binary search). At acc ≥ 0.9 we get 784 / 2528 / 8320 / 27648 facts for d = 16/32/64/128 vs their published 784 / 2560 / 8320 / 27648.
2. **A parameter-matched bilinear layer `y = D(Lx ⊙ Rx)` memorizes more facts than the ReLU MLP at every size**: +13–25% at acc ≥ 0.9 and +16–70% at acc = 1.0. **SwiGLU performs the same or slightly better than pure bilinear**, so the bilinear layer is not handicapped by removing the gate nonlinearity.
3. The bilinear advantage is a **prefactor effect that shrinks with scale** (×1.25 at d=16 → ×1.13 at d=128 for acc ≥ 0.9); fitted scaling exponents are similar across architectures (~d^1.65–1.7 over this range). The gap is largest at acc = 1.0 and small d, where ReLU-MLP training struggles to nail every last fact.
4. Bilinear layers also **memorize much faster**: at d=128 with 24,576 facts, the bilinear net first hits 100% train accuracy in ~250 epochs vs ~1,750 for the MLP.
5. First hand-coding attempt (**CP/ALS decomposition of the fact tensor → bilinear weights**) is a clean **negative result**: capacity ≈ hidden width m (~1 fact per rank-1 component), a factor of ~200 below the trained bilinear at the same width. Least-squares reconstruction of the 0/1 fact tensor is the wrong objective — trained models only need argmax *margins* and tolerate enormous interference.
6. **Random frozen bilinear features are no better than random ReLU features** (actually worse per parameter). The entire bilinear advantage comes from *learning the input matrices* L, R — not from generically better quadratic features.
7. Mechanistically, the trained bilinear layer stores facts in **dense superposition**: no zero activations, per-label interaction matrices with effective rank ~20 out of a possible 36 (at d=32), and ~7–8 effective neurons contributing per fact — remarkably similar concentration to the MLP despite having barely half as many neurons.

---

## Setup

### Task and protocol (theirs, replicated exactly)

Facts are random pairs of input tokens mapped to random labels; there is nothing to generalize, only memorize. The challenge architecture ("Figure 4" of the post) folds the MLP into the embeddings/unembedding:

```
x_enc = [onehot(t1) ; onehot(t2)]        # (2·V_in,)
MLP:      h = relu(W x_enc)              # W: (m, 2·V_in)
logits = D h                             # D: (V_out, m)
```

with `V_in = 2d`, `V_out = d`, `m = d`, giving `5d²` parameters. Protocol (from `hc2_full_train_capacity_search.py` in their repo, which produced the "trained" curve in their Figure 5):

- facts generated with seed 42; labels split evenly across the output vocab
- 11 training attempts with init seeds 0–10, weights `U(±1/√fan_in)`
- full-batch cross-entropy, Adam lr = 1e-2, ≤ 5000 epochs, early stop at accuracy 1.0 or 100 epochs without improving the best accuracy; score = best accuracy seen
- a fact count *passes* if **any** of the 11 attempts reaches the threshold (0.9 or 1.0)
- binary search over `n_facts ∈ [1, 4d²]` until `hi − lo < 0.02·hi`

### Extensions (ours)

Two new hidden layers, everything else identical:

```
bilinear:  h = (L x_enc) ⊙ (R x_enc)         # L, R: (m, 2·V_in)
SwiGLU:    h = silu(L x_enc) ⊙ (R x_enc)
```

These have `9dm` parameters, so we set `m = round(5d/9)` (9, 18, 36, 71 for d = 16, 32, 64, 128) to match the MLP's `5d²` within 0.5%.

Known deviations from their runs, none of which should matter statistically: facts drawn with a CPU RNG (their GPU runs used a CUDA RNG — their own footnote notes the two permute differently at the same seed, so bit-identical fact sets were never on the table); all 11 seeds trained simultaneously as a batched tensor dimension with per-seed early-stopping bookkeeping; weight-draw order within a seed differs. Code: `capacity.py`, `run_capacity.py`.

---

## Result 1: Capacity scaling — bilinear > MLP at equal parameter count

![Capacity scaling](results/fig_scaling.png)

Max facts memorized (any of 11 seeds, binary search to 2% precision):

| arch | thr | d=16 | d=32 | d=64 | d=128 |
|---|---|---|---|---|---|
| MLP (ReLU), ours | 0.9 | 784 | 2528 | 8320 | 27648 |
| **their trained MLP (published)** | 0.9 | **784** | **2560** | **8320** | **27648** |
| Bilinear | 0.9 | 976 | 3072 | 9600 | 31232 |
| SwiGLU | 0.9 | 992 | 3136 | 9984 | 32256 |
| MLP (ReLU), ours | 1.0 | 536 | 2240 | 7680 | 26112 |
| **their trained MLP (published)** | 1.0 | **568** | **2176** | **7680** | **25600** |
| Bilinear | 1.0 | 912 | 2880 | 9216 | 30208 |
| SwiGLU | 1.0 | 848 | 2816 | 9472 | 30720 |

The MLP row replicating their published numbers to within binary-search precision validates the harness; the bilinear/SwiGLU comparison is then apples-to-apples.

In bits stored per parameter (`n_facts · log₂(V_out) / params`, acc ≥ 0.9): the MLP holds steady at ~2.4 bits/param across sizes, bilinear starts at ~3.0 and declines to ~2.7, SwiGLU ~3.1 → 2.8. Two readings are possible: the gating architectures may just have a better prefactor that fades into a common asymptote at scale, or they may genuinely hold a modest (~10–15%) edge at large d — d=256 runs would help distinguish these.

Two secondary observations:

- **acc = 1.0 is where ReLU suffers most.** At d=16 the bilinear stores 70% more facts at perfect accuracy. The post noted their hand-coded construction also degrades specifically at acc = 1.0; apparently even *trained* ReLU MLPs find the last few facts disproportionately expensive, while multiplicative gating does not.
- **Bilinear memorizes far faster.** Epochs for the best seed to first reach 100% accuracy at d=128: n=16384: MLP 202 / bilinear 119 / SwiGLU 166; n=24576: MLP 1745 / bilinear 247 / SwiGLU 409; n=28672: MLP >5000 (never) / bilinear 625 / SwiGLU 959.

Since SwiGLU ≈ bilinear throughout, the pure bilinear layer is *not* "bad at memorizing" — if anything the swish gate adds little, which is convenient: the bilinear layer is the analytically tractable one (its logits are an exact quadratic form in the inputs).

## Result 2: The advantage is in the learned input weights, not the feature type

The post's "rand-emb" baseline freezes random input weights and trains only the readout. Repeating that for both feature types (plus a fully closed-form ridge-regression readout — no gradient descent anywhere), at acc ≥ 0.9:

![Random features](results/fig_randfeat.png)

| readout | feats | d=16 | d=32 | d=64 | d=128 |
|---|---|---|---|---|---|
| Adam (their rand-emb) | ReLU | 40 | 184 | 800 | 2592 |
| *their published* | ReLU | *36* | *196* | *800* | *2624* |
| Adam | bilinear | 18 | 48 | 140 | 432 |
| ridge (closed form) | ReLU | 40 | 116 | 328 | 1008 |
| ridge (closed form) | bilinear | 24 | 64 | 208 | 664 |

With frozen random input weights, bilinear features are *worse* than ReLU features — partly because param-matching gives them ~half the neurons, but even per neuron they underperform (d=64: 140/36 ≈ 3.9 facts/neuron vs 800/64 = 12.5). Random quadratic features are not magic. **All of the bilinear layer's capacity advantage is created by training L and R.** Any hand-coded bilinear construction therefore has to encode facts into the input matrices; a clever readout on top of generic features won't get there (consistent with the post finding hybrid ≫ rand-emb).

(Ridge vs Adam readouts: ridge is better on bilinear features, worse on ReLU features — Adam-CE with early stopping finds better margins than one-shot least squares when features are sparse and nonnegative.)

## Result 3: Hand-coding attempt #1 — CP decomposition — fails informatively

A width-m bilinear layer can exactly implement a rank-m CP (PARAFAC) decomposition of the `(V_in, V_in, V_out)` fact tensor: put input factor `a_n` in L's position-1 block, `b_n` in R's position-2 block (zeros elsewhere), output factor in D. Their repo even contains a CP-fitting side experiment (`bilinear.py`), never wired into a model. So: fit CP with ALS to the 0/1 fact tensor, load into bilinear weights (`handcoded_cp.py`), measure capacity:

| | d=16 (m=9) | d=32 (m=18) | d=64 (m=36) |
|---|---|---|---|
| CP hand-coded, acc ≥ 0.9 | 10 | 16 | 40 |
| trained bilinear, acc ≥ 0.9 | 976 | 3072 | 9600 |

Capacity ≈ m: ALS effectively spends one rank-1 component per fact and finds no shared structure — unsurprising in hindsight, since random facts give a max-rank-like tensor and least squares penalizes the *reconstruction* error everywhere. The trained model doesn't reconstruct the tensor at all: it only needs the correct logit to be *largest*, and happily accepts huge interference everywhere else. **The compression lives in the gap between "reconstruct the lookup table" and "win every argmax."** A viable hand-coded bilinear construction should optimize margins (e.g. a hinge/perceptron-style condition on `e₁ᵀ M_c e₂`), not squared error — that's the concrete next step.

## Result 4: How the trained bilinear layer stores facts

Models analyzed at d=32, each trained at its own acc ≥ 0.9 capacity (MLP: n=2528, m=32; bilinear: n=3072, m=18; best seed of 11). Code: `analysis.py`.

![Weight distributions](results/fig_weights.png)

Weights are smooth, wide, unimodal in all matrices — like the post's trained MLPs (their Figure 6, third row) and unlike their trimodal hand-coded {−1, 0, +1} solution. Nothing about the bilinear solution looks discrete.

![Activations and contribution concentration](results/fig_activations.png)

- The MLP runs at ~51% exactly-zero activations; the bilinear has **zero** zeros — signed, heavy-tailed, roughly symmetric activations. Whatever storage scheme it uses, it is not "silence as a checkable condition" (the trick the post's hand-coded construction is built on) — it must instead rely on signed cancellation in the readout.
- Despite having 18 neurons vs the MLP's 32, the bilinear layer uses about the **same effective number of neurons per fact** (participation ratio of contributions to the correct logit, ~7–8), i.e. each bilinear neuron participates in *more* facts — denser superposition.

![Per-label interaction spectra](results/fig_spectra.png)

Because the bilinear layer is an exact quadratic form, each label's logit decomposes into a cross-position interaction matrix `M_c[a,b] = Σₙ D[c,n]·(L1[n,a]R2[n,b] + R1[n,a]L2[n,b])` (rank ≤ 2m) plus per-position terms. The trained spectra decay smoothly and fill most of the available rank (effective rank ~20 of 36; no sharp cutoff, no small set of dominant directions), and the heatmap shows `M_c` as noise-like with the label's fact cells sitting disproportionately in the positive tail. This is a distributed, margin-based code: each fact is a slightly-above-interference bump in a dense random-looking quadratic form. That is *also* what makes it hand-codable in principle — choosing `M_c` directly is a margin-satisfaction problem over low-rank matrices, which feels much more tractable than reasoning about ReLU activation patterns.

---

## Files

| file | purpose |
|---|---|
| `capacity.py` | data gen (their algorithm), batched 11-seed trainer, binary search, grid cache |
| `run_capacity.py` | driver for the main sweep (`results/capacity.jsonl`) |
| `randfeat.py` | frozen-random-feature capacities, Adam + ridge readouts (`results/randfeat.jsonl`) |
| `handcoded_cp.py` | CP/ALS hand-coded bilinear (`results/handcoded_cp.jsonl`) |
| `analysis.py` | trains the d=32 analysis models, makes figs 2–4 (`results/analysis_models.pt`) |
| `plots.py` | scaling + random-feature figures and the capacity table |
| `ref_repo/` | clone of their code (reference protocol + published result JSONs) |

Reproduce everything with `.venv/bin/python run_capacity.py && .venv/bin/python randfeat.py && .venv/bin/python handcoded_cp.py && .venv/bin/python analysis.py && .venv/bin/python plots.py` (~25 min total on an RTX 5080).

## Update: hand-coding the bilinear layer (research-loop results)

An automated research loop (see `research_log.md` for the full trail)
reverse-engineered the sparse trained solutions and built a fully non-GD
construction. Key findings:

1. **Trained models use a silence code.** Pruned to minimal readouts, most
   seeds converge to `D ≈ −I` variants: a label wins by its neuron being
   *quietest*, with facts stored as approximate cancellations
   `L[n,t1] ≈ −L[n,V+t2]` — the bilinear analog of the post's hand-coded
   "inactive neurons" trick. Across a 10-seed fleet (d=2–8), solutions share
   *family statistics* (≈d/2 signed readout taps per label, frequent
   zero-tap "default" labels, balanced ±1 excitation/veto pairs) but not
   weights (Hungarian-aligned cosine similarity only 0.28–0.66) — the target
   of hand-coding is a solution family, not a matrix.
2. **Exact silence is provably useless** — each label's own-fact graph is one
   giant component whose exact-null vector also nulls 60–93% of *wrong*
   facts (mass ties). Each neuron's real problem is a 1-D signed-graph
   embedding (own edges attract to cancellation, others repel), whose
   spectral relaxation is a generalized eigenproblem.
3. **The construction** (all challenge-legal: eigensolves, reweighting,
   greedy accept-if-better; no gradient descent):
   anti-Rayleigh silence directions → iterative fact reweighting →
   hinge-margin greedy repair with cancellation-preserving paired moves.
   Accuracy at the *full* 4d² ceiling: 1.0 / 1.0 / 1.0 / 0.91 / 0.89 /
   0.84 / 0.70 / 0.63 for d = 2/3/4/5/6/8/12/16. Max-facts at acc ≥ 0.9
   (binary search, same protocol as everything else): d6 **135/144 (94% of
   ceiling)**, d8 200/256, d12 387/576, and at d16 — where the trained
   model no longer saturates its ceiling, making the comparison honest —
   **672 vs trained 960: 0.70× trained capacity**. The post's hand-coded
   MLP reached ≈0.1× at its scales.
4. A shared-neuron circulant tap-graph variant (each neuron silences its own
   label and excites its neighbor) reaches 1.0 at the d=4 ceiling; larger
   sizes in progress.

![construction progression](results/fig_construction.png)

## Suggested next steps

1. **Margin-based hand-coding for bilinear**: choose per-label low-rank `M_c` to satisfy `M_y(t1,t2) > M_c(t1,t2) + γ` for all wrong c — a perceptron/hinge feasibility problem; solvable rank-constrained via alternating least squares on hinge-active constraints (still "just least squares" in the challenge's spirit) or via a random-projection + linear-solve scheme à la Dugan et al.
2. **d=256 runs** to pin down whether the bilinear advantage asymptotes away (the ×1.25 → ×1.13 trend suggests it might).
3. **Equal-width comparison** (bilinear m=d, 9/5× params) to separate "per-parameter" from "per-neuron" capacity.
4. **Interpret the trained bilinear directly**: with the exact quadratic form available, per-fact attribution is closed-form; clustering the rank-1 neuron components `D[c,n]·L_n⊗R_n` across labels might reveal the sharing scheme.
