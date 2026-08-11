# Program 2: Do the toy-model memorization findings transfer to real data?

*Follow-up to the sequence-memorization challenge work (`results.md`,
`submission_draft.md`). Question: which of the toy bilinear findings —
silence codes, max-margin/λ structure, the tension web, triage economics,
certified editing — survive contact with real image data and depth?
All experiments: `p2_*.py`, logs in `results/p2_*.log`.*

## What transfers

1. **The silence code and its capacity.** `D = −I` ("quietest neuron wins")
   memorizes random-labeled images with striking efficiency and near-identical
   capacity on MNIST and SVHN: m=10 neurons hold ~1–2k random labels
   (1.00 at n=1024, ~0.96–0.97 at 2048, collapsing by 4096 on *both*
   datasets, despite 4× different input dims). Free-D with m=32 holds ≥8k
   on MNIST.
2. **Triage economics (structure vs memorization).** Below capacity there
   is *no competition*: models memorize all noise and all structure, and
   held-out accuracy degrades smoothly with noise fraction — no sharp
   flip, no generalize-first-then-memorize dynamics (train and held-out
   accuracy rise together by epoch ~3k, then both freeze). At **capacity
   saturation** the spectrum appears: the model keeps 65–82% of
   clean-label images vs 16–34% of random-label ones (MNIST, 3 seeds;
   SVHN: 66% vs 30%) — structure is cheap (one rule covers thousands of
   images), facts are expensive, and the model buys structure first.
3. **The margin discriminator.** Within one saturated model, clean-kept
   facts carry ~2.3× the normalized margin of random-kept facts (all
   seeds): memorized facts hug the decision boundary (support-like,
   dedicated storage); structured facts inherit interpolation slack. The
   toy's free/support geometry, transposed.
4. **Where facts live (amended): wherever the widest trainable input
   space is — not "the first layer" per se.** Initial probes suggested a
   first-layer law, but were confounded by parameter/width asymmetry
   (caught by the user). Deconfounded experiments: (a) with matched
   trainable budgets but L2 reading a 30-dim bottleneck, learn-L1 (0.90)
   ≫ learn-L2 (0.26) — input dimensionality, not parameters, is the
   medium; (b) in a d_model architecture (embedding + two identical
   64×64 bilinear blocks), freezing EITHER block alone costs nothing
   (1.0/1.0; both frozen: 0.20) — storage is opportunistic across
   width-matched layers, with or without a residual stream. The robust
   law: memorization capacity ∝ trainable input-dimensionality of a
   layer; "facts live early" in real models only insofar as early layers
   read the widest/least-compressed representations.
5. **Editing asymmetry — in softened form.** On a saturated mixed-label
   model, certified rank-1 forgetting removes a memorized fact at ~35%
   lower collateral than a structured fact (44 vs 68 broken siblings of
   ~5,600, at 0.06% weight change). Direction as predicted.

## What does not transfer

6. **Zero-collateral surgery — refined.** It dies on saturated MNIST but
   NOT because of dense inputs per se: on a dense-but-uncorrelated toy
   (unit-sphere inputs, below capacity), zero-collateral rank-1 forgetting
   works at both early and late training checkpoints (late costs ~1.9×
   the weight change — high margins raise the price, not the
   feasibility). The killers are (a) SATURATION (thousands of
   near-zero-margin facts to avoid) and (b) input CORRELATION (on MNIST
   the vulnerable facts' images span the target's image, so
   orthogonalized directions can't reach it). Real-model unlearning is
   damage-minimization when models are saturated and features correlated
   — which is the typical regime.
7. **Fast max-margin convergence.** The support-set pileup that made λ
   extraction exact in the toy emerges on MNIST but ~10× slower
   (support fraction 0.032 after 100k epochs vs 0.19–0.62 in the toy).
   Margin-based attribution on real models needs very long training or
   explicit margin-seeking finetuning.
8. **Sharp phase transitions.** No flip point in the noise fraction; the
   structure→memorization transition is graceful and driven entirely by
   the capacity budget, not by dataset composition per se.

## Corrections recorded along the way

- A 1-layer m=10 silence model has a genuine *capability* ceiling on digit
  structure (~50–58% held-out even with clean labels); free-D m=32 reaches
  ~92%. Architecture choice, not label noise, explained part of the early
  "spectrum collapse."
- The naive 2-layer stack fails at chance, but NOT for the conjectured
  signedness reason (all-positive inter-layer activations memorize fine
  with Adam + normalization): it was optimization. Residual and RMSNorm
  are neither necessary nor harmful for capacity here.
- One SVHN free-D run failed at chance under the MNIST recipe
  (lr/conditioning artifact; not rerun).

## Practical takeaways

- For attribution/editing on real models: target the **first-layer input
  weights**; expect **collateral-cost gradients** rather than clean
  excisions; use **margins** to distinguish memorized from structural
  storage; train with plain SGD and long horizons if you want the
  max-margin λ machinery to bite.
- The unlearning asymmetry (memorized facts cheaper to remove than
  structured ones — and *free-riding* facts impossible to remove by data
  deletion) is architecture-robust and should be expected in real models.
