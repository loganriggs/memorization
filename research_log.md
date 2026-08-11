# Research log: reverse-engineering the sparse bilinear memorizer

Goal: find THE solution family for sym-bilinear D-only-sparse memorizers
(d=2..8), then write an algorithm that selects weights with great accuracy and
generalizes across d. Ultimate test = hand-coded construction matching
trained capacity.

## Established facts (from earlier sessions)
- Arch: logits = D((Lx)⊙(Lx)), x = [onehot(t1); onehot(t2)], V_in=2d, V_out=d,
  m=d. Facts: n=4d² random pairs→labels (seed 42), saturated for d≤12.
- Gauge freedoms: whole-row sign of L; per-D-column positive scale (fold √s
  into L row). Canonical form: D columns max-|entry|=1, then leftover D
  entries are irreducible.
- D-only pruning frontiers (acc≥0.9): d2: 2 taps [1,1]; d3: 4 [2,1,1];
  d4: 6 [2,2,0,2]; d5: 10 [1,2,4,1,2]; d8: 27 [3,2,2,5,2,2,8,3].
- Motifs seen: (a) per-label detector (d2: D=I, logit_c = h_c);
  (b) excitation+veto chain (d4: label c = +0.4·h_c − 1.0·h_{c+1}, canonical);
  (c) default label (all-zero D row wins when others vetoed);
  (d) mostly-negative D at high sparsity ("least vetoed wins").
- Elementwise |L| breaks models (signed cancellation is load-bearing);
  cross-term 2ab in (a+b)² is where pair interaction lives.
- Weight quantization: ~6 bits/weight lossless; L more sensitive than D.

## Hypotheses
- [x] H1 (RESOLVED: partial fail — detectors decay with d): each label's neuron = top generalized
      eigenvector of (A_c, B_c), A_c = Σ_{pairs with label c} xxᵀ,
      B_c = Σ_{other pairs} xxᵀ + εI, normalized vᵀB_c v = 1; D = I.
      Predict argmax (v_c·x)². STATUS: testing now.
- [x] H2 (RESOLVED: refuted at weight level; family-level universal): 10 seeds of train+prune converge, after
      canonicalization + Hungarian neuron matching, to one cluster per d
      (or a small number of motif-clusters). STATUS: fleet launched.
- [x] H3 (RESOLVED: motif census confirms taps/label ≈ d/2, mixed signs): the excitation/veto chain
      appears iff taps-per-label < ~2; with ≥2 taps/label, per-label
      detectors dominate. Check across the 10-seed fleet.
- [x] H4 (RESOLVED: ridge D never helps — readout not bottleneck): if H1 underperforms, add veto structure:
      D from small L1-regularized linear solve on fixed h features (ridge on
      squares is still closed-form-ish).
- [x] H5 (RESOLVED: failures uniform, no hotspots — see task 6): for any construction that fails, the gradient
      of loss wrt L at the constructed point shows which constraint family is
      binding (per-fact margin gradients cluster by label?).
- [x] H6 (RESOLVED: modular excitations, shared load-bearing vetoes): thanks to sparse D, the model
      factors into label-subset discriminators (e.g. d4 chain = 3 pairwise
      splits + default). Test: ablate one D tap → which fact subsets die.
- [x] H7 (RESOLVED: 0.4 not a constant; balanced ±1 is the mode): in canonical D the d4
      chain is +0.4·h_c − 1.0·h_{c+1}. Check whether ~0.4 (or other specific
      ratios) recurs across the 10-seed fleet and across d; if so, derive it
      (margin-balance between "my excitation beats your veto" inequalities?).
      Ratio census = part of fleet analysis.

## Task queue (keep ordered; add, never silently drop)
1. [x] Analyze multiseed fleet: done (see notes; H2 refuted at weight level,
       family-level structure universal).
2. [x] H1 Rayleigh test d2..8: done (partial fail).
3. [x] Scale calibration subsumed by ridge-D test (no gain).
4. [x] OBSOLETE — d2 solved exactly by constructions (H1b/H9b reach 1.0).
5. [x] H6 ablation done (cron #10): excitation taps are perfectly
       label-local (kill exactly own label's facts); veto taps are shared
       infrastructure — every veto ablation kills the default label
       entirely plus collateral. Chain decomposes asymmetrically. Written
       into tiny_models/sym_random/d4.md.
6. [x] Gradient forensics done (cron #11, forensics.py +
       results/fig_forensics_d8.png). KEY NEGATIVE: failures of the
       constructed d8 model are spread UNIFORMLY over tokens (4-10 failed
       facts per token, no hotspots) and gradient pressure is small and
       uniform across neurons — no localized structure for block moves to
       exploit. Strengthens the "global credit assignment" diagnosis; token-
       or neuron-targeted heuristics are unpromising. (Run used 1 repair,
       acc 0.562 — structure conclusions, not capacity numbers.)
20. [x] DONE (cron #12/#13): ReLU-MLP port of the silence construction (challenge's actual
        architecture): D=−(assignment matrix), ReLU(-side) silence needs
        pre-activations ≤ 0 on own facts — a LINEAR feasibility problem
        (one-sided!), potentially EASIER than bilinear cancellation.
        Next iteration.
21. [x] d32 scaling DONE (bracket): capacity ≈2200 vs trained 3008 →
        0.73×, matching d16's 0.70× — constant prefactor, same exponent.
        Written into submission_draft.md. (d64+ would need GPU-ized repair;
        optional.)
7. [x] Capacity test done (task 19): 0.70× trained at d16 uncensored.
8. [x] Asym variant tested: +0.04 for 1.8× params — not the lever.
9. [x] OBSOLETE — superseded by the silence-code theory (H11).
10. [x] Writeup: results.md section + submission_draft.md written
        (cron #9). Remaining polish: measure scaling exponent d>16 if
        engineering budget allows; port story to ReLU MLP.
11. [x] H1b reweighted Rayleigh: done (helps d≤4 only; d2 hits 1.0).
12. [x] OBSOLETE (d3 solved at 1.0 by H12b): error analysis of H1/H9 at d3: list missed facts, check token
        overlap; does the miss-set match trained models' smallest-margin
        facts?
13. [x] Compare trained L to Rayleigh: done — found the two-family split
        (detector vs silence); silence is modal.
14. [x] H9b reweighted anti-Rayleigh: done (SOTA spectral construction).
15. [x] H11 cycle-structure analysis: done (supercritical; exact silence
        ties out; it's a signed-embedding ordering problem).
17. [x] H12b improved repair: done — breakthrough (see notes).
18. [x] H13 tested (cron #7): circulant tap graph loses to plain D=−I + repair; not the lever.
19. [x] DONE (cron #7 + notification): capacity binary search of
        H12b/H12c construction — d6 135, d8 200, d12 387, d16 672
        (0.70× trained uncensored). results/handcoded_h12c.jsonl.
16. [x] H10 SUBSUMED by H13/H13a (shared-neuron tap graphs tested;
        circulant variant loses to plain D=−I + repair).

## PROGRAM 2: TRANSFER TO REAL (user directive, July 25)
Goal: take the toy findings (silence codes, max-margin object, tension
web, λ-attribution, certified editing) toward real models, both ways.

### Task queue P2 (ordered; never silently drop)
P2-1. [x] Editing sweep DONE (p2_edit_scale.py, results/
      p2_edit_scale.jsonl): 9/10 facts across margin quantiles at d4/d8
      are locally editable — earlier "SGD web-locked" was pick-specific.
      Patterns: (a) edit cost RISES with margin (free-ness is a gradient:
      m=6-11 facts cost 0.3-4%; m=37-65 facts need joint-4 or fail);
      (b) joint-4 space always dominates 2-param (cheaper + unlocks);
      (c) BELOW capacity (d8 n=200) editing is much cheaper (0.3-0.7%)
      than at-capacity — the taut-web lock is a capacity-regime effect.
      Remaining sub-item queued as P2-1c': at-capacity d8/d16 sweep +
      6-8 param spaces for the hardest facts.
P2-2. [~] Beachhead DONE (p2_mnist.py): the silence code TRANSFERS to
      real images — D=−I with only m=10 neurons memorizes 1024
      random-labeled MNIST images at 100% (capacity ~2048 at 97%;
      collapses by 4096). Free-D m=32 memorizes all 8192 tested at 1.0.
      Continuous dense inputs are far more capacity-efficient per param
      than one-hot tokens. Support-set fractions ≈ 0 at 8k epochs —
      likely UNDER-TRAINED for max-margin structure (toy needed 60k+);
      REMAINING: long-train run for margin pileup + λ extraction on
      MNIST; interference/editing story without token locality (dense
      gradients — every fact couples to every fact).
P2-3. [x] DONE — triage + margins + editing all transferred (see notes):
      triage 2.5-4.5×; margin discriminator 2.3×; editing asymmetry
      survives as COLLATERAL-COST GRADIENT (44 vs 68 broken siblings for
      memorized vs structured fact at 0.06% |Δ|) but zero-collateral
      surgery DIES with dense inputs (token-locality was load-bearing).
P2-3-old. [~] TRIAGE CONFIRMED in the capacity-limited regime
      (p2_saturation.log): saturated negI-m10 keeps 81.5% of clean vs
      32.5% of random facts at n=8192/p=.25 (58-75% vs 16-17% at 16k) —
      structure is cheap, facts expensive, model triages as toy
      economics predicted. Unsaturated models (freeD-m32) never triage
      (1.0/1.0). Anomalies to resolve: negI 16k p=.5 BETTER than p=.25
      (triage sharpens with hopelessness? seed noise?); freeD 16k p=.5
      optimization collapse (lr). RUNNING: 3-seed margin analysis of the
      four subsets (clean/rand × kept/dropped) on the triage model
      (p2_triage_margins.log) — the fact-level discriminator.
P2-4. [~] Ladder rungs (b,c,d) substantially DONE: 2-layer works with
      Adam + inter-layer unit-norm (signedness hypothesis refuted;
      resid/RMSNorm neither needed nor harmful for capacity). LAYER
      ROLES (decisive): frozen-L1 collapses to 0.21; frozen-L2 keeps
      0.95; frozen-D keeps 1.00. Noise: L1 fragile (0.58 @ σ_rel=.03),
      L2/D nearly immune (≥0.99 @ σ=.1). ⇒ MEMORIZATION LIVES IN THE
      FIRST LAYER — the toy law ("capacity in L, never the readout")
      survives depth and strengthens: everything downstream is a nearly
      random-replaceable mixer. Matches toy rand-feat + LLM fact-storage
      lore. Remaining: (a) multi-token toy variant (low priority).
P2-5. [x] SVHN DONE: silence-code capacity replicates almost exactly
      (1.0/0.96/0.68 at n=1k/2k/4k, matching MNIST despite 4x input dims);
      triage replicates (clean 0.66 vs rand 0.30). One artifact: freeD
      m=32 failed at chance under the MNIST recipe (lr/conditioning; not
      a capacity result; rerun with Adam if ever needed).
P2-6. [x] DONE: p2_transfer_report.md — what transfers (silence capacity,
      triage economics, margin discriminator, first-layer storage, soft
      editing asymmetry), what doesn't (zero-collateral surgery, fast
      margin convergence, sharp flips), corrections, practical takeaways.

## PROGRAM 3: DENSE TOY ("spherical cow", user directive)
Bridge between one-hot toy and real images: random unit-sphere inputs
(dim 64), random labels, negI m=10. Questions: (1) capacity scaling on
dense inputs; (2) margin/tension convergence speed (tests the
input-correlation hypothesis for MNIST's slowness); (3) editing before
vs after margin convergence (early = slack, late = taut?) with plain +
orthogonalized (should work here — overlaps ~1/8) + functional-proximal
methods; (4) USER IDEA: functional distance from weights alone — for
quadratic nets the function IS the forms M_c = Σ D_cn l_n l_nᵀ, so
model similarity = generalized cos over {M_c} is closed-form; test
whether ||ΔM|| predicts edit collateral better than ||ΔL|| → "precise
edit = proximal in function space" + edit-and-reinsert guided by the
pre-edit model as spec (weights-only fact recovery).
Launched: p3_dense_toy.log (capacity sweep, margin trajectory n=256,
early-vs-late editing with functional-distance logging).

### P3 queue (cron 1a17e2ec)
P3-A. [x] DONE (round 2, p3_refined.log). VERDICTS:
      (1) SATURATION IS THE PRIMARY SURGERY-KILLER, correlation
      secondary: n=256 → 2/2 zero-collateral at BOTH c=0 and c=1;
      n=768 (acc 0.75) → 0/2 at both, best collateral 8 (c=0) vs 9-22
      (c=1 — the aggravator); n=1024 mixed (1/2 at c=0). MNIST's
      surgery death was mostly crowding, partly correlation.
      (2) FUNCTIONAL DISTANCE VALIDATED IN THE RIGHT REGIME: overall
      Spearman tied (wd .925 / fd .919), BUT within narrow weight-norm
      bands fd still ranks collateral at ρ=0.90 (smallest band),
      decaying for large edits [0.90, 0.65, 0.40, 0.18, 0.06] — among
      same-size small edits, ||Δsym(M)|| carries real predictive info
      beyond ||ΔL||. Raw min-fd selection is noisy (picked coll 5 vs
      oracle 1); the METHOD (P3-D) should minimize fd within a small-wd
      constraint, not raw fd.
      Round 1 notes:
      (1) surgery SURVIVED all correlation levels (2/2 at mean overlap
      up to 0.30, MNIST-like) at moderate load → correlation alone
      insufficient; SATURATION is the prime suspect. Missing cells
      launched: saturated toy ± correlation (n 256/768/1024 × c 0/1).
      (2) fd-vs-wd INCONCLUSIVE: within rank-1-fixed-direction
      candidates the metrics are monotone in each other (both ρ≈0.96-
      0.99) — no separation possible. Direction-diverse candidate test
      launched (random rows/directions/ranks; fd-within-wd-band
      analysis + selection comparison). Asym symmetrization machinery
      validated (works identically).
P3-B. [x] DONE (p3_dose.log, 0.015-0.03; 0.04/0.05 runs dropped): the
      unlearning dial is a SMOOTH monotone trade, not a threshold:
      rand_acc 0.60/0.53/0.46/0.35 and heldout 0.72/0.73/0.77/0.79 as
      σ_rand rises 0.015→0.03 (with σ_clean=0.1). Each increment of
      per-example noise converts memorized facts into structure capacity
      — a continuous differential-forgetting knob.
P3-C. [x] DONE (p3_dmodel_sat.log): at saturation storage REBALANCES —
      frozen-L1 0.45 / frozen-L2 0.54 at n=8192 (vs 1.0/1.0 unsaturated);
      same pattern at 16k/32k. Both width-matched layers recruited under
      pressure; mild (~8pt) earlier-layer preference. Full story:
      opportunistic under slack, distributed at saturation.
P3-D. [x] DONE (p3_method.log, 4 facts, crowded n=768): PROXIMAL
      SELECTION IS A 10× WIN — within-budget min-fd (26.5) and min-wd
      (25.25) both land near-oracle (23.0) vs random-feasible 258.5
      mean collateral. BUT min-fd ≈ min-wd: the functional metric adds
      no selection advantage over plain weight-proximity in this edit
      family (minimum-norm feasible edits are already minimum-functional
      ones). fd's unique value is FORECASTING (within-band collateral
      ranking ρ=0.90), not selection. Verdict on the conjecture:
      proximity-optimized editing fully validated; the specific metric
      (tensor-sim vs norm) is a wash for rank-1 edits.
P3-E. [x] DONE: p3_report.md (6 findings + corrections).

## Session notes (newest first)
- POST-P3 (user Qs — honesty audit + validations launched,
  p3_real_validation.log): (a) admitted gap: the 10×-proximal /
  ρ=0.90-forecaster results are TOY-only; MNIST editing evidence was
  2 facts + crude search. Launched: budgeted proximal recipe on MNIST,
  10 facts across margin quantiles, memorized-vs-clean oracle split.
  (b) admitted: the 99.9% same-digit entanglement result used PIXEL
  overlap — certifies data geometry, not the model. Launched
  dissociation test: model-space J (margin-gradient kernel) vs pixel
  overlap, correlated against TRUE digits vs MODEL labels, separately
  for clean and random-label groups. Prediction: on the random group,
  model-J follows model labels not true digits (entanglement = the
  model's functional organization → usable as general feature-finder
  via co-vulnerability clustering); pixel overlap keeps following true
  digits there.
- PROGRAM 3 COMPLETE (all P3 items [x]); p3_report.md written; cron
  1a17e2ec stopped. Three programs now closed: challenge work,
  certified-insertion/explanation, transfer + editing science.
- ENTANGLEMENT RESULTS (user Qs answered):
  (1) UNSUPERVISED COLLATERAL PREDICTION WORKS: raw input overlap
  |x_i·x_k| predicts per-fact break-rates under real edits at
  ρ = 0.82-0.87 (dense toy, crowded n=768); overlap×fragility adds
  nothing (overlap dominates). The entanglement map is a genuine
  pre-edit forecast, computable without touching the model.
  (2) MNIST top-entanglement is ~PERFECTLY semantic: same-true-digit
  rate among top-3000 overlap pairs = 0.999 vs base 0.101. Stronger
  than predicted — on simple visual classes, the strongest
  representational entanglements ARE semantic pairs (while the
  random-label toy proves the notions are separable in principle; and
  lower-overlap entanglement includes cross-digit geometry).
- P3 (user Qs): ENTANGLEMENT test launched (p3_entangle.log):
  (a) dense toy n=768 crowded — do unsupervised entanglement scores
  (input overlap; overlap×fragility "J-pred") predict per-fact
  break-rates under many real edits? (b) MNIST — is entanglement
  semantic: same-true-digit rate among top-overlap pairs vs base rate.
  Conceptual notes: entanglement = representational geometry (overlap ×
  rival structure), NOT semantics per se — random-label toy has zero
  semantic content; natural data correlates geometry with semantics
  partially. Weights-only entanglement structure = eigenframes of the
  forms M_c (facts sharing near-null eigendirections co-move under
  edits) — data needed only to NAME facts, not for the partition.
  P3-D design confirmed with user: minimize ||Δsym(M)|| s.t. forget AND
  ||ΔL|| ≤ budget (fd informative only within small-wd bands).
- TRIPLE RESULTS (P3 + noise-grid + d_model):
  P3 DENSE TOY: capacity ~512-700 facts at dim=64/m=10 (~1 fact/param —
  best medium yet); margin pileup FAST (support 0.19 @10k vs MNIST 0.03
  @100k) → MNIST slowness = input CORRELATION confirmed; editing:
  ZERO-COLLATERAL SURGERY WORKS on dense uncorrelated inputs at both
  early AND late checkpoints (n=256 < capacity; late edit costs 1.9×
  more |ΔL| — margins raise price but not feasibility). ⇒ MNIST surgery
  death = SATURATION + CORRELATION, not density per se.
  NOISE 2-GROUP (user experiment): sharp dose threshold — σ_rand=0.01
  harmless; σ_rand=0.03 (under clean-aug) SELECTIVELY unlearns the
  memorized group (0.71→0.35) and REALLOCATES capacity to structure
  (heldout 0.69→0.79); equal noise 0.1/0.1 wipes memorization (0.12)
  and lifts heldout to 0.92 ≈ pure-clean ceiling. Ball-storage
  robustification real but modest (eval-noise acc 0.23→0.39 at
  σ_rand-only 0.03). Noise = a differential unlearning dial.
  D_MODEL LAYERS (user architecture): with width-matched 64×64 blocks,
  frozen-L1 → 1.0 AND frozen-L2 → 1.0 (both-frozen 0.20) — NO depth
  privilege; storage is OPPORTUNISTIC. Reconciled law: the storage
  medium is a layer's INPUT DIMENSIONALITY × trainability, not its
  depth position (layer-fair's L2 read 30-dim → crippled; here both
  read 64-dim → equal). "First-layer law" holds only because layer 1
  typically reads the widest space. Residual stream: same pattern.
  Transfer report amended.
- THREAD C (user's architecture fix): d_model design launched
  (p2_dmodel.log) — embed 784→64 + two IDENTICAL 64×64 bilinear blocks
  + readout; the widened-L2 pathology avoided; layers param- AND
  width-matched by construction. Probes: frozen-random-embed configs
  (pure depth-position test: freeze L1 vs L2 vs D vs both), ±residual
  stream (the standing open question), and trainable-embed noise probes
  (does storage migrate into the widest-input map E?). n=2048 random
  labels, Adam + inter-layer unit-norm.
- THREAD B (user idea): two-group noise grid launched
  (p2_noise2group.log): σ_clean ∈ {0, 0.1} × σ_rand ∈ {0, 0.01, 0.03}
  (+ equal-noise 0.1/0.1 control). Questions: does SMALL noise on the
  memorized group (a) robustify those facts (point→ball storage, forced
  input-space margins — measured via eval-noise-0.05 accuracy), (b)
  overload them (a ball costs more capacity than a point), (c) protect
  against the clean-stream erosion effect (1.0→0.71 at σ_rand=0)?
  σ_rand-only conditions isolate the effect from clean-augmentation.
- RESULTS TRIO (see chat): ortho-edit fails on MNIST (correlated inputs:
  vulnerable set spans the target image — residual 0.05-0.27); LAYER-FAIR
  deconfounds the first-layer law (matched trainable budgets: learn-L1
  0.897 vs learn-L2 0.263 — input-space width is the medium, params are
  not; caveat: all-free widened model trains badly 0.36); AUGMENTATION
  σ=0.1: heldout ↑ (0.678→0.692), margin ratio widens 1.8×→6.5× ✓✓, but
  fixed random group UNEXPECTEDLY erodes (1.0→0.71 — fresh-noise
  gradients regularize away stored facts); σ=0.2 over-regularizes;
  shift-aug runs diverged (artifact).
- POST-P2 (user caught a CONFOUND in the layer-role result): the probed
  2-layer had L1 = 30×784 = 23,520 params vs L2 = 30×30 = 900 (same
  width, 26× param asymmetry) — so frozen-L1 collapse & frozen-L2
  retention were largely predictable from PARAMETER SHARE, not depth
  position. The honest claim is only: storage medium = each layer's
  INPUT space (784-dim vs 30-dim). LAUNCHED param-matched rerun
  (p2_layer_fair.log): L2 widened to 784×30 = 23,520 (= L1); frozen +
  noise probes. If facts still concentrate in L1 → first-layer default
  is real; if storage spreads/flips → previous result was param share.
- POST-P2 (user idea): AUGMENTATION experiment launched
  (results/p2_augment.log) — clean group augmented each epoch (pixel noise
  σ∈{0.1,0.2}, ±2px shifts, both) so exact clean inputs never repeat;
  random-label group FIXED (memorizable). freeD m=32, over-parameterized
  regime. Predictions: (a) heldout ↑ vs static baseline (memorizing
  clean instances is useless → forced rule-learning); (b) margin ratio
  clean/rand widens beyond the static 2.3× (clean = pure structure,
  rand = pure memorization — cleanest separation yet); (c) rand group
  still memorized at 1.0 (fixed points remain cheap).
- PROGRAM 2 COMPLETE (P2-cron final): SVHN replication in (P2-5 [x]) and
  p2_transfer_report.md written (P2-6 [x]). All queue items closed.
  Program-2 headline: the toy's storage laws transfer (capacity of the
  silence code is even dataset-invariant; facts live in layer 1 at any
  depth; triage + margin discriminators are real), while the toy's
  SURGICAL affordances do not (locality was load-bearing). Cron can be
  stopped.
- ITERATION (P2-cron): layer-role results logged into P2-4 (memorization
  lives in the first layer; downstream nearly random-replaceable).
  P2-5 LAUNCHED (p2_svhn.log): SVHN replication — random-label capacity
  (negI m=10, freeD m=32) + triage at saturation (n=16k, p=.25).
  After SVHN: P2-6 writeup is the last queue item.
- ITERATION (P2-cron, depth rung 1 RESOLVED — my diagnosis was WRONG):
  ALL variants succeed with Adam (0.98-1.0 at n=4k/8k), INCLUDING plain
  all-positive h1. The signedness/centering hypothesis is REFUTED as the
  blocker: unsigned inter-layer activations memorize fine. The rung-1
  failure was optimization (SGD lr=1.0 on a composed-quartic landscape)
  plus missing inter-layer scale normalization — with unit-norm h1 and
  Adam, depth works, full stop. (Whether signed variants store
  DIFFERENTLY remains open but is not a capacity question.) SGD lr
  sweep for depth still unexplored — SGD-cleanliness of λ geometry at
  depth is an open question since Adam is required-ish here.
  NEXT (launched): layer-role probes on working 2-layer (Adam): per-layer
  noise sensitivity + frozen-L1/frozen-L2 retrains — where do facts live?
- ITERATION (P2-cron, depth rung 1): naive 2-layer bilinear FAILS at
  chance (0.10) in ALL configs (±resid ±rms, every n) while param-
  matched 1-layer memorizes 8192 at 1.0. DIAGNOSIS (structural, not just
  lr): symmetric squares emit ALL-POSITIVE h1 — the second layer gets
  unsigned mean-dominated inputs, and bilinear storage requires SIGNED
  cancellation (the toy's core mechanism); plus vanishing scale under
  composition. Depth needs signedness restored between layers. LAUNCHED
  fix sweep (p2_depth2.log): (a) center h1 (LayerNorm-style), (b)
  asymmetric signed layer-1 (Lx)(Rx), (c) plain+unit-scale control,
  each × SGD/Adam. If centering/asym rescue depth, that's a clean
  mechanistic account of why norm layers with CENTERING matter for
  stacked multiplicative architectures.
- ITERATION (P2-cron): real-data editing collected (see P2-3 [x]);
  LAUNCHED P2-4 rung 1 (results/p2_depth.log): param-matched 1-layer
  (m=32) vs 2-layer bilinear (30/30) ± residual ± RMSNorm on
  random-label MNIST, n ∈ {4k, 8k, 16k}: does depth add memorization
  capacity per param? Next rungs: layer-role probes (where do facts
  live — per-layer editing sensitivity, freeze/reinit), then toy-side
  multi-token.
- ITERATION (P2-cron): 3-SEED TRIAGE MARGINS in — the fact-level
  discriminator EXISTS on real data: clean-kept facts carry ~2.3× the
  margin of random-kept facts, consistent across all seeds (2.8e-5 vs
  1.2e-5; 6.4e-6 vs 3.4e-6; 3.0e-5 vs 1.4e-5). Memorized facts hug the
  boundary (support-like); structured facts ride interpolation slack —
  the toy free/support geometry transposed to MNIST. Triage ratio
  seed-stable (clean 65-82% vs rand 31-34%); seed variance is large
  enough that the 16k p=.5 anomaly is plausibly noise. P2-3 essentially
  done pending the editing asymmetry. LAUNCHED: real-data editing test
  on the saved triage model — rank-1 (delta ⊗ x_k) 10-param certified
  forget of a median rand-kept vs clean-kept fact, zero-collateral and
  min-collateral variants (results/p2_edit_real.log). Prediction:
  rand-kept editable, clean-kept locked by the shared rule.
- ITERATION (P2, flip-sweep results): FOUR findings.
  1. CAPABILITY control: negI-m10 has a genuine structure ceiling
     (heldout 0.49-0.58 even pure-clean, SGD≈Adam — not bad optimization)
     while freeD-m32 CAN do digit structure (0.92-0.93). Earlier
     "spectrum collapse" was partly architecture capability.
  2. NO SHARP FLIP: heldout degrades smoothly with p_rand (freeD: 0.74/
     0.59/0.43/0.25/0.11 for p=.1/.25/.5/.75/1.0). Graceful erosion,
     not a phase transition — at this scale capacity >> n so there is
     NO competition: train acc = 1.0 on BOTH subsets at every p.
  3. Data-size control: heldout FLAT in n (0.60/0.60/0.58 for n=1k/4k/
     8k at p=.25) — not data starvation.
  4. DYNAMICS: no generalize-first-then-memorize: train hits 1.0 by
     ep3000 with heldout jumping to 0.599 simultaneously, then BOTH
     frozen (heldout 0.593 at 30k — no late erosion). Structure and
     memorization co-form fast; noise damage is paid at fit time via
     feature interference, not by late overwriting.
  ⇒ The interesting regime is CAPACITY-LIMITED (user's saturation
  point): launched triage sweep (n 8k/16k >> capacity, p .25/.5, both
  archs): prediction — the model triages, keeping structure (covers
  many clean images per parameter) and dropping random facts; measure
  train_acc_clean vs train_acc_rand divergence. results/p2_saturation.log.
- ITERATION (P2, user-directed): launched p2_flip_sweep.py (b3czj13hb):
  capability controls (p=0 pure-clean, SGD vs Adam, negI-m10 vs
  freeD-m32 — rules out bad-optimization/architecture-incapacity),
  p_random sweep {0.1..1.0} to locate the memorization flip, data-size
  control (n 1024-8192 at p=0.25), and dynamics tracking (train/heldout
  acc every 500 epochs at p=0.25 — generalize-first-then-memorize?).
  User's smaller-model/dataset saturation suggestion queued as the
  follow-up knob if controls are ambiguous.
- ITERATION (P2-cron #2, results): (a) P2-2 long-train: support pileup
  IS emerging on MNIST but very slowly — support_frac 0.002 (8k epochs)
  → 0.032 (100k); dense-input directional convergence is much slower
  than the toy's (which hit 0.19-0.62). Max-margin structure: present,
  weak at feasible budgets. (b) P2-3 mixed p=0.5 SURPRISE: NO
  margin separation (median clean 1.65e-5 ≈ random 1.54e-5) and
  held-out clean acc only 0.245 — the m=10 D=−I silence-code bilinear
  memorizes EVERYTHING as facts; it barely learns digit structure even
  when half the labels are true. The structure/memorization spectrum
  needs capacity+inductive room: next → free-D larger m (32-64), lower
  noise fractions (p_rand 0.1-0.2), per-image margin-vs-generalization.
  Refined into queue as P2-3b.
- ITERATION (P2-cron #1): launched p2_mnist_structure.py (beonu8pqd):
  (a) P2-2 completion — pure-random n=1024 at 8k vs 100k epochs (does
  the support pileup emerge with long training?); (b) P2-3 — mixed
  p=0.5 clean/random n=2048, 60k epochs: margin split clean-vs-random +
  held-out generalization (structure stored free vs memorization needing
  support). Model saved for follow-up λ/editing analysis
  (results/p2_mixed_model.pt). Collect next iteration.
- EDITING TESTS (user session): certified 2-parameter local forget
  (edit one row at the fact's two columns; exact eval; min-norm
  zero-collateral).
  v1 NEGATIVES: representer-term subtraction fails spectacularly
  (margins quadratic in L → overshoot, margin +2020); full-gradient
  direction fails for free facts (breaks floor-hugging token-sharers).
  v2 RESULTS: SGD λ-top support fact 36 FORGOTTEN at 4.7% |ΔL|, zero
  collateral ✓ (#6 verified with proper local computation). Optimizer
  grid (min-margin "support-like" + max-margin "free-like" per model):
  SGD: both picks FAIL in 2-param space — the taut web leaves neighbors
  no slack; Adam: support-like trivially editable (0.3% |Δ|), free-like
  (margin 16388) uneditable; Muon: both editable but expensive (29%/18%).
  PUNCHLINE: editability inverts robustness — SGD's tension web resists
  local surgery precisely because everything is load-bearing; Adam's
  slack-rich geometry is the most surgically editable. Free facts
  (deeply implied) resist local editing in every optimizer, confirming
  the interpolation theory. Insertion (#8) needs no new test (divergence
  experiment + training itself).
- MUON FOLLOW-UPS (user request), two honest nulls:
  1. Multi-dyad spectral-KKT fit: residual 0.5662 — identical to the
     rank-1 fit (optimal mu = pure top dyad). Muon's solution is NOT at
     the idealized spectral max-margin KKT point at this budget/impl.
     Fair statement: GD's Frobenius convergence is theory-backed and
     verified (0.06); Muon's spectral analog is conjecture-level for
     nonlinear homogeneous nets, and practical Muon (momentum + NS
     approx) shows only the qualitative signature (equalized spectrum at
     d3: 1.49 vs 1.8-1.9) — which itself did NOT replicate at d8
     (ratios: gd 2.51, adam 2.18, muon 2.21).
  2. Packing metrics at d8 (n=256): PR facts-per-neuron 94-103, PR
     neurons-per-fact 3.7-3.8 across ALL three optimizers — no
     meaningful optimizer effect on fact-packing statistics. The
     superposition-structure hypothesis for Muon: not supported here.
  Side note: plain SGD at d8 ceiling reached only 0.801 in 60k epochs
  (vs Adam 1.0, Muon 0.934) — the clean-geometry optimizer pays an
  optimization-speed cost at capacity.
- OPTIMIZER-GEOMETRY TEST (user Q re Muon): steepest-descent-norm theory
  CONFIRMED at d3 (multi-rival NNLS representer fit): Adam residual
  0.742 (targets a sign/ℓ∞-margin object — not noise, a different
  geometry); plain GD+momentum residual 0.0596 — 12× cleaner, near-exact
  Frobenius-KKT λ's. Muon answer: would be clean in the SPECTRAL-norm
  max-margin geometry (KKT in top singular subspace), i.e., wrong
  geometry for Frobenius λ's but potentially interesting for
  superposition structure (spectral margin equalizes singular values).
  PRACTICAL RULE: for per-fact λ attribution / editing on trained
  models, train with plain (S)GD, not Adam; or re-derive the representer
  in the optimizer's own norm.
- SANITY CHECK (user request): property suite re-run on the ARTIFICIAL
  d2 KKT point (our maximin construction, floor 1.37e-1, 8 support/8
  free): A. representer residual **0.0002** (vs 0.74 on Adam's solution)
  with all 8 λ>0 — test code validated; the Adam-solution slop is real
  (Adam bias/finite time/multi-rival), not a bug. C. noise certificates
  even sharper: Spearman −0.883. B. removal: free fact still implied ✓;
  support-fact removal ALSO leaves it predicted at d2 — a small-scale
  degeneracy (16 pairs/2 labels: remaining constraints imply the removed
  one; the asymmetry needs d≥3's freedom, where GD showed it cleanly).
- POST-PROGRAM (user session): MAX-MARGIN OBJECT PROPERTIES verified on
  GD d3 solution (22 support / 14 free facts):
  A. Representer (L in cone of support margin-gradients): PARTIAL —
     rel residual 0.74 (caveats: Adam ≠ gradient flow; my per-fact
     gradient uses only argmin rival, true KKT uses all active (i,c)
     rivalry pairs); λ mass still concentrates on support (18 vs 9).
  B. REMOVAL ASYMMETRY (clean): retrain without a FREE fact → fact is
     STILL predicted (implied by the web; data-deletion unlearning
     FAILS); retrain without a SUPPORT fact → fact no longer predicted.
  C. Per-fact noise certificates: radius m_i/||∇m_i|| predicts empirical
     flip-rate under weight noise, Spearman −0.65.
  Editing implications catalogued in chat: isolation masks (2 cols ×
  2 rows per fact), λ as keystone score, support-set/λ-spectrum as the
  right solution-comparison invariant, KKT sensitivity for edit
  prediction. TODO if pursued: exact multi-rival representer fit; SGD
  (not Adam) check; λ-guided surgical unlearning experiment.
- FINAL (insertion-cron #18): certified-insertion program COMPLETE; loop
  stopped (cron bea49c3d deleted). Final consistency edit: title and
  abstract now lead with the robust 0.70-0.73× headline; the certified
  785/2867 numbers framed throughout as criterion-gameability results.
  All 5 original queue items done; all follow-on questions (keystones,
  robustness hierarchy, divergence, basins, re-tensioning, min-smoothing,
  integrity audit) resolved and written into submission_draft.md.
  Restart with /loop anytime; open directions parked in earlier notes.
- ITERATION (insertion-cron #17): INTEGRITY AUDIT of headline capacities.
  d16 capacity-point solution: 784/785 stored facts lie within ±1e-9 of
  exact ties; at ANY margin epsilon (1e-12+) certified capacity ≈ 0.
  The 785 (and by extension 2867 at d32) are argmax-criterion-legal but
  are statements about favorable float64 round-off placement —
  potentially not even bit-reproducible across BLAS builds. REFRAMED in
  submission_draft.md as a criterion-gameability finding: the challenge's
  argmax criterion without a margin floor is gameable by tie-placement;
  a margin/quantization-robustness requirement would sharply separate
  trained solutions (pass at 6 bits) from tie-constructions (fail at any
  epsilon). The FAIR headline under a margin-corrected criterion is the
  repair pipelines' 0.70-0.73× trained. Certified insertion's lasting
  contributions: the ledger/explanation framework, the tie-manifold
  discovery, and the criterion-gameability result itself.
- ITERATION (insertion-cron #16): NEWTON FINISHER FAILS — and explains
  why. d3 unchanged (−1.1e-5), d4 worse (Newton's least-squares
  compromise trades acc for nothing). DIAGNOSIS: the stall points are
  NEGATIVE LOCAL MAXIMA of the hard min-margin landscape: the active
  gradients' convex hull contains 0 (maximin p≈0 with floor<0) — no
  direction raises all active margins to first order, and linearized
  KKT can't manufacture feasibility that isn't there. GD+CE escapes
  these because cross-entropy is a SMOOTHED min (softmax weighting
  trades margins across facts smoothly) — the essential role of GD is
  not credit assignment per se but MIN-SMOOTHING. A soft-min ascent
  with exact line steps would work but is honestly gradient ascent on a
  surrogate (challenge-legality doubtful; noted, not pursued).
  PROGRAM CHARACTERIZATION COMPLETE: certified/hard methods reach
  (a) tie-manifold solutions (max capacity, zero robustness) and
  (b) at small d, true max-margin tension (d2 beats GD); at d≥3 the
  hard-min landscape's negative local maxima wall them off, and
  smoothing — the one tool that crosses — is the definition of the
  forbidden method. This is the deepest version of the greedy-vs-GD
  gap the program has produced.
- ITERATION (insertion-cron #15): re-tensioning v1.1/v2.
  (a) Stage-2 starts: DEGENERATE like certified (spectral margins are
  also tie-scale) — floors stall at 1e-33. Random starts are the right
  regime.
  (b) v2 stall-escape (random probes orthogonal to active gradients,
  exact line steps): d3 floor −1e-2 → −1.31e-5 (acc 0.75), d4 → −4.9e-7
  (acc 0.984) — NEARLY feasible but asymptotically creeping; cannot
  cross into positive floors (GD: +1.6e-2/+4.5e-3). The boundary
  crossing needs coordinated multi-dir adjustment = the global credit
  assignment wall again, now in margin space.
  STATUS: certified re-tensioning fully works at d2 only (floor 1.34e-1
  > GD). NEXT IDEAS (queued): proper second-order maximin step at
  near-feasible points (active-set Newton: solve the KKT system of the
  local max-margin QP over active facts — a linear solve, certified);
  alternate insertion sweeps with re-tensioning phases (store-then-
  tension-then-store); larger probe budgets.
- ITERATION (insertion-cron #14): CERTIFIED RE-TENSIONING v1
  (maxmargin_cert.py: active set → min-norm-point maximin direction via
  Frank-Wolfe-with-away-steps → exact-eval line step).
  RESULTS: (a) from the certified tie-solution: 0 steps — the tie
  manifold is first-order degenerate for max-margin ascent too (all
  active gradients ≈ 0 → spurious KKT); consistent with everything.
  (b) FROM RANDOM INIT: d2 acc 1.000 with floor 1.34e-1 — EXCEEDS GD's
  1.11e-1 — in 24 exact steps, support frac 0.44. FIRST ROBUST CERTIFIED
  SOLUTION; better max-margin point than GD's at d2. (c) d3/d4 stall at
  maximin saddles (negative floors, acc 0.25-0.31). Fixes queued:
  start from stage-2 spectral (nondegenerate, structured); proper
  root-based line search (v1 uses a crude grid); second-order/curvature
  escape at saddles (maximin over quadratic forms = small eig problem);
  more restarts; then scale d and compare floors + basins vs GD.
- NEW TOP DIRECTION (user session): CERTIFIED MAX-MARGIN ("re-tensioning
  without GD"). Empirical basis: GD(D=−I) solutions carry the hard-margin
  fingerprint (support-set pileup at the min normalized margin: 62%/42%/
  19% of facts at d2/3/4; certified: none) and the architecture is
  degree-2 homogeneous → Lyu-Li: CE GD converges in direction to KKT
  points of max γ s.t. margin_i ≥ γ||L||²_F. Certified and GD solutions
  are DIFFERENT even at d2 (row cos 0.48/0.03 at 100% acc both).
  ALGORITHM (v1): active set A = facts near min normalized margin;
  direction = maximin ⟨∇m_i, u⟩ (min-norm point in conv{∇m_i} via
  Frank-Wolfe-with-away-steps, each step closed form); exact line step on
  the min normalized margin via quadratic-root candidates; accept iff
  floor rises. Each update names its support facts (λ's) = per-update
  explanation. Verify at d2/3 against GD's min margins (0.111/0.016).
- GD-FROM-CERTIFIED (user question resolved): the certified endpoint is
  NOT a local minimum of CE and is NOT usefully trainable-from. The FIRST
  Adam epoch destroys it: d8 0.855→0.188, d16 0.740→0.101 (CE at exact
  ties has near-uniform softmax → large gradients; the tie manifold is a
  HIGH-loss unstable set, not an optimum). From the rubble, lr=1e-2
  rebuilds to ordinary-GD territory (d8 0.871, margins median 1.2;
  d16 only 0.630 — BELOW certified's 0.733 and below random-init GD);
  lr=1e-3 never recovers (0.305/0.220). CONCLUSION: no warm-start value
  in either direction — the tie-manifold and tension-web solution
  geometries are mutually inaccessible: GD can't preserve ties while
  robustifying, and (from cron #4-5) construction can't build tension.
  The two methods are genuinely different phases of solution, not two
  routes to one optimum.
- ITERATION (insertion-cron #13): DIVERGENCE EXPERIMENT (queue item c)
  at d8, same Stage-2 init, same new fact (231: (13,0)->7):
  | | certified exact step | GD to convergence (S+f) |
  | weight change |dL|_F | ~1e-15 (tie-nudge) | 30.9 |
  | columns touched | effectively 0 | all 32 |
  | change in f's own columns | — | only 5.1% |
  | f's margin after | ±1e-28 (tie) | 1.42 (real) |
  | stored retention | 113/114 (1 sacrificed) | 114/114 |
  | collateral | +2 | +13 free extra facts |
  | epochs | 1 step | 3091 |
  ONE fact insertion: GD performs a GLOBAL re-tensioning ~10^16 larger in
  weight movement, 95% of it OUTSIDE the fact's own columns — buying a
  real margin and 13 free facts. Certified flips a tie. This is the
  basin/tension story at the single-fact level and directly confirms the
  user's hypothesis (GD accommodation is distributed web adjustment).
  NUMERICAL-INTEGRITY CAVEAT found: certified tie-facts sit at float64
  ambiguity (accept-eval said stored; re-eval margin −3e-28). Queue: add
  minimum relative step slack (~1e-9) to acceptance so stored facts are
  stable under re-evaluation; re-verify headline capacities with it.
- ITERATION (user session, interconnectedness): defined per-fact-pair
  coupling J_ij = <grad margin_j, unit grad margin_i> (nonzero only for
  token-sharing pairs). THREE-WAY RESULT at d8 (fig_interconnect_d8.png,
  fig_interconnect_hist_d8.png):
  | method    | coupling per fact (pos/neg) | basin (acc vs noise σ) |
  | certified | UNDEFINED — 100% of facts gradient-degenerate (all pre≈0) |
  |           | collapses at any σ (0.17 at σ=0.001) |
  | repair    | +1.9 / −2.1 | degrades from σ≈0.01 |
  | GD D=−I   | +11.1 / −15.0 | holds 0.85 at σ=0.1, 0.70 at σ=0.3 |
  INTERPRETATION: interconnection = load-bearing tension. GD builds a
  taut frustrated web (neg slightly > pos) whose tension IS the wide
  basin; repair is loosely coupled; certified sits on a measure-zero
  zero-tension tie manifold (every fact at a flat degenerate point).
  This reframes solution multiplicity: not "1 solution vs many" but
  "measure-zero manifold vs fat basins".
  QUEUED (user asks): (c) divergence experiment — same init, add ONE new
  fact: certified exact step vs GD-trained-to-convergence on all facts;
  compare weight-change support/magnitude/collateral. (d) multiplicity
  accounting: characterize GD solution DIVERSITY from same init under
  batch order/noise (we have seed-spread; add same-init perturbation
  spread). GD-from-certified warm-start job still running (bleyukihh).
- ITERATION (insertion-cron #12): d32 FINAL — certified 0.902 @ n=2867
  → capacity ≥2867 = **0.954× trained** (3008). Certified ratios RISE
  with scale (d16 0.82× → d32 0.95×): tie supply grows with d and exact
  tie-placement exploits it; under the argmax criterion the certified
  construction nearly matches gradient descent at d32. (ε-tie caveat
  applies throughout; robustness hierarchy unchanged.) Draft updated.
  MEASUREMENT PROGRAM ESSENTIALLY COMPLETE. Remaining optional: exact
  d32 capacity endpoint (search 2867-3008); maximin-from-scratch (open
  problem, parked); ReLU composite+compensation (parked).
- ITERATION (insertion-cron #11): ROBUSTNESS HIERARCHY completed —
  repair solutions have real-but-thin margins (median 0.043; retains
  0.648 at γ=0.01, 0.234 at γ=0.1), sitting between certified (exact
  ties, total γ-collapse) and GD (median 1.69). Three-tier table added
  to submission_draft.md; headline retitled 0.70–0.82×; d8 table row
  corrected to per-size best (H16: 248 = 0.97×). d32 n=2867: sweep 1 at
  exactly 0.900 — final sweep running; if it holds, certified d32
  capacity ≥ 2867 = 0.95× trained. Collect next iteration.
- ITERATION (insertion-cron #10): MAXIMIN RETROFIT REFUTED. Null-part
  scaling probe (λ up to 30) on the certified d8 solution: acc FALLS
  (0.855→0.637) and zero facts clear margin 0.1. Mechanisms: (a) |λp+r|
  non-monotone — loudening can cross zero and flip facts; (b) FATAL: the
  margin is a min over rivals and at d8 essentially every fact has a
  within-component rival pinned at tie scale (giant components), so the
  min never escapes ties no matter how loud cross-component rivals get.
  CONCLUSION: robust margins cannot be retrofitted onto silence-code
  constructions; they require breaking component structure globally —
  i.e., margin-aware construction from scratch (open problem; GD does it
  by paying ~10% capacity). The capacity-robustness tradeoff framing in
  the submission is final. d32 n=2867 still in sweep 0 (pid alive).
- ITERATION (insertion-cron #9): submission_draft.md updated — d32
  partial result (0.962 @ 2458 vs repair 0.868) and the three-entry
  ledger showcase folded into the certified section, with tie-scale
  framing adjacent to the robustness caveat. d32 n=2867 eval still
  running (init 0.500 → longer sweeps; pid 3493200 alive). Remaining
  queue: collect n=2867; maximin-margin robust variant (big open
  design); optional d32 full capacity binary search if n=2867 passes.
- ITERATION (insertion-cron #8): d32 FIRST RESULT — certified 0.962 at
  n=2458 (repair: 0.868 there) → certified capacity at d32 is ≥2458 and
  likely well beyond; n=2867 eval running (would put capacity within 5%
  of trained 3008). EXPLANATION SHOWCASE built (3 rendered ledger
  entries for the writeup). Sharpest specimen: fact (3,10)->7 obstructed
  SOLELY by fact (1,10)->7 — same label, shared t2 column, competing for
  one tie (blocker margin 3.3e-30). Showcase also displays the caveat
  vividly: certificates/steps print at 1e-15..1e-30 scale — tie-shuffling,
  not margin engineering. For the draft, showcase values should be
  reported in tie-scale units with the robustness caveat adjacent.
  NEXT: collect d32 n=2867; add showcase + d32 to submission; maximin
  robust variant remains the big open design.
- ITERATION (insertion-cron #7): two honest ReLU nulls + d32 launched.
  1. Row-basis directions for ReLU insertion: ZERO gain (0.566 = 0.566)
     — bottleneck is the AND-condition (all S own neurons quiet
     simultaneously), not direction poverty. Structural insight: bilinear
     fixes are rank-1 (m=1 neuron/label); ReLU S=3 fixes need coordinated
     multi-row moves.
  2. Atomic composite own-silencing moves: WORSE (0.527) — compound
     silencing steps collateral-damage token-sharing facts, and greedy
     path-dependence amplifies it. ReLU certified insertion stuck
     ~0.53-0.57 at d8 vs repair 0.613. Tentative conclusion: certified
     insertion is bilinear-NATIVE; the quadratic tie-structure it
     exploits has no ReLU analog. (Composite + rival-compensation designs
     still untried; low priority.)
  3. d32 certified evals launched (pid 3493200): n=2458 init 0.746,
     insertion sweeps running; collect next iteration. If ≥0.9 at 2458,
     certified capacity already ≥ repair's entire bracket at d32.
- ITERATION (insertion-cron #6): THREE results.
  1. CERTIFIED d16 CAPACITY (challenge metric, acc≥0.9): **785** vs
     repair 672, trained 960 → 0.82× trained — best construction on the
     challenge metric to date. (Caveat stands: epsilon-margin storage.)
  2. Gauge check: certified stored margins are RELATIVE 1e-30 — exact
     ties, scale-invariant; fragility is absolute, not units.
  3. ReLU PORT of certified insertion implemented (relu_insert.py,
     piecewise-linear kink candidates): d8 ceiling 0.305 (ridge init)
     → 0.566. Generalizes mechanically (doubles the init) but trails the
     ReLU repair pipeline (0.613) — direction set still minimal (no row
     basis, no segment-crossing candidates). The bilinear's tie-exploit
     advantage is weaker under ReLU (silence = one-sided halfline, no
     exact cancellation ties to win). Improvements queued.
  Next: d32 certified capacity; richer ReLU direction set; maximin-margin
  robust variant; update submission table with 785.
- ITERATION (insertion-cron #5): GAMMA SWEEP — certified insertion is
  STRICTLY an argmax-criterion method as built: at ANY slack gamma>0 the
  d8 run collapses to the init (0.445; zero facts clear margin>0.01 —
  even init-stored facts are epsilon-margin, since the spectral silence
  keeps rivals near zero too). GD(D=−I) contrast: median stored margin
  1.69, retains 0.859 at gamma=0.1, 0.629 at gamma=1.0. Root cause: our
  constructions never push MIN rival response up (normalization fixes
  the MEAN off-response only). Robust-margin certified insertion needs a
  maximin objective (push quietest rival loud) — queued. Launched
  certified-insertion capacity search at challenge metric, d16
  (pid 3491992, results/cert_capacity_d16.log). User Qs answered in
  chat: MLP portability (ReLU ports EXACTLY — piecewise-linear intervals;
  SwiGLU does NOT — transcendental roots), their SOTA + metric summary.
- ITERATION (insertion-cron #4): KEYSTONE MECHANISM RESOLVED — and it
  reframes the d16 result. Keystone profile: all 8 keystones (and the
  MEDIAN stored fact!) have margin ≈ 0. Margin-inflation sweeps
  (count-preserving hinge-improving polish on thinnest stored facts):
  ZERO effect at d8 — margins are not accidentally thin, they are H11
  STRUCTURAL TIES: own row ≈ 0 AND structurally co-quiet rival ≈ 0; the
  certified insertion wins ties by epsilon placement. Consequence:
  (a) keystones = thin facts on contested columns — any shared-column
  move flips their tie; (b) the d16 win over GD(D=−I) (0.733 vs 0.686)
  is partly CRITERION ARBITRAGE: argmax evaluation accepts margin→0⁺
  storage which cross-entropy GD cannot exploit (CE demands real gaps).
  The certified solution is high-capacity but ZERO-ROBUSTNESS (any
  weight noise kills tie-facts — contrast trained models' 6-bit
  quantization robustness). HONEST FRAMING NEEDED in submission: report
  capacity as a function of required margin slack γ (capacity-robustness
  curve) for certified vs GD vs trained. NEXT: implement slack-γ
  acceptance (store only if margin > γ), sweep γ, add tradeoff curve +
  caveat to submission_draft.md.
- ITERATION (insertion-cron #3): HEADLINE — d16 certified insertion
  FINAL 0.733 at ceiling: best construction at d16 (repair 0.628-0.66)
  AND above same-architecture GD (D=−I frozen: 0.686). Keystone
  concentration extreme at d16: 8 facts × 22-23 blocks each of 273
  obstructed. Item 3 (pair moves): implemented pair_rescue (depth-2:
  fix-then-reinsert-broken, commit iff net gain) — modest: +1 fact at d8
  (0.855→0.859); keystone conflicts are deeper than pairwise. Item 5
  DONE: certified-insertion narrative written into submission_draft.md.
  Queue remaining: deeper keystone analysis (why do those 8 facts sit on
  contested columns? token histogram shows mild concentration), depth-3+
  or keystone-eviction strategies, d16 pair_rescue run.
- ITERATION (insertion-cron #2): ROOT-CAUSED the d16 silent deaths — a
  segfault (exit 139) in torch 2.13 scalar ops inside root_candidates'
  hot loop (both "harness kill" theories wrong). Fixed by pure-Python
  float arithmetic (also faster). IMPLEMENTED V3: category-B rival
  least-squares direction (g[c*] = (X_ownᵀX_own+εI)⁻¹ x_i — un-quiets
  rival with minimal own-fact disturbance) + per-direction affected-set
  computation (dense directions handled correctly). Results (stage 2):
  d4 0.984 (1 obstructed, was 2); d8 0.855 (was 0.844 strict). Keystone
  ledger sharper: fact 90 blocks 8 insertions, 241×7, {55,70,75}×6.
  d16 relaunched with fixed code (pid 3488967). Next: pair moves aimed
  at keystones (item 3), d16 results, then narrative (item 5).
- PRIOR LEDGER (user idea — init as pre-inserted facts): built for Stage 2
  d8 (114/256 stored). Certificate EXACT: own-fact response ≡ residual
  response (null part contributes 0, verified 0.0e0). Unstored causes:
  100× category A (own residual too loud), 42× category B (H11 structural
  tie: rival exactly co-quiet because fact's tokens lie in one component
  of the RIVAL's own-fact graph), 0 accidents. ⇒ INSERTION V3 DESIGN:
  read each fact's ledger category and pick the matching certified move —
  A: own-row residual moves (existing machinery); B: RIVAL NULL-SPACE
  moves (perturb rival's component coefficients: un-quiets rival on the
  contested fact while preserving rival's own facts EXACTLY — safe by
  construction, only third-party effects need interval certificates).
  This is the full explanation-first construction: every stored fact has
  a certificate, every unstored fact a cause, every fix a matched move
  type. TOP PRIORITY for next cron iterations: implement v3.
  (d16 nohup run still in sweep 0, pid 3486170 — collect later.)
- ITERATION (insertion-cron #1): queue item 1 DONE — ledger bookkeeping
  fixed (strict count-increasing insertion vs count-neutral polish moves
  separated; blockers recorded from ALL i-fixing candidates). Item 4
  first pass: at d8 the KEYSTONE STRUCTURE the user predicted appears —
  40 wrong-and-obstructed facts, with facts {0, 64, 78, 211} each
  blocking 5 insertions (and {36,135,221,239} ×4). d4: only 2 obstructed,
  both blocked by fact 52. Token histogram mildly concentrated (t2=1,
  t1=2, t1=13). NOTE: strict acceptance converges to 0.844 vs loose
  0.875 at d8 — polish-always hybrid queued with pair moves (item 3):
  pair moves should target exactly the keystone blockers (move blocker
  + insertee jointly). Item 2: first d16 background run died silently
  after the start line (harness kill suspected; profiling showed compute
  fine at ~1s/fact). Relaunched detached via nohup (pid 3486170,
  results/insert_d16.log, sweeps=4 fast, ~40 min); collect next iteration.
- INSERTION V2 (certified, richer directions): per-row sym/antisym basis
  moves over each fact's 2-column subspace + v1 candidates, exact root
  steps, affected-only evaluation, (count, hinge) lexicographic acceptance
  (NOTE: this also accepts count-neutral margin-polish moves — good for
  progress, but ledger bookkeeping is loose; fix queued). From Stage 2:
  d4 0.984, d8 0.875 — BEATS blind H12b repair (0.844) at d8, within
  0.016 of overall best (H16 0.891), fully certified. New cron loop
  (bea49c3d, 20 min) continues: ledger fix, d16, pair moves, ledger
  structure analysis, submission narrative. Files: interval_insert.py (v1),
  insert_v2.py.
- H16 CAPACITY FINAL: d8 248/256 (97% of trained 256 — was 200); d16 528 —
  WORSE than H12c's 672. H16 is scale-sensitive: at d16 the giant component
  swallows nearly all tokens, the null space degenerates toward the useless
  constant vector, and the init underperforms reweighted anti-Rayleigh.
  Best-known construction per size is now MIXED: d≤8 → H16 pipeline,
  d≥12 → H12c pipeline. (Submission should quote per-size best.)
- INTERVAL INSERTION v1 (user's interference idea): per-fact gradients
  factorize exactly, g_i = alpha_i ⊗ x_i, so interference =
  (alpha_i·alpha_j)×(token overlap); 78% of pairs exactly zero; shared-token
  pairs mean cos +0.145 (cooperative). v1 algorithm (3 rank-1 direction
  candidates per fact, exact quadratic-root step candidates, accept iff
  global count increases): d4 0.625→0.906 in 3 sweeps; d8 0.227→0.621.
  Below repair, but every step is certified and every failure names its
  blocking set (obstruction ledger; note ledger overcounts — facts fixed
  cooperatively by others' steps aren't popped). Queued improvements:
  richer direction space (full 2-column/2m-dim subspace per fact via small
  eigensolve), margin slack, pair-insertion moves, start from Stage 2.
- QUEUED (user idea, awaiting full explanation): "Stage 1 (the
  anti-Rayleigh negative init) already gets you many facts stored..." —
  user will explain the continuation next session. Likely direction:
  exploit the facts Stage 1 already stores (freeze/protect them and handle
  only the remainder incrementally?). DO NOT start guessing-implementing
  before they explain.
- LAUNCHED: H16 capacity binary search at d=8 and d=16
  (h16_capacity.py, byedo08vp). Variance handling: 8 restarts/eval
  alternating sigma 0.02/0.05, 16-restart recheck for near-boundary
  evals (acc in [0.85, 0.9)). Compare against handcoded_h12c
  (d8: 200, d16: 672) and trained (256*, 960).
- POST-PROGRAM (user session) — H16 SUCCESS, direct payoff of the
  null-space/residual analysis: parametrize each neuron as
  (exact-silence null part: component coefficients chosen by a tiny
  eigenproblem to maximize cross-component rival loudness) +
  beta × (complement anti-Rayleigh targeted at co-nulled in-component
  rivals). Raw init is WEAK (0.20-0.26 — unbalanced loudness) but it is
  the right BASIN: after the standard hinge repair (8 restarts, sigma
  0.05): d6 0.944 (was 0.889), d8 0.891 (was 0.844) at the ceiling —
  best construction numbers to date. Structure beats accuracy in an
  init: exact-silence skeleton gives repair a landscape where greedy
  ordering fixes work. TODO: H16 capacity at d16 (does the 0.70× ratio
  improve?); tune beta/noise jointly; fold into submission if it holds.
- POST-PROGRAM (user session): (a) residual-after-null analysis of the d4
  GD D=−I model: null-space share of row energy varies wildly per neuron
  (80/62/38/3%), residuals mutually independent (flat SVD) and ≤chance
  aligned with other labels' null spaces — beyond the H11 skeleton the
  loudness arrangement is idiosyncratic. (b) FACT-SEED ROBUSTNESS: all
  prior runs used fact seed 42 only (weight init was the sole randomness).
  Tested fact seeds 42-46 at d=3,4: GD with D=−I frozen reaches 1.000 at
  the ceiling for EVERY label draw → the silence architecture is not
  label-structure-dependent (at these sizes). Prune-frontier motifs across
  fact seeds stay within the same family (excite+veto pairs dominant,
  defaults in ~40%, occasional pure −permutation or all-detector) — same
  family, different member per draw. Caveat: ceiling-saturated sizes;
  near-capacity label-dependence at larger d untested.
- POST-PROGRAM (user experiment): GD with D=−I FROZEN, L trained (standard
  11-seed protocol). Ceiling accs: d4-6 1.000 (= free-D), d8 0.961,
  d12 0.776, d16 0.686. d16 capacity: 864 vs free-D 960 vs construction
  672. DECOMPOSITION of the construction gap at d16:
  (a) silence-code restriction (free D → fixed −I, both GD): −10% —
      the pure silence architecture holds 90% of free capacity;
  (b) optimizer (GD → eig+greedy, both D=−I): −22% — two-thirds of the
      remaining gap. Confirms Stage-0 (D=−I) was the right call and the
      improvement target is global optimization of L. Also: the −I
      restriction bites hardest at the CEILING (d12: 0.776 vs 1.000) —
      the last few facts need free-D flexibility, the first 85% don't.
- POST-PROGRAM (user session): D-mimicry test — imported trained canonical
  sparse D graphs (fleet seeds) as fixed readouts, built L from scratch via
  per-neuron two-sided eigensolves + fast repair. UNDERPERFORMS D=−I at
  every d: d5 0.83-0.90 vs 0.910; d6 0.81-0.83 vs 0.889; d8 0.69 vs 0.844.
  Interpretation: trained sparse D graphs are CO-ADAPTED artifacts of their
  own L, not transferable blueprints — a graph forces asymmetric per-neuron
  jobs (loud-for-c AND quiet-for-c') that construction satisfies worse than
  uniform silence, where every neuron has the same job and the constructor
  keeps maximal freedom. Fourth independent confirmation that readout
  structure is not the lever. Side-by-side canonical D figures (10 seeds
  each, all acc≥0.9): tiny_models/sym_random/multiseed/fig_D_d{3,4,5}.png.
- FINAL (cron #16): housekeeping — all hypothesis and task checkboxes now
  closed with verdicts. PROGRAM COMPLETE: submission_draft.md holds both
  headline results (bilinear construction at 0.70-0.73× trained, d6-32;
  ReLU port at 3-3.5× the challenge authors' construction). Recurring
  research loop stopped; restart with /loop anytime, or point a new loop
  at: GPU-ized repair for d≥64, draft polish for posting, or the
  full-transformer variant.
- ITERATION (cron #14): S-sweep at the ReLU d16 capacity point: S=4 edges
  S=3 at reduced budget but at full budget tops out at n=296 → acc 0.899
  (just under threshold). Capacity 280 (S=3) stands as robust; no update
  to the submission numbers needed. d32 bilinear bracket launched
  (bjauj501n: n ∈ {2048, 2458, 2867}, ~1h) — collect next iteration to
  finish task 21.
- ITERATION (cron #12): task 20 (ReLU-MLP port) first results. Pipeline:
  balanced S-neuron-per-label assignment, D=−assignment, per-neuron ridge
  init (targets −1 own/+1 other), fast_repair generalized with act="relu"
  (h12c_fast.py now supports both). Ceiling accs: d8 best 0.613 (S=3),
  d16 ~0.40 (S=3; S=4 still running). SURPRISE: ReLU is HARDER for our
  pipeline than bilinear (0.613 vs 0.844 at d8) — one-sided silence needs
  ALL S own neurons ≤0 (AND-condition) while wrong labels need coverage;
  bilinear's signed cancellation is friendlier. BUT the comparison that
  matters: their published hand-coded MLP d16 capacity = 80-92 facts
  (0.11× trained 784). Our d16 ceiling-acc 0.40 implies capacity ~300-400
  → likely 3-4× their construction ON THEIR ARCHITECTURE. Task next:
  binary-search the ReLU-port capacity at d16 to nail that number.
- [x] Task 22 DONE: ReLU-port capacity at d16 = 280 facts (acc≥0.9, S=3)
      vs their hand-coded 80-92 (→ 3.0-3.5×), their hybrid ~256-268
      (≈ parity, but we use no GD at all), trained 784 (0.36× vs their
      0.11×). submission_draft.md updated with the same-architecture
      section. Optional follow-ups: d32 ReLU capacity; S sweep at capacity
      n (S was tuned at ceiling n).
- ITERATION (cron #11): task 6 done — base queue COMPLETE. Forensics
  verdict: constructed-model failures are uniformly spread over tokens and
  neurons (no hotspots), killing the block-move hope and confirming the
  greedy-vs-GD global-credit-assignment diagnosis. Figure:
  results/fig_forensics_d8.png. Added tasks 20 (ReLU-MLP port — note the
  one-sided ReLU silence condition is LINEAR feasibility, plausibly easier
  than bilinear cancellation; high challenge value) and 21 (d≥32 scaling).
- ITERATION (cron #10): task 5 (H6 ablation) done — see task list entry.
  Confirms the veto/default economy: the zero-tap default label "rents"
  every veto tap in the model (any single veto ablation kills it 16/16),
  while excitation taps are perfectly modular. Only task 6 (gradient
  forensics tooling, low priority) remains in the base queue; open-problem
  directions (block moves, d≥32 scaling, ReLU-MLP port) await user
  direction. Loop is in diminishing-returns territory.
- ITERATION (cron #9): task 10 done — wrote submission_draft.md: full
  challenge-entry draft (algorithm spec, capacity table, mechanistic
  "why silence" story, limitations, reproduction pointers). Queue pruned
  with explicit [x]-obsolete marks (4, 9) and completions (7, 8, 10).
  Remaining open items: 5, 6 (low-priority interpretability), plus the
  open-problem directions listed in the draft (block moves, message
  passing, LP relaxation, d≥32 scaling, ReLU-MLP port). Program is at a
  milestone; further iterations have diminishing returns without user
  input on direction — loop will idle on low-priority items.
- ITERATION (cron #8): task 8 (asym variant) tested — h=(Lx)(Rx), D=−I,
  alternating weighted eigensolves + asym fast repair, m=d (1.8× params,
  diagnostic): d6 0.924, d8 0.824, d12 0.740, d16 0.675 vs sym
  0.889/0.844/0.703/0.628. Only ~+0.04 for 1.8× params — OR-cancellation
  helps slightly; not the lever. PATTERN across H13a/H15/asym: every
  variant lands ~0.63-0.68 at d16 ceiling. The bottleneck looks like the
  greedy/spectral paradigm's global credit assignment, not architecture:
  GD optimizes all facts simultaneously; our repair moves one entry at a
  time against a frozen rest. Remaining unexplored: block/coupled moves,
  massive restart diversity — expected to grind small gains.
- STRATEGIC NOTE: at 0.70× trained capacity (uncensored d16) with a fully
  interpretable, challenge-legal algorithm + complete mechanistic story
  (silence code, signed-embedding theory, solution-family census), this is
  a natural milestone to WRITE UP. Recommend: consolidate into results.md /
  a challenge-submission draft; keep gap-closing as future work.
- CAPACITY FINAL (task 19 DONE, fast engine, 6 restarts): max facts at
  acc≥0.9: d6 135/144 (94% ceiling), d8 200/256 (78%), d12 387/576 (67%),
  d16 672/1024 vs trained-sym 960 → **0.70× trained, uncensored**.
  results.md updated. Remaining program: close the 0.7→1.0 gap via the
  L-side ideas (richer repair moves, restart diversity, asym variant), or
  declare the construction result and write the challenge submission.
- ITERATION (cron #7): two negative results that sharpen the picture.
  1. H13a circulant tap graph WITH fast repair: d6 0.875, d8 0.805,
     d12 0.649, d16 0.532 — LOSES to plain D=−I silence + repair
     (0.889/0.844/0.703/0.628). The fixed shared-tap circulant is not the
     lever; d4's earlier 1.0 was matched by plain H12b-fast anyway.
  2. H15 alternating ridge-D / repair-L: exactly zero gain at d6/8/12 —
     THIRD confirmation the readout is never the bottleneck. The whole
     remaining gap is L's expressiveness / repair's reachable set.
  Launched definitive fast capacity search (h12c_capacity.py, bc69k2pg7)
  for d=6,8,12,16.
- Ideas for the L-side gap (queue): richer repair moves (wrong-edge pairs,
  2-row coupled moves, token-column block moves); more diverse restarts
  (vary h9b reweight trajectory: beta, rounds, EPS); accept-equal random
  walk (plateau surfing, still greedy on ties); m>d neurons at matched
  params via smaller... not possible sym (m fixed by 5dm) — try asym arch
  (L≠R, m=5d/9) construction variant to test whether asymmetry helps
  constructions the way it doesn't hurt trained.
- ITERATION (cron #6): ENGINEERING WIN — h12c_fast.py rewrites repair with
  incremental candidate evaluation (changing L[r,j] shifts only pre[:,r];
  all 25 candidates scored in one batched op): ~1000× faster (d8: 0.1s vs
  ~2 min per repair). With 8 restarts + 12 passes now affordable:
  ceiling accs d4 1.000, d6 0.889, d8 0.844 (was 0.758), d12 0.703,
  d16 0.628 (trained at d16 ceiling ≈ 0.88-0.9 — uncensored gap now
  measurable: ~0.7× on accuracy at ceiling). d16 run 6.3 min, dominated by
  repair; h9b_solve is cheap. KILLED the two obsolete slow jobs (old d12
  capacity search, old H13a grid) — rerun both on the fast engine.
- Next up: (a) fast H13a grid d6-16 (fast_repair already supports arbitrary
  D — pass the circulant); (b) fast capacity binary search d8-16 →
  finish task 19 + update results.md numbers; (c) idea: mixed tap graphs
  from fleet census rather than pure circulant.
- ITERATION (cron #5): H13a d4 = 1.000 at ceiling (beta=1.0, gamma=1.0) —
  beats H12b's 0.938; d6/d8 grid still computing (slow, CPU-contended with
  d12 capacity search; both left running). Light-work iteration: task 10
  partially done — added "hand-coding the bilinear layer" section +
  results/fig_construction.png to results.md (construction progression
  H1→H9→H9b→H12b→H13a vs trained). Queue unchanged otherwise; obsolete
  items to prune next pass: task 4 (d2 exact analysis — solved by
  construction), task 9 (binary warm-up — superseded), task 12 (d3 error
  analysis — d3 solved).
- ITERATION (cron #4): task 19 capacity (ultimate test) partial results:
  H12b construction max-facts at acc≥0.9: d6 117/144 ceiling (81%),
  d8 196/256 (77%); d12 running (n=360 → 0.975, looks ~75-80%). Trained
  saturates the ceiling at these d (capacity censored at 100%), so the
  construction sits at ≈0.8× trained — versus the post's hand-coded MLP at
  ~0.1× (though at much larger d; honest comparison needs d≥16 where
  trained no longer saturates: trained sym d16 = 960/1024). Queued: d16
  capacity run (slow — repair cost grows; budget ~1h).
  H13a circulant tap-graph launched (bfffl79b0) — no output yet after
  10 min; repair_D may be slower than expected; check next iteration
  (if d4 grid alone is this slow, profile before scaling).
- Files: h12b_capacity.py (results/handcoded_h12b.jsonl), h13_tapgraph.py.
- ITERATION (cron #3): H12b BREAKTHROUGH. Hinge-margin repair (tau=0.3,
  margins not counts) + paired moves along own-edges + 3 restarts, on top of
  H9b: d3 1.000, d4 0.938, d5 0.910, d6 0.854, d8 0.758 (was 0.889/0.844/
  0.780/0.646/0.609). Fully non-GD (eig + reweight + greedy). Clears
  acc≥0.9 AT THE 4d^2 CEILING through d=5 — i.e. construction capacity =
  dataset ceiling there, matching trained models. Remaining gap: d≥6.
  Code: h12b_repair.py.
- Next: (a) task 7 capacity curve — binary-search max n for H12b at
  d=6,8,12,16 and compare vs trained/published curves (the "ultimate
  test"); (b) H13 shared-neuron tap graph to attack d≥6 (constructions so
  far use label-owned neurons only; trained models share); (c) consider
  more restarts/passes at d8 (repair is ~1-2 min there — budget).
- ITERATION (cron #2): three results.
  1. H9b (reweighted anti-Rayleigh) — best pure-spectral construction yet:
     d2 1.000, d3 0.889, d4 0.828, d5 0.720, d6 0.597, d7 0.551, d8 0.465,
     d12 0.373, d16 0.322. Beats H1/H1b/H9 at every d.
  2. H11 RESOLVED (theory): own-fact graphs are supercritical (one giant
     component + 2-7 cycles per label). Exact-null space per component is
     1-dim (v = α on t1-side, −α on t2-side), and 60-93% of WRONG facts sit
     inside the same component → exact silence co-nulls them → mass ties.
     Silence must be approximate-with-margins; per-neuron problem is a 1-D
     SIGNED-GRAPH EMBEDDING (own edges attract to cancellation, wrong edges
     repel), of which anti-Rayleigh is exactly the spectral relaxation.
     Relaxation→ordering gap is the remaining obstacle.
  3. H12 (greedy per-entry grid repair on count objective, no GD): helps at
     d≥5 (d8 0.465→0.609, d5 0.67→0.78) but local-optima-bound (d3 stuck
     0.889). Construction SOTA now: d8 0.609 vs trained 1.0.
- New queue items:
  - [ ] H12b: better repair — hinge-margin objective (not raw count),
        multi-scale candidate grids, paired-entry moves along own-edges
        (escape cancellation-preserving directions), random restarts.
  - [ ] H13: neuron-sharing tap graph (m=d shared neurons, each label reads
        ~d/2 signed taps) — construct assignment from fleet census, then
        solve L per neuron by signed-embedding with the union of its
        consumers' constraints.
  - [ ] H14 (vague, park): joint all-label solve; pairwise constraints are
        differences of squares (hyperbolic) — look for a joint
        diagonalization trick.
- ITERATION (cron #1): H1b reweighted Rayleigh: d2 1.000 (!), d3 0.889,
  d4 0.734, no gain d≥5, late rounds destabilize (beta=2, 60 rounds).
  TASK 13 DONE — MAJOR FINDING: trained minimal d3 [1,1,1] solutions form
  TWO families: (a) 1 seed all-taps +1, L rows ≈ Rayleigh directions
  (cos up to 0.97!) — detector code; (b) 5 seeds all-taps −1: logit_c = −h_c,
  argmax = QUIETEST neuron — pure silence code, cos ~0 to Rayleigh.
  Silence is the modal solution. H9 (anti-Rayleigh: min generalized
  eigvec, D=−I) tested: beats H1 at d≥5 (d7 0.49 vs 0.36; d16 0.28 vs 0.20)
  but still ≪ 1.0. Since trained proves a perfect 1-direction silence
  solution EXISTS at d3, the failure is the objective: min-eig minimizes
  AVERAGE own-fact response; task needs per-fact ordering. → H9b.
- New hypotheses:
  - [ ] H9b: reweighted anti-Rayleigh (upweight own-facts that stay loud &
        wrong-label facts that go quiet); expect d3 → 1.0.
  - [ ] H10: mixed economy per label (some detector taps, some veto taps),
        matching fleet motif census; construct via 2 directions/label
        (max-eig + min-eig) with D = [+1, −1] taps... note m budget: m=d
        means sharing — neurons must serve as detector for one label AND
        silencer for another. Tap-graph assignment problem.
  - [ ] H11: exact silence via linear algebra: own-fact constraints
        v·x=0 form a bipartite graph (t1-tokens vs t2-tokens, 4d edges);
        forest components admit exact zeros, cycles give parity obstructions.
        Analyze cycle structure of label subgraphs; build v from spanning-
        forest solution + least-squares on cycle residuals. This may explain
        WHICH labels can be silenced perfectly (and why silence beats
        detection: nulling is linear, detecting is not).
- FLEET ANALYZED (H2/H3/H7): 68/70 frontiers found (2 d=2 seeds stuck at
  dense acc 0.5). Hungarian-aligned canonical similarity is LOW: mean L-cos
  0.66 (d2) → 0.28 (d8), D-cos ~0.43-0.56 — seeds do NOT share one weight
  solution; H2 refuted at weight level. What IS universal (H3-ish):
  (a) taps/label ≈ d/2 (1.0, 1.2, 1.7, 1.9, 2.4, 2.9, 3.9 for d=2..8);
  (b) default (zero-tap) labels common for d≤5, absent d≥6;
  (c) ~half of taps negative at all d;
  (d) H7: in 2-tap rows the pos/|neg| ratio is most often exactly 1.0
      (balanced excitation/veto), NOT the 0.4 seen in the one d4 model;
      ratio drifts lower at d≥5 (median 0.36-0.7). 0.4 was not a constant.
  CONCLUSION: the target of hand-coding is a solution FAMILY (tap-graph
  motifs + constraints), not a fixed weight matrix. Figure:
  results/multiseed_similarity.png. Next: H1b iterative reweighted Rayleigh,
  d3 error analysis (task 12), H6 tap-ablation.
- H1 TESTED (partial fail): Rayleigh detectors, D=I, acc at ceiling:
  d2 0.94, d3 0.78, d4 0.69, d5 0.53, d6 0.44, d7 0.36, d8 0.39.
  Trained gets 1.0 everywhere here. H4 (ridge D on Rayleigh h): NO gain
  (±0.02) — readout not the bottleneck; L must encode more than one
  detector direction per label. New ideas queued: iterative reweighted
  Rayleigh (boost misclassified facts each round — still eig+reweight, no
  GD); study WHICH facts Rayleigh misses at d3 (token overlap structure?);
  fit trained L rows: are they close to generalized eigvecs of anything?
- (init) Created log. Launched 10-seed × d2..8 prune fleet in background
  (multiseed.py, snapshots to tiny_models/sym_random/multiseed/);
  implemented H1 (rayleigh.py).
