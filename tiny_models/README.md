# Tiny-model reports

One folder per setting; each contains per-model `d{n}.md` reports (capacity,
folded-embedding/unembedding weight imshows, full logit tensor, worked
examples by margin tier), saved weights, and a README table.

| folder | architecture | labels | sizes |
|---|---|---|---|
| [bilinear_random/](bilinear_random/README.md) | `D((Lx) ⊙ (Rx))` | random (memorization) | d = 1–4 |
| [sym_random/](sym_random/README.md) | `D((Lx) ⊙ (Lx))`, R = L | random (memorization) | d = 1–8, 16, 32 |
| [sym_sequential/](sym_sequential/README.md) | `D((Lx) ⊙ (Lx))`, R = L | sequential = rule `label = t1 // 2` | d = 1–8 |

Notes:
- Scaling family: `V_in = 2d`, `V_out = d`; hidden width m is parameter-matched
  to the ReLU MLP's 5d² (asymmetric: m ≈ 5d/9; symmetric: m = d).
- Random labels saturate the 4d² ceiling through d = 12 and break at d = 16
  (960/1024), so d = 16 and 32 are the first reports with populated
  "below the margin" / "not memorized" tiers.
- Sequential labels are a learnable rule, not memorization — those models hit
  100% on all pairs at every size tested (verified up to d = 64).
