# RRS — relearn resistance score, forget05

RRS = mean_t [control_rouge(t) − subject_rouge(t)] over relearn steps; control = never-knew retain95.
**RRS > 0 = genuine resistance; < 0 = head start for the attacker.** Headline = min over lrs.

| subject | lr | RRS (rouge) | RRS (prob) |
|---|---|---|---|
| ours min-token γ4 (selected) | 1e-05 | -0.043 | -0.124 |
| ours all-token γ4 (deep suppressor) | 1e-05 | -0.067 | -0.164 |
| NPO tuned lr 2e-5 | 1e-05 | +0.004 | -0.070 |
| ours v3-CE | 1e-05 | -0.048 | -0.192 |
| ours min-token γ4 (selected) | 5e-05 | -0.014 | -0.078 |
| ours all-token γ4 (deep suppressor) | 5e-05 | -0.018 | -0.111 |
| NPO tuned lr 2e-5 | 5e-05 | -0.013 | -0.071 |
| ours v3-CE | 5e-05 | -0.018 | -0.076 |

**Headline (min over lrs):** ours min-token γ4 (selected): -0.043, ours all-token γ4 (deep suppressor): -0.067, NPO tuned lr 2e-5: -0.013, ours v3-CE: -0.048
