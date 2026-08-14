# RRS — relearn resistance score, forget05

RRS = mean_t [control_rouge(t) − subject_rouge(t)] over relearn steps; control = never-knew retain95.
**RRS > 0 = genuine resistance; < 0 = head start for the attacker.** Headline = min over lrs.

| subject | lr | RRS (rouge) | RRS (prob) |
|---|---|---|---|
| ours min-token γ4 (selected) | 1e-05 | -0.043 | -0.124 |
| ours all-token γ4 (deep suppressor) | 1e-05 | -0.067 | -0.164 |
| NPO tuned lr 2e-5 | 1e-05 | +0.004 | -0.070 |
| ours v3-CE | 1e-05 | -0.048 | -0.192 |
| hybrid joint | 1e-05 | -0.061 | -0.181 |
| hybrid seq-all (pin on NPO) | 1e-05 | +0.081 | +0.034 |
| hybrid seq-min | 1e-05 | -0.035 | -0.156 |
| AltPO | 1e-05 | -0.143 | -0.307 |
| t25_altpo_pin_s0 | 1e-05 | +0.030 | -0.076 |
| t25_coherent_s0 | 1e-05 | -0.240 | -0.431 |
| t25_far_s0 | 1e-05 | -0.293 | -0.516 |
| ours min-token γ4 (selected) | 5e-05 | -0.014 | -0.078 |
| ours all-token γ4 (deep suppressor) | 5e-05 | -0.018 | -0.111 |
| NPO tuned lr 2e-5 | 5e-05 | -0.013 | -0.071 |
| ours v3-CE | 5e-05 | -0.018 | -0.076 |
| hybrid joint | 5e-05 | -0.028 | -0.095 |
| hybrid seq-all (pin on NPO) | 5e-05 | +0.000 | -0.055 |
| hybrid seq-min | 5e-05 | -0.004 | -0.082 |
| AltPO | 5e-05 | -0.021 | -0.084 |
| t25_altpo_pin_s0 | 5e-05 | -0.009 | -0.084 |
| t25_coherent_s0 | 5e-05 | -0.013 | -0.110 |
| t25_far_s0 | 5e-05 | -0.021 | -0.096 |

**Headline (min over lrs):** ours min-token γ4 (selected): -0.043, ours all-token γ4 (deep suppressor): -0.067, NPO tuned lr 2e-5: -0.013, ours v3-CE: -0.048, hybrid joint: -0.061, hybrid seq-all (pin on NPO): +0.000, hybrid seq-min: -0.035, AltPO: -0.143, t25_altpo_pin_s0: -0.009, t25_coherent_s0: -0.240, t25_far_s0: -0.293
