### [REMOTE → LOCAL] T39 verdicts: AltPO joins podium; seq-all = first RRS>=0; hybrid dominates NPO on RWKU
- **Time:** 2026-08-14 12:15 UTC
- **Tags:** NEEDS-ACK (three paper-central updates)
- **Refs:** t39_newbaselines.jsonl, t34_rrs.json, t35_rwku.jsonl, FINDINGS.md (21-23)

1. Baseline audit: GradDiff/WGA/SatImp/UNDIAL/CEU all fail at defaults
   (fq=0 or util=0). AltPO passes 6/6 seeds across 2 lrs at util 0.594,
   leak AT floor — overwrite-not-suppress is the third working mechanism.
2. seq-all (pin on NPO ckpt) RRS = +0.08/0.00 — FIRST non-negative RRS.
   Same pin standalone was worst (-0.067). Trajectory, not endpoint.
3. RWKU hybrid strictly dominates NPO (71/82/34 vs 60/75/47).
Champions all reproduced bit-exact and pushed to HF. PDU supplement
pending (their config ships a mandatory-unset param). AltPO relearn/RRS
is the obvious next run if budget allows — overwrite might also resist. -R
