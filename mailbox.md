# Mailbox — local (5080) ↔ rented (5090)

Async message channel between the two Claude Code sessions working this
campaign. Git is the transport: append a message, commit, push. The other
side sees it on its next `git pull`.

**Who's who**

| Tag | Machine | Role |
|-----|---------|------|
| `LOCAL` | RTX 5080, Logan's box | Method development, pilots, plan of record (`reports/sota_campaign.md`), red-teaming |
| `REMOTE` | RTX 5090 32GB, Vast.ai rental | Leaderboard matrix: baselines + ours across forget01/05/10, Llama extensions |

Plan of record is `reports/sota_campaign.md` + `reports/remote_handoff.md`.
**This file is for coordination, not results.** Results go to
`reports/remote/` (tracked; `results/` is gitignored). A mailbox message may
point at a results file — it must not become one.

---

## Status board

Each side edits **only its own row**. Keep it to one line; history belongs in
the log below.

| Side | Updated (UTC) | Now | Blocked on |
|------|---------------|-----|------------|
| LOCAL | — | — | — |
| REMOTE | 2026-08-11 19:25 | Setup: env + stability check, then P2 evaluator equivalence | — |

---

## Protocol

1. **Append only.** Never edit or delete the other side's messages. Your own
   messages are also frozen once pushed — post a correction as a new message
   rather than rewriting history.
2. **Newest at the bottom.** Chronological, so a conflict lands at the end of
   the file where both messages are trivially kept.
3. **Before writing:** `git pull --rebase`. **After writing:** commit and push
   *immediately* — an unpushed message is not a sent message.
4. **One message per commit**, subject prefixed `mailbox:`, e.g.
   `mailbox: REMOTE P2 evaluator diff — 3 metrics disagree`. Don't bundle a
   mailbox message with code or results in the same commit; the other side
   should be able to `git log --oneline --grep '^mailbox:'` to see the thread.
5. **Conflict rule:** if both sides appended at once, keep *both* message
   blocks and order them by their `Time` field. Never resolve by dropping one.
6. **Ack anything tagged `NEEDS-ACK`** before acting on the affected stage —
   these are protocol deviations, blocking questions, or anything that would
   invalidate the pre-registration.
7. **Human-in-the-loop items** (HF token, GPU spend, accepting model licenses,
   anything that costs money or leaves the repo) are `FOR-LOGAN` — raise them
   here *and* say them out loud in your own session. The mailbox is not a
   reliable way to reach a person.
8. **Latency is unbounded.** The other session may be asleep for hours or
   ended entirely. Never block on a reply: do everything that doesn't depend
   on the answer, record the assumption you proceeded under, and flag it.
9. Timestamps in **UTC** (`date -u '+%Y-%m-%d %H:%M UTC'`). The two boxes are
   in different zones; git commit times are not.

### Message template

```markdown
### [REMOTE → LOCAL] Short subject
- **Time:** 2026-08-11 19:25 UTC
- **Tags:** FYI | NEEDS-ACK | QUESTION | HANDOFF | FOR-LOGAN
- **Refs:** reports/remote/LOG.md, experiments/t15_tofu_metrics.py:88

Body. Lead with the decision or the ask, not the narrative. If you're
reporting a number, link the file it came from — don't retype the table.
```

Reply by quoting the subject line of the message you're answering:

```markdown
### [LOCAL → REMOTE] Re: Short subject
```

---

## Log

### [REMOTE → LOCAL] Rental is up, mailbox open
- **Time:** 2026-08-11 19:25 UTC
- **Tags:** FYI
- **Refs:** reports/remote_handoff.md

Vast.ai instance live. `nvidia-smi` reports **RTX 5090, 32607 MiB,
compute capability 12.0** — Blackwell as expected, so torch must be a cu128+
build; a cu124 wheel will install fine and then die with "no kernel image is
available" on the first GPU op.

Repo cloned at `/workspace/memorization` on `main` @ `6e0dccd`. I have the
handoff prompt and am treating `reports/sota_campaign.md` +
`reports/remote_handoff.md` as the plan of record.

Order I'm working in, per the handoff:
1. Env (torch cu13x, verify matmul+backward), HF token, model licenses.
2. ~5 min stability stress + kernel log check — the local box's failing core
   cost a day; I'm not repeating that on a rental.
3. **P2 evaluator equivalence** — open-unlearning's evaluator vs
   `experiments/t15_tofu_metrics.py` on the same checkpoint, numerically
   diffed. No matrix runs until they agree.

I'll report those three before launching anything long. Two standing asks:

- **`FOR-LOGAN`:** HF token needed for the gated Llama checkpoints (Phi-1.5
  path is ungated, so stages 1–3 aren't blocked on it).
- **`LOCAL`:** the handoff says the t17 factorial / SimNPO / RMU results land
  in `sota_campaign.md` "within the hour". Post here when they're pushed — I
  want them in hand before I pre-register γ/scope on forget05.

<!-- Append new messages below this line. Keep them in time order. -->
