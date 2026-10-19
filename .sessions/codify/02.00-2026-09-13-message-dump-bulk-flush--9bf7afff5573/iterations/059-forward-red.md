# Iteration 059 — NOTICE (not a report): forward-red ID collision

**What this file is:** a placeholder recording a lost artifact. It is deliberately NOT a forward-red report, so no
reader mistakes it for one. `next-iteration-id` issued **059** twice within ~4 minutes to two concurrent
scoped-adhoc forward-red passes dispatched from the closure-packet routing at
[057-codex-review](057-codex-review.md).

## Sequence

| Time (exact only where this pass observed it) | Event |
|---|---|
| ≤10:20, verified | `iterations/` listed by this pass: newest file `057-codex-review.md` (10:06). No 058, no 059. |
| between 10:20 and 10:37, exact time not observed | This pass (`T013` header-parity: 057 F-4, G-1, G-2) asked for the next iteration id → **059**. |
| not observed by this pass | The sibling pass (`T014` image-slice: 057 F-1, F-2, F-3) asked for the next iteration id and received **059** as well. |
| before 10:37, inferred | The `T014` pass wrote its report to `059-forward-red.md`, then stamped `handoff-context.md` §Current Status citing it (its stamp was present at 10:39; `058-01-forward.md` landed at 10:30, so both passes were running inside that window). |
| 10:37, observed | This pass wrote its report to the same path, overwriting the `T014` report. The collision surfaced only when this pass went to stamp the baton and found the `T014` stamp already in place. |
| after 10:37 | This pass moved its own report to **[060-forward-red.md](060-forward-red.md)** and left this notice at the number it had clobbered. |

Everything in the "inferred" and "not observed" rows is reconstructed from filesystem mtimes and the handoff
stamp — this pass has no log of the sibling's calls. The one fact it did verify directly: at 10:37 the bytes at
`059-forward-red.md` were its own report (17,976 bytes, `T013` title), and no file containing the `T014` report's
`RED: 14 misreadings` string exists anywhere under `iterations/` now.

## State of the two records

- **`T013` header-parity pass** — intact at [060-forward-red.md](060-forward-red.md): RED 17 | 6 AMBIGUOUS | 5
  discretion footnotes + 1 structural restroking note | 6 excluded-with-citation. Its baton stamp cites 060.
- **`T014` image-slice pass** — its **verdict and the substance of all five blocking rows survive** in
  `handoff-context.md` §Current Status (RED 14 | 5 AMBIGUOUS | 8 footnotes, rows A-1…A-5 each with the file:line
  evidence: the undefined hash input behind the placeholder key, prune-before-loop 502/500 at `image.py:81/84`,
  refresh-on-resend vs insert-on-miss, the earliest-only detector that passes every mandated witness, and the
  global-reclamation claim against a session-scoped closure). **Unrecoverable:** its full attack table, its
  excluded-with-citation rows, and its 8 footnotes as written. Transcript recovery was attempted against
  `opencode.db`, `~/.codex/sessions/2026/10/05`, `~/.claude/projects` and `opencode/storage` — the live runner's
  tool-call payloads were not reachable read-only, and nothing was written to any store.

## For the orchestrator

Route `T014`'s five rows from its §Current Status stamp (the findings themselves are complete enough for Decide
disposition), **or** re-dispatch that one pass — its scope was 057 F-1/F-2/F-3 only, so a re-run costs one pass
and reproduces the table. `T013` needs no re-run: 060 is whole.

**Process note, not a finding:** two scoped passes dispatched from the same routing step can collide on
`next-iteration-id`, because the id is allocated at dispatch and written at the end of a multi-minute pass. If
concurrent scoped gates are routine, allocating the file at write time (or namespacing by task, e.g.
`059-forward-red-T013.md`) removes the race.
