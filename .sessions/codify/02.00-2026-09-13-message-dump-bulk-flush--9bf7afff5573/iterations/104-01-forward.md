## Orchestrator Prompt (verbatim)

> T022.001

## Result: IMPLEMENTED — the `T022` drain's own fixed Correctness line is `[x]` with its DRAIN evidence, all four reporting surfaces state 8 routed / 11 closed (7 earlier, 1 fixed, 3 accepted risks), and the test comment at `tests/test_stream_relay_commit.py:51` says "is swallowed"; this pass re-derived every done-when clause on the current bytes and re-ran the focused module.

## Increment Brief

- Task `T022.001 · close the T022 drain's own fixed line`: verify the closure `DC-T022-Y-1-r1` (applied at Checkpoint 239 by [103](103-assumption-decide.md)) against the current tree and record the Forward pass.
- Anchors touched: none of the plan's Anchor Map rows. Scope is session text plus one test-comment word; no production referent moves.
- Will NOT create: a backlog line, a production helper, a test, a roadmap row, a registry/invariant record, or a QA gate.
- Reuse evidence: the fixed witness is the existing `write-position-os-error` param in `POST_COMMIT_FAULTS` (`tests/test_stream_relay_commit.py:54+`); the counts come from `quality-backlog.md`'s own `## Correctness` paragraph.
- Inventions: none.

## Done-when re-derived

| Clause | Result | Evidence |
|---|---|---|
| `sed -n '26p' quality-backlog.md` starts `- [x]` | holds | `quality-backlog.md:26` is `- [x] tests/test_stream_relay_commit.py:48-54 …` with its `DRAIN 2026-10-18 (T022)` and `CLOSED … DC-T022-Y-1-r1` evidence |
| Four surfaces state 8 routed / 11 closed (7/1/3) | holds | `quality-backlog.md:14`; `101-01-forward.md:32` and `:80`; `plan-summary.md:353` (`T022` row); `handoff-context.md:17` |
| Paragraph-aware count: 8 open / 11 closed, 8 routed | holds | `## Correctness` block: 19 entries, 8 `- [ ]` each carrying the literal `→ routed:` marker, 11 `- [x]`; closed split 7 earlier + 1 fixed in this wave + 3 accepted risks (`:23`, `:45`, `:46`) |
| `tests/test_stream_relay_commit.py:51` says "is swallowed" | holds | `nl -ba` line 51: `# needs two chunks: the first write is swallowed and the second raises.` |
| Focused module 14 passed | holds | `uv run pytest tests/test_stream_relay_commit.py -q --no-cov` → **14 passed** (`runs/104-01-forward-focused.txt`) |
| `codify scan path tests/test_stream_relay_commit.py --git --detailed` clean | holds | `No git changes found … nothing to analyze` (the change is already in Checkpoint 239); full-file scan shows only the three pre-existing test helpers (two `P:5`, one `P:4`) (`runs/104-01-forward-scan*.txt`) |

## What changed

| Surface | State | Evidence |
|---|---|---|
| `quality-backlog.md:26` | `[x]` with DRAIN + CLOSED evidence; no edit needed this pass | `103-assumption-decide` applied it at Checkpoint 239 |
| `quality-backlog.md:14` | `8 open, every one routed … 11 closed — 7 from earlier drains, 1 fixed in this wave, 3 accepted risks` | paragraph-aware count above |
| `101-01-forward.md:32` / `:80` | both state `8 Correctness` / `8 routed / 11 closed` | `nl -ba` re-read |
| `plan-summary.md` `T022` row | `8 open (all routed) and 11 closed — 7 … 1 … 3` | row re-read |
| `handoff-context.md` §Current Status | `8 open (every one routed) / 11 closed (7 earlier, 1 fixed, 3 accepted risks)` | bullet re-read |
| `tests/test_stream_relay_commit.py:51` | "is swallowed" | `nl -ba` line 51; `git show 239bd57` confirms the one-word diff |

No file needed a correction: every required byte is already on `HEAD b84b224`. This pass's own writes are the iteration report and the two session files that record it.

## Failure-path evidence

No behavior path changed in this increment. The focused module's existing post-commit fault table — including `write-position-os-error`, which is the witness this line was closed for — ran green as part of the 14-case module (`runs/104-01-forward-focused.txt`).

## Capacity gate

`Capacity gate: not applicable - no additive production surface.` Touch set is session text plus one comment word in `tests/test_stream_relay_commit.py` (193 lines, Green). `codify scan path tests/test_stream_relay_commit.py --git --detailed` → `No git changes found … nothing to analyze`; the full-file scan prints only the three pre-existing test helpers (two `P:5`, one `P:4`), unchanged.

## Backlog pickup

Absorbed: `QB:test-stream-relay/write-position-injection` — this task closes its own line; the row is already `[x]` at `quality-backlog.md:26` with DRAIN evidence, re-read this pass. No other open line names the touch-set.

## Concerns

1. The closure was applied by the `assumption-decide` route at Checkpoint 239, so this Forward pass carries no new implementation bytes; it re-derives the clauses rather than re-landing them. Acceptance, `resolved`, and the `Delivered` entry stay with the closing Yield Red check.
2. The tree is not clean: 5 tracked concurrent edits (`tests/qa/*`, `.agents/skills/qa-project/SKILL.md`) and 50 untracked run-49 artifacts predate this pass and are left untouched; none is in this checkpoint.

## Next

Yield / the closing check reviews the closure on these bytes; `T022.001` remains `implemented`, not `accepted`, until then.
