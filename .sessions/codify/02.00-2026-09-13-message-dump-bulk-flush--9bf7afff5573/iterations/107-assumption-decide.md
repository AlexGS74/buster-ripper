# 107 · Assumption Decide — `S21` CE-2 residual routing after the run-49 gate repair

## Orchestrator Prompt (verbatim)

> You are a subagent spawned by another session.
> Session ID: 02.00-2026-09-13-message-dump-bulk-flush--9bf7afff5573
> Worktree root: /Users/alexhubitski/Projects/buster-ripper
> Mode: composition routing (S21 residual after the run-49 gate repair)
> Assumptions directory: /Users/alexhubitski/Projects/buster-ripper/.sessions/codify/02.00-2026-09-13-message-dump-bulk-flush--9bf7afff5573/assumptions/
> Registry root: /Users/alexhubitski/Projects/buster-ripper/invariants/
>
> Composition id to disposition: `S21.md` (`COUNTEREXAMPLE`, one residual CE after the run-49 repair). Apply the Reach Gate before deciding, then one disposition, naming the task/ledger row, the counterexample, reach basis, evidence reference, and required proof.
>
> The residual, as the composer recorded it: prior CE-1 is repaired (`_content_lines` structurally consumes the `@@ … @@` control line and recovers the glued first content line) and prior CE-3 is repaired (`rendering_section` preserves the rendering's trailing newline instead of `.strip()`ing it). What remains is CE-2: the expected-side renderings are read from the `self-contained` artifact through `read_text` (`tests/qa/harness.py:150-154`), whose default universal-newline mode converts `\r\n`/`\r` to `\n` *before* `production_diff_text` computes the expected diff, while production's diff is computed *before* the write over raw renderings. A line-ending-only change (turn N−1 `line1\r\nline2`, turn N `line1\nline2`) yields `expected=[]` and `observed=['-line1\n', '+line1\n']`, failing `assert observed == expected` at `test_qap_ms001.py:280`. Reach names `_write_diff_turn` (`dump.py:180-189`) → gate `test_qap_ms001.py:266-283`.
>
> The question you must answer on the bytes: is this reachable in the gate's own scenarios, or latent? The gate's own bodies are `turn_body` markers written by the harness, which writes text through `write_text` and therefore LF; no CRLF body exists in `tests/qa/`. But the milestone's contract says "today's content and line format", and a client could send `\r\n` inside message content, which `_render_body` would carry into the rendering and production's diff would treat as a change. Say whether a line-ending-only drift is a shape this gate must catch, and whether the fix is a one-line read-mode change in the gate (`read_text(newline="")` or an explicit binary read) or something larger.
>
> Route it as a gate repair if it is worth fixing before the boundary closes, or record it as latent/inert with its named owner if the gate's own inputs cannot produce it. Do not reopen `T021`/`T021.001`/`T021.002` (accepted-local) and do not touch production code: production's writer is verified correct by `P001` and the gate is the only surface.
>
> Standing constraints: `MS-001`'s Yield Phase is `SEMANTIC-APPROVE` at Checkpoint 244, the final Codex review is clean of new actionable findings, and `resolved` plus phase acceptance are withheld behind this routing. `S22` is `SAFE`; `S23` is `SAFE`; `S18` stands as re-read at run 47. `FK-011` remains the session's only open Fork; a question is admissible only where materially different outcomes remain unranked after base parity and `G-003` are applied.

Result: **ASSUMPTIONS: 1 routed | 1 absorbed | 0 questions | 0 breaks | 0 malformed | 0 accepted | 0 capped**

`S21` CE-2 is **real and latent**: the gate's own bodies cannot produce it, but it is the same
oracle defect `T021.001` claimed to repair, and the one admissible remediation is a small gate read-mode
change plus one focused regression. Absorbed into `T021.003` under `T021`; ledger `DC-S21-r2`. No question was
admissible — one remediation remains, and no alternative preserves the contract. `S21`'s single post-repair
recomposition allowance is consumed by this pass; a further non-`SAFE` `S21` composition is capped by rule.
`FK-011` is untouched and remains the only open Fork.

## Reach and disposition

| Finding | Reach | Reach basis | Disposition | Task / ledger |
|---|---|---|---|---|
| `S21` CE-2 · artifact-read line identity | **`latent`** | the only consumer is the QA gate `tests/qa/test_qap_ms001.py:266-283`; its own `TC-107`/`TC-108` bodies are `turn_body` markers, the harness builds every body from LF-only Python strings (`json.dumps`), and **no `\r` byte exists anywhere under `tests/qa/`** (measured: all eight modules LF-only) — so no gate scenario produces the CRLF rendering pair. The same read-mode asymmetry is also the false-pass surface for a writer that normalizes line endings on a mixed change, which the gate's marker/coarse checks do not separate. | **`absorbed`** → `T021.003` | `DC-S21-r2`; task row `plan-summary.md`; report §`S21` below |

## `S21` CE-2 — reproduced on the bytes, then routed

**Reproduction (this pass, `HEAD 14dcec8`, clean production tree).** `_render_body` over turn N−1 content
`"line1\r\nline2"` renders `'\n\n[message 0 / user]\nline1\r\nline2\n'`; turn N content `"line1\nline2"` renders
`'\n\n[message 0 / user]\nline1\nline2\n'`. Production's `_unified_diff` + `"".join` writes
`-line1\r\n`/`+line1\n`. The gate's observed read of that file (`read_text`, universal newlines) yields
`['-line1\n', '+line1\n']`. The gate's expected side reads the `self-contained` artifact through the same
`read_text`; both renderings arrive as `'\n\n[message 0 / user]\nline1\nline2\n'`, so `changed_lines` is `[]`.
`assert observed == expected` fails with **`expected=[] observed=['-line1\n', '+line1\n']`** — the composition's exact
numbers, reproduced. Reverse direction (LF → CRLF) and a lone-`\r` change behave identically. A temporary
`read_text` with `newline=""` makes both sides agree (`['-line1\r\n', '+line1\n']`) in all three directions.

**Why the gate's own scenarios cannot produce it.** No `\r` byte exists in any of the eight `tests/qa/*.py` modules;
the documented-format bodies are `turn_body` markers, and `json.dumps` emits no raw CRLF. The only CRLF renderings in
the file are the in-memory CE-2 sensitivity pair (`:665-674`), which never reaches `read_text` or
`assert_documented_contents`. **Reach `latent`, not `reached`.**

**Must this gate catch a line-ending-only drift?** The gate's *scenario set* need not contain one: its contract is
"today's content and line format" over the bodies it chooses, and every body is LF. The gate's *oracle* must still be
correct for any rendering it is given — `T021.001`'s own required behavior says "a correct writer must not be rejected on
CE-2", and that is not met on the artifact path. The same asymmetry is a false-pass surface for a writer that normalizes
line endings on a mixed change (non-empty diff, so the coarse `:268` check cannot separate it). **The oracle is the defect;
a CRLF E2E body is additive coverage, not a precondition of the repair.**

**Is it one line or larger?** The read-mode change is one line in `tests/qa/harness.py` (`read_text(path, newline="")`
or a raw/binary sibling) plus two call-site arguments in `assert_documented_contents` (`:267` observed diff, `:454`
expected renderings), because both sides must be read the same way. It is **not sufficient alone**: the existing CE-2
regression is in-memory (`:665-674`), so it passes before and after; the proof must round-trip a CRLF rendering through a
written artifact. That regression is the required proof; a CRLF body in `TC-108` is optional coverage, not a precondition.

**Required proof.** A focused regression writes a `self-contained` artifact holding `previous = "\n\n[message 0 / user]\nline1\r\nline2\n"`
and `current = "\n\n[message 0 / user]\nline1\nline2\n"` through the writer's `"\n\n".join(records) + "\n"` shape, reads it
with the gate's own `read_text`, extracts with `rendering_section`, and asserts the recovered renderings are the raw ones and
`changed_lines == diff_changed_lines` of the writer's own `"".join(unified_diff(...))` — **red on the pre-repair reader**
(`expected=[]`, `observed=['-line1\n', '+line1\n']`), green after. It also rejects the writer that drops the line-ending change
(`"-line1\r\n"` → `"-line1\n"`). The existing CE-1/CE-2/CE-3 in-memory regressions stay green, the two `T021` mutation
replays still red, unmutated `QAP-MS001` 16/16, `QAP-MS002` 15/15, `QAP-MS004` 6/6, hermetic unchanged; zero production,
config, plan, contract or registry bytes.

**Invariant cross-reference: Neutral** — zero `active`/`wip` invariants exist; every registry `Invariant:` row is `planned`.
The repair serves the planned `PI:T002/dump_format/documented_format_is_retained` and its witness `TC-108`; no duplicate candidate,
no authority created, no `invariants/` byte written.

**Premise gap (disclosed, not routed).** CE-2's mechanism rests on `open`'s default `newline=None` universal-newline translation,
which is not one of `S21`'s nine member rows. Same disposition as `QB:assumption-registry/s22-ambient-proxy-premise`: re-extract it
as a `lang:python` atomic proposition before `S21` is promoted; owning route: assumption collector. The gate repair does not wait for it.

## Seam and ledger state

- `assumptions/seams.md` — `S21` stays `recheck` / `absorbed` / `—`; its member list, Subjects, boundary value and every dimension
  are untouched. One §Run-51 changelog records the residual, `T021.003`, and the consumed recomposition allowance.
- `assumptions/claims.md` — rows 426–431 and 438–440 stay `HOLD` / `absorbed`; no row moved.
- **Recomposition allowance: consumed by this pass.** `098` left `S21`'s single post-repair recomposition allowance intact;
  the run-50 recheck was that recomposition and it found CE-2, and this pass is the routing that spends the allowance. Any further
  non-`SAFE` composition of `S21` is a later one and is capped by rule: Reach First, accept what it permits, route nothing else,
  and give the remainder to the user in one message. Only a fired `Invalidates On` trigger (1–8) may reopen the seam.
- **No `## Capped seams` entry is written**: nothing is being withheld behind a cap today.

## Receipt (what this pass wrote)

| File | What moved |
|---|---|
| `iterations/107-assumption-decide.md` | **new** — this report |
| `plan-summary.md` | `T021.003` row + pointer block; `Last Updated` |
| `handoff-context.md` | `T021.003` Planned Task Context; `DC-S21-r2`; §Current Status/NEXT; trim of the consumed `T022.001` block to the archive |
| `handoff-archive.md` | the moved `T022.001` block and the superseded `Last Updated`/`Current Status` bullets, verbatim |
| `plan-completed.md` | the superseded `Last Updated` paragraph, verbatim |
| `assumptions/seams.md` | §Run-51 changelog for `S21` (cells untouched) |
| `chronicle.md` | one line for this pass |

Zero production, config, test, `decisions.md` or `invariants/` bytes moved. The only Git operations were `rev-parse HEAD` and
`status --porcelain`; no `.git/` path was read, listed or followed.

## Open

- `T021.003` implementation and local Yield — owner: the dispatcher's `forward` → `yield` route; it is the last gate repair before
  the closing Yield Red check.
- The `S21` post-implementation recomposition after `T021.003` — owner: the dispatcher's assumption gate. It is a later composition and
  is **capped** if non-`SAFE`; its remainder goes to the user in one message.
- `MS-001`'s `resolved`, phase acceptance and the `Delivered` entry — owner: the closing Yield Red check.

## Next

Dispatch `T021.003` (gate-only, `lite`-sized), then local Yield, then the capped post-fix `S21` recomposition, then Yield Red.

```text
ASSUMPTIONS: 1 routed | 1 absorbed | 0 questions | 0 breaks | 0 malformed | 0 accepted | 0 capped
```
