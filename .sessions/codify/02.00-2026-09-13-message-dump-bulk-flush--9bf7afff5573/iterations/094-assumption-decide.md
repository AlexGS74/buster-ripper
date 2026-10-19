# 094 · Assumption Decide — external-review routing (CR5, delta after CR4)

Routes `CR5-F-01` (P3) from the fifth external review — the delta `c5e2be4…HEAD` — plus the one record-only
item the same review raised without calling it a finding (the `§A` label in
`evidence/t019-002-fail-first-and-replays.txt`). **Zero absorbed, zero questions, zero breaks, zero malformed,
one accepted, zero capped.** No production byte, no test byte, no `invariants/` byte and no `decisions.md` byte was
written; `FK-011` remains the session's only open Fork and no `codify reserve-decision-record` was run. `MS-001`'s
`SEMANTIC-APPROVE` at Checkpoint 213 is untouched: nothing here opens a task, a wave or a recomposition, and Yield Red
is still the next act.

**Review round: 5 (late: yes, round 5 by count and the only finding is a P3 docstring inaccuracy — unusual or rare by the
finding table, with no normal-use finding in the round for it to ride along with).**

## Pre-flight

- **Session identity: verified.** `session-identity.json` → `session_uid: 9bf7afff-5573-402a-b102-827423a1dded`, which
  matches the dispatch handle's `--9bf7afff5573`. Not `BLOCKED-SESSION-IDENTITY`.
- **Repository basis.** `git rev-parse HEAD` from the supplied worktree root →
  `6a2069b3f21fe9c439475aae9406f7eafa582b30` (Checkpoint 213). `git status --porcelain` was empty before the first
  mutation and empty again after the last revert; the two mutated files hash to their `§E` pins
  (`normalize.py` `sha256:87ebcd05…`, `routes/count_tokens.py` `sha256:c6d6f924…`), so **the bytes this pass measured are
  the bytes CR5 reviewed** — not a later tree and not a mid-mutation state.
- **Git metadata boundary honored.** The only Git operations were `rev-parse`, `log`, `diff`, `show <rev>:<path>`, and
  `checkout <rev> -- <path>` / `checkout HEAD -- <path>` on two named files, each revert verified by `git status
  --porcelain`. Nothing under `.git/` was read, listed or followed as data.
- **Claims and seams.** `assumptions/claims.md` carries **zero live `Routing: pending` rows**; no composition verdict was
  supplied and no seam carries a non-`SAFE` verdict awaiting a disposition here. Standing constraints respected: `S1`–`S15`
  terminal, `S16`–`S19` recomposed with their allowances already spent, `S20` terminal. **Third-party subjects only for
  ledger rows: this pass created no `repo:` claim row and no `repo:` seam.** No closure ledger entry is owed — no finding was
  absorbed.
- **Invariant cross-reference applied to both items** (skill `invariant-cross-reference`): no session `invariants.md` exists;
  `invariants/index.md`'s handle index holds the dump, server, route, package and `CL:` records — **no entry speaks to a test
  docstring, a witness attribution, or an evidence label**; the `PI:` rows are `Status: planned`, the `CL:`/`CO:` rows are
  descriptive evidence with `HOLDS`/`SAFE` verdicts, and the four `OI:` provenance blocks are overrides — none reaches
  either item. Zero `active`/`wip` invariants are contradicted; no `Override:` matches. Classifications below are evidence
  for routing, not dispositions; no permanent `CL:`/`CO:` record was read as authority or promoted.

## `CR5-F-01` — re-measured, not adopted

The finding is the `test_a_shape_base_already_tolerated_stays_tolerated` docstring at
`tests/test_count_tokens_parity.py:95-103`. It claims the four already-tolerated shapes redden on "a pass that narrows this
route's normalization too far — or that hardens the loops and changes what the estimate is computed from". The reviewer says
neither is true: the class stays at 4 passed under both, while the digest class and the `/v1/messages` posture class redden
instead.

Measured on `6a2069b`, each mutation applied alone and reverted byte-exact:

| Mutation | Result | What reddened |
|---|---|---|
| (i) delete `strip_billing_nonce(body)` at `count_tokens.py:87` — the extreme narrowing | **3 failed / 32 passed** | `TestCountTokensKeysOnTheNormalizedDigest` ×3 (`{'input_tokens': 47/54/64} != {'input_tokens': 1234}`). The four-shape class: **4 passed** |
| (ii) estimate source → `estimate_tokens(normalize(body, session_id="")[0])` | **3 failed / 32 passed** | `test_a_malformed_member_still_answers_an_estimate` ×3 (500 != 200, the `AttributeError` re-entering the route). The four-shape class: **4 passed** |
| (iii) harden `normalize()`'s element loops (`isinstance` guards in `_normalize_tools` / `_strip_cache_control_from_messages`) | **3 failed / 32 passed** | `TestMessagesMalformedMemberPosture::test_a_malformed_member_still_answers_bas_own_500` ×3 (502 != 500). The four-shape class: **4 passed** |
| (iv) my own control: make `_read_body` answer 400 on the unparseable body | **1 failed / 34 passed** | `[unparseable-body]` (400 != 200). The other three params green |

So the reviewer's conclusion is confirmed by execution: **the four-shape class is a real over-reach guard — mutation (iv) reddens
it — but neither trigger the docstring names reddens it.** The class's actual witness is a pass that refuses a body base
tolerated, not one that narrows the key or hardens the loops.

**One supporting measurement in the review is imprecise, and is corrected here rather than repeated.** The review says
mutation (ii) "reddens nothing anywhere because no test asserts a `count_tokens` estimate value". The second half is true — no
test asserts an estimate *value*, so a value-changing estimate would pass silently — but the stated mutation does redden something:
routing the estimate through `normalize()` reintroduces the CR4-F-01 500 for the three malformed-member bodies. The four-shape
class is still green, which is the reviewer's conclusion; only the "nothing anywhere" phrasing is wrong. The pass relies on its
own measurements, not on the review's summary of them.

## `CR5-F-01` — `accepted` by Reach First, no closure required

- **Reach verdict: `unreachable` in the production sense** — a docstring is read by no production caller and no consumer;
  the only reader is a future maintainer. This is Reach First's **inert-wording** class (a docstring), not the
  `unreachable`-input class: the suite is sound and the guard exists, so nothing is dismissed as a defect.
- **Disposition: `accepted`.** One line under `handoff-context.md` → §Accepted risks with this Reach citation, and one doc
  line at `QB:tests/count-tokens-parity/over-reach-docstring-witness`. No task, no Fork, no wave, no re-extraction; the
  seam does not stay open.
- **Deterministic Closure Gate — read, and not reached.** Had the finding not been inert, exactly one aligned remediation
  exists: rewrite the docstring to name the witnesses that do redden (the digest class for a narrowed key, the `/v1/messages`
  posture class for hardened loops, and a stricter `_read_body` for the class's own over-reach direction). The reviewer's
  alternative — pin the estimate with a value assertion — is **inadmissible**: `090-01-yield`'s F-1 records the nonce-sized
  estimate residue as deliberate, and no done-when clause asks for the value, so pinning it would turn a settled record into a
  test's authority. With one admissible remediation there is no `question`; but Reach First assigns `accepted` before the gate,
  so the remediation is recorded as the doc line rather than a task.
- **Forward touch before the boundary closes: no.** In a late round an unusual-or-rare finding's fix reads `not cheap`, and
  the only permitted repair is a ride-along in a repair wave already entering that code. No repair wave is scheduled: `MS-001`
  is `SEMANTIC-APPROVE` at Checkpoint 213, the assumptions recheck and Yield Red are the next acts, and opening a brief
  now would be the brief of its own the late-round rule forbids. **It belongs to the next pass that edits
  `tests/test_count_tokens_parity.py`**, which is the trigger recorded on the doc line. Nothing in `MS-001`'s approved
  boundary moves, and Yield Red reads the same bytes.

## Record-only item — the `§A` label in `t019-002-fail-first-and-replays.txt`

The reviewer's observation is accurate. `§A` is labelled "final test code", but its run prints **3 failed / 12 passed**
(15 witnesses) while the final module carries **35**. Measured, not inferred:

- `git show c5e2be4:buster_ripper/normalize.py | grep strip_billing_nonce` → **zero hits**; the file has only
  `def normalize` (`:202`), and `git show c5e2be4:buster_ripper/routes/count_tokens.py` prints
  `body, _ = normalize(body, session_id="")` at `:87` — exactly the frame `§A` shows.
- The final module imports `strip_billing_nonce` (`tests/test_count_tokens_parity.py:36`), so it **cannot collect** at those
  bytes: `ImportError: cannot import name 'strip_billing_nonce' from 'buster_ripper.normalize'` (measured by checking the
  final test module out against the pre-fix production files, then reverting both).

So `§A` is an earlier draft of the fail-first module, not the final one. **Decision: corrected in place, annotation only.**
The transcript body is the dated record and stays byte-identical; the label is a claim *about* the body and was false, so the
label is the one thing corrected. The correction quotes the old wording and names the reason, so a future repair cannot take
`§A` for the final module's fail-first state and chase a witness count that never ran there. **The fail-first direction
itself is untouched and independently reproduced**: the three failures print the escaping `AttributeError` frames from
`_normalize_tools` / `_strip_cache_control_from_messages` through `count_tokens.py:87`, which is the pre-fix route at
`c5e2be4`. The pointers that cite `§A` — `quality-backlog.md:106`, `handoff-archive.md:1062`, and `plan-summary.md`'s
`T019.002` row — all claim only "the trio is red at pre-fix bytes", which is true of the draft and unchanged by the
correction.

## Invariant cross-reference

- `CR5-F-01` — **Neutral**. No session or registry entry speaks to a test docstring or a witness attribution; the
  nearest surfaces are the `CL:` records for `count_tokens._read_body` and `handle_count_tokens`, which are claims about
  behavior, not authority over prose. No duplicate candidate is created.
- Record-only item — **Neutral**. No invariant speaks to an evidence label.
- Triad: `git status --porcelain -- invariants` empty before and after. Zero registry bytes created, edited, superseded,
  deleted or promoted.

## Context budget

`handoff-context.md` entered at **1200 lines** (already over its 200-line cap; the standing disclosure is
[093-yield-phase](093-yield-phase.md) and the earlier passes). This pass **moves 42 lines out verbatim** — the superseded
§Closure-packet routing — 2026-10-07 (CR4 batch) narrative → `handoff-archive.md` §Trimmed 2026-10-17
(`094-assumption-decide`), leaving a pointer line — and **adds** the CR5 routing section, the one §Accepted risks line, the
§Current Status update and the pointer: **1200 → 1191**, still over cap. The deep trim remains
`phase-boundary-finalize`'s. `plan-summary.md` is untouched at **391** lines, under its 400-line cap.

## Receipt (what this pass wrote)

| File | What moved |
|---|---|
| `iterations/094-assumption-decide.md` | **new** — this record: the four mutations, the correction to one review measurement, the disposition, and the record-only decision |
| `quality-backlog.md` | one doc line — `QB:tests/count-tokens-parity/over-reach-docstring-witness`, with the correct witness names and the next-touch trigger |
| `handoff-context.md` | §Current Status (CR5 result + next act unchanged), new §Closure-packet routing — 2026-10-17 (CR5 batch) with the disposition table and the no-Forward-touch ruling, one §Accepted risks line for `CR5-F-01`; the CR4 routing section moved out verbatim with a pointer line left in its place |
| `handoff-archive.md` | the CR4 §Closure-packet routing section, verbatim, under §Trimmed 2026-10-17 (`094-assumption-decide`) |
| `evidence/t019-002-fail-first-and-replays.txt` | §A label corrected in place (annotation only; transcript body byte-identical) |
| `evidence/INDEX.md` | one bracketed correction sentence on the existing `t019-002-fail-first-and-replays.txt` row |
