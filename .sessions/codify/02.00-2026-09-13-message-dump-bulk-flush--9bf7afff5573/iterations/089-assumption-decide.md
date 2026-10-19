# 089 · Assumption Decide — closure-packet routing (CR4 batch, MS-001 boundary)

Routes `CR4-F-01` (P2) and `CR4-F-02` (P3) from [088-codex-review](088-codex-review.md), the verbatim fourth external
review of the post-`T019.001` bytes. Closure-packet mode: the packet is 088 itself, it names no assumption seam, so
resolution follows the owning route's acceptance gate and no composition is required. **One `absorbed`, one
`accepted`, zero questions, zero breaks, zero malformed.** No production byte, no test byte, no `invariants/` byte and
no `decisions.md` byte was written by this pass; `FK-011` remains the session's only open Fork and no
`codify reserve-decision-record` was run.

## Pre-flight

- **Session identity: verified.** `session-identity.json` → `session_uid: 9bf7afff-5573-402a-b102-827423a1dded`, which
  matches the dispatch handle's `--9bf7afff5573`. Not `BLOCKED-SESSION-IDENTITY`.
- **Repository basis.** `git rev-parse HEAD` from the supplied worktree root →
  `c5e2be44ba1a7ab984f02104472f06f849e4f773`, captured before the first routing write and used verbatim in
  `DC-CR4-F-01-r1`'s `Repository state`. `git status --porcelain` was empty apart from the untracked
  `iterations/088-codex-review.md`, so **the bytes this pass measured are the bytes 088 reviewed** — the condition 080
  checked for the CR3 batch, re-checked here rather than assumed.
- **Git metadata boundary honored.** The only Git operations were `rev-parse`, `log --oneline`, `diff --name-only`,
  `diff --stat`, `show <rev>:<path>` and `git archive <rev> | tar -x` into a scratch directory. Nothing under `.git/`
  was read, listed or followed as data.
- **Claims and seams.** `assumptions/claims.md` carries **zero live `Routing: pending` rows** — the three grep hits
  are prose in the run-40/run-42 routing changelogs (the same reading Checkpoints 199/200 recorded). No seam carries a
  non-`SAFE` verdict awaiting a disposition; no composition file was supplied. Standing constraints respected:
  `S1`–`S15` untouched and terminal, `S16`–`S19`'s single post-repair recomposition allowances not spent or extended by
  this pass, `S20` terminal per [086](086-assumption-decide.md). **Third-party-subjects only for ledger rows: this
  pass created no `repo:` claim row and no `repo:` seam.**
- **Invariant cross-reference applied to both findings** (skill `invariant-cross-reference`): no session
  `invariants.md` exists; `invariants/` holds `buster_ripper/dump` and `buster_ripper/server` invariant files plus the
  claim index — **no `buster_ripper/routes/*` or `buster_ripper/normalize` invariant file**, and every registry entry
  touching this session is `Status: planned`, none of which reaches either site. Zero `active`/`wip` invariants are
  contradicted. Classifications below are evidence for routing, not dispositions, and no permanent `CL:` record was
  read as authority or promoted.

## Re-established on current bytes — by execution, at all three points

The dispatch required the matrix re-run rather than trusted, because the whole weight of `CR4-F-01` is the claim that
base and the pre-range branch tolerated what `HEAD` does not. Three trees (`git archive` of `72e3a1c6`, `6508941`,
`c5e2be4`) were probed with the harness the package's own tests use — the real `create_app()` on a real
`aiohttp.test_utils.TestServer` — plus base's monolith under Starlette's `TestClient` against **a live upstream stub**,
which turned out to matter: with a dead upstream, base's `/v1/messages` answers 500 to *every* body including the
well-formed control, so the reviewer's "base 500s there too" is unverifiable without one. Everything the reviewer
reported holds; two things are corrected, both by measurement, and are recorded in the ledger rather than argued.

| Probe | base `72e3a1c6` | pre-range `6508941` | HEAD `c5e2be4` |
|---|---|---|---|
| `count_tokens` `{"tools":[null]}` | **200** | **200** | **500** ← `normalize.py:75` |
| `count_tokens` `{"messages":[null]}` | **200** | **200** | **500** ← `normalize.py:117` |
| `count_tokens` `{"messages":["x"]}` | **200** | **200** | **500** ← `normalize.py:117` |
| `count_tokens` `{"system":[null]}`, `{"messages":[{"content":[null]}]}` | 200 | 200 | 200 |
| `count_tokens` `[1,2]` (non-dict body) | 500 (`_session_id`) | 500 (`session.py:95`) | 500 (`normalize.py:52`) — **unchanged status** |
| `/v1/messages` — the same three bodies | **500** (`normalize` `:374`/`:402`) | **500** | **500** |
| `/v1/messages` — well-formed control, `{"system":[null]}` | **200** | 502 (dead upstream) | 502 |

Both axes run: `default` (STRIP_DATE off, vllm) and `stripdate-sglang` — the trio's statuses are identical on both.

1. **The regression claim holds at every point.** base's `count_tokens` (`buster_ripper.py:672`, body read `:690`,
   keyed `:696`; in this tree `buster_ripper_legacy.py:726`/`:744`/`:750`, byte-unchanged across the range) never
   calls `normalize`: `except Exception: body = {}` then `_session_id(body)`, which touches only `system`. The
   pre-range package matched it. `HEAD` does not, and the `AttributeError` escapes `handle_count_tokens` — so this is
   a delta-introduced 500 on an exposed route, of the class `DC-CR3-F-04-r1` already repaired for at `CR3-F-04`.
2. **The third raise site is real but not independently reachable today** — a correction to the relayed table. Under
   `STRIP_DATE` + sglang, `{"messages":[null]}` still dies at `:117`, because step 3 iterates the same list before
   step 4 does. `:192` becomes the raising frame only after `:117` is hardened (measured by simulation in the probe
   process, §6). Consequence: the reviewer's alternative candidate is not complete in its stated form; the complete
   form is three loops.
3. **The reviewer's parity claim about `/v1/messages` holds, and needed the stub to prove** — base's own `normalize()`
   raises the identical `AttributeError` at the identical three bodies (direct-call table *and* HTTP statuses against a
   live stub), while the control body answers 200 through the same stub. So the 500 there is base's, faithfully
   migrated, at both ends of the range.
4. **The narrowing's invariant is measured, not asserted.** `session_id()` reads `system` only (`session.py:93-98`)
   and `_normalize_system` is the only step that writes its text; over 9 bodies × 2 nonce states × 2 axes — the
   documented Claude Code shape with the billing nonce as string and as block, the nonce-only block that the strip
   deletes outright, `cache_control` on `system` blocks, tools reordering, an image body under `STRIP_IMAGES` — the
   digest after full `normalize()` equals the digest after the system step in **18 of 18 cases**. Rebinding
   `count_tokens`' call to the system step in-memory reproduces base's and pre-range's status column exactly.
   Transcript: `evidence/cr4-probe-matrix.txt` (9 sections, 385 lines, `sha256:e3d0482f…`).
5. **`CR4-F-02` holds on bytes**: `6508941` ends `…value.\n`, `HEAD` ends `0a0a`, the introducing commit is `9c90675`;
   `tests/test_dump.py` carried `0a0a` before the range and still does; no `.github/workflows`, no `[tool.ruff]`, and
   the tree's only lint row selects `F`, which excludes `W391`.

**Reach verdict, recorded once and used everywhere below: `latent`** — the label `CR3-F-04` established for exactly this class (a network-supplied payload producing a proxy 500 on an exposed route). The gate's literal reading would call it `reached`, since "exposed surfaces are reached by definition" and a `POST` of `{"tools":[null]}` is a cited path to the site; the distinction changes nothing here, because both labels carry the same rule — fix-or-pin — and nothing pins it: no witness anywhere sends a request body with a non-mapping element inside `tools` or `messages` at any route (`tests/test_upstream_answer_relay.py:163`'s `b"null"`/`b"[1, 2]"`/`b'"a string"'` are **upstream answers** relayed at `/v1/chat/completions`, not client bodies, and not array elements). So `CR4-F-01` resolves to **fix** on the route whose behavior moved, and to **parity** on the route whose did not.

## Deterministic Closure Gate — results

### `CR4-F-01` — `CLOSURE: absorbed`, one remediation, `Question required: no`

```text
CLOSURE: absorbed
Finding: CR4-F-01 (P2), iterations/088-codex-review.md §[P2]
Required outcome: /v1/messages/count_tokens answers an estimate for every JSON-object body, whatever its members'
  shapes, as base and the pre-range branch both do — while the stats lookup keeps keying on the post-strip `system`
  digest that /v1/messages records stats under.
Authority: the carried preserve-only invariant EI:buster_ripper.session/session_id/derived_after_normalize
  (plan-summary §Invariant Reconciliation, inside the user-approved plan; handoff §Notes; deliberately never
  registry-active per invariants/index.md §Source Sessions) · base parity, the session's settled migration contract
  (tests/test_header_parity.py:4, the direction T013-T017 and T018-T020 both applied) · G-003 ("Proxy behavior on
  /v1/messages is unchanged"; constraint column "No new endpoints, no normalization changes") · DC-CR3-F-01-r1's
  rejected-candidate list, which already forecloses routing out a rejection this branch introduced · DC-CR3-F-04-r1's
  rejection of catch-all widening in this same file, one pass ago.
Admissible candidates: one — narrow the call to the step the digest depends on.
Rejected candidates: harden normalize()'s element loops -> changes /v1/messages (500 -> 502 measured) on the route
  G-003 holds still, and is incomplete as stated (:192 inherits the crash); try/except Exception at the handler ->
  masks every class including future bugs, leaves a half-mutated body reaching estimate_tokens, the family
  DC-CR3-F-04-r1 rejected; a tolerant/strict parameter on normalize() -> new mechanism at a two-caller function for an
  outcome the narrowed call produces with fewer bytes; revert T019 step 1 -> preserves the verified CR3-F-02 defect,
  breaks the carried invariant, reddens an accepted witness, reopens an accepted increment; accept or route to the
  roadmap -> fails the latent fix-or-pin rule (nothing pins it) and is the candidate DC-CR3-F-01-r1 rejected.
Selected closure: count_tokens.py:87 calls the system-strip step (normalize.py:50-61) instead of all four steps;
  normalize.py changes no behavior — byte-exact, or with one delegating entry for the step if the implementation
  prefers a public name; session.py byte-exact; sid = session_id(body) stays where T019 put it.
Risk shape: unchanged — one call site; four production files byte-exact; T019's accepted witnesses keep their green;
  the only observable delta is a 500 becoming the 200 base and pre-range both gave.
Deferred additive work: owner roadmap.md §RM-012 (route-input posture pass, co-owned with §RM-001); trigger recorded
  there. Plus, for Interview: if the carried EI: handle is ever promoted, its statement should name the step the
  digest depends on rather than the whole function.
Owning route: implementation Forward — T019.002, nested under the owning task, own one-member wave.
Question required: no
```

**Why the pre-question gate admits nothing on the second half.** The dispatch asked whether the `/v1/messages` crash
is protected by parity, whether it should be absorbed here, or whether it belongs on the roadmap with its evidence
pinned, and named the tension honestly: this session has a pattern of declining to route out a 500 its own migration
introduced, and `T018` absorbed the `client_max_size` cap rather than defer it. **That pattern is followed, and it
points the other way here**, because the pattern keys on *who introduced the rejection*: `T018`'s 413 was a rejection
base never imposed and a documented configuration admits; this 500 is a rejection base imposes today, at the same line
of its own `normalize()`, measured. Applying base parity and `G-003` leaves one admissible outcome for this session —
do not change that route — so there is no unranked pair to ask about. Absorbing it would be the gate's "broadens scope
or cumulative risk beyond a smaller complete remediation" rejection and would breach an approved goal's constraint
column; deferring it *without* a named owner would orphan a real, reached-by-exposure behavior. So: parity protects,
not absorbed, not a question, **`roadmap.md` §RM-012** with this pass's evidence file pinned, an owner and a trigger,
one §Accepted risks line, one §Elevated to user line, and a fail-first witness inside `T019.002` (Done-when clause 5)
that goes red 500 → 502 if anyone hardens `normalize()` under cover of the repair.

### `CR4-F-02` — `accepted` by Reach First, no closure required

Inert file hygiene: nothing reads a file's final blank line, no gate on this tree can flag it, and the sibling shape
pre-exists and pre-dates the range. Under Reach First this is the inert-wording/unreachable class — disposition
`accepted`, one line under §Accepted risks with its citation, one backlog row for the shape, no task, no Fork, no
wave, and no claim row: `QB:tests/blank-line-at-eof`, owner **the next Forward pass that edits either named file**
(the touch-set-pickup treatment `DC-CR3-F-04-r1` gave in-range lint hits). It is not worth a wave, and the honest
answer to "is it worth a line of Forward work" is: worth a line **when that file is already open**, not worth a
dispatch of its own.

## Invariant cross-reference

- `CR4-F-01` — **Aligned**, binding to the existing handle `EI:buster_ripper.session/session_id/derived_after_normalize`
  (preserve-only, carried, never registry-active). The finding's required outcome and the carried authority point the
  same way; `HEAD`'s route is the site that over-satisfies it, and the narrowing preserves it by a different route —
  which is why this is not a regression-of-disposition, exactly as 088 itself stated. **No duplicate invariant
  candidate is created**; the ledger records the *wording* obligation for Interview instead. `G-003` is a goal, not an
  invariant, and is cited as ranking authority rather than classified.
- `CR4-F-02` — **Neutral**. No invariant of any kind speaks to a file's trailing byte; the nearest registry surface is
  the lint backlog row, which is debt, not authority.
- Triad: `git status --porcelain -- invariants` empty before and after. Zero registry bytes created, edited,
  superseded, deleted or promoted.

## Wave and durable transmission

**One wave, anchor `T019.002`, single member, owning route implementation Forward, dispatchable now; one local Yield.**
`remediation-waves.md` was read and applied: no existing wave can carry it — `T018`/`T019`/`T020` are accepted-local at
[083-01](083-01-yield.md) and `T019.001` at [085-01](085-01-yield.md), all four with fail-first evidence produced
against bytes this change would invalidate, and a ride-along joins a wave that is *scheduled*, of which there are none.
`CR4-F-02` opens no wave and joins none. Retry owner: `T019.002`'s own counter, starting at 0; `T019`'s counter stays
0 and its landed increment is not reopened.

**Ordering obligation, stated for the orchestrator:** `T019.002` lands **before** the MS-001 boundary's assumptions
recheck. That recheck is `S16`–`S19`'s single allowed post-repair recomposition; running it first would spend the
allowance on bytes this task is about to move. Then this wave's local Yield → the recheck → **Yield Phase**
(re-executing `QAP-MS001` on the final bytes — [087](087-forward-qa.md)'s `READY-PASSING` is byte-scoped to `8dfa5af`)
→ **Yield Red**, which owns `resolved` for `T015`–`T020`, `T019.001` and now `T019.002`. Nothing is `resolved` by this
pass.

Transmission: `handoff-context.md` §Planned Task Context → `T019.002` carries the finding id, evidence path,
counterexample, required behavior, the seven Done-when clauses with each witness's fail-first direction, the Yield
replay including both mutation runs, the capacity answer, the invariant classification, the closure-ledger key and the
execution order; `plan-summary.md` carries the task row and its pointer block; the ledger row is
`DC-CR4-F-01-r1` at repository state `git-commit:c5e2be44ba1a7ab984f02104472f06f849e4f773`. Forward receives session id
plus task id and reads the rest from these artifacts.

## Capacity — the dispatch asked for it stated plainly

**The fix lands without touching `messages.py` or `chat.py`.** `messages.py` is at 334 raw lines, in the 301-500 review
band, over its ceiling this session with the split itself owned by §RM-001 as a design decision
(`QB:routes-messages/file-size-yellow`, whose misplacement check [085-01](085-01-yield.md) already ran and declined);
`chat.py` is frozen at 300/300. Neither is in the touch-set: the change is one call at `count_tokens.py:87` (106 lines)
plus, if the implementation names it, a three-line public entry in `normalize.py` (228 lines) — and both `normalize.py`
and `session.py` are byte-unchanged across the whole range (`git diff --stat 6508941..HEAD` reports only
`chat.py` ±2, `count_tokens.py` +12/−2, `messages.py` +57/−14, `server.py`), which is why clause 6 requires AST-level
byte-exactness on all four. Test-side capacity is named too, because the honest answer is not "no production file, so
no capacity problem": `tests/test_routes_integration.py` sits at **495** raw lines, so a ~30-line addition there
crosses review-file (301-500) into extract-file (501-700). The witnesses belong in `tests/test_transport_parity.py`
(220, whose subject is already what base did and the package must keep doing) or in a new focused module; the choice is
Forward's, crossing a band silently is not.

## Context budget

`plan-summary.md` was at its 400-line cap entering this pass and is **at 400 leaving it**: the consumed `T019.001`
pointer block moved verbatim to `plan-completed.md` §Trimmed 2026-10-07 (`089-assumption-decide`) and the two
surviving italic notes in the table's footer were merged. `handoff-context.md` was already over its cap by the standing
disclosure at [075-01-yield](075-01-yield.md) and Checkpoints 122/168; this pass moved 83 lines out verbatim — the
superseded run-45 §Assumption routing narrative → `handoff-archive.md` §Trimmed 2026-10-07 (`089-assumption-decide`),
with a pointer left in place naming where each durable consequence now lives — and added the protocol-required ledger
row, blueprint, routing section, two §Accepted risks lines, one §Elevated to user line and the status/next updates. The
deep trim remains `phase-boundary-finalize`'s.

## Receipt (what this pass wrote; repository state at routing time `git-commit:c5e2be44ba1a7ab984f02104472f06f849e4f773`)

| File | What moved |
|---|---|
| `evidence/cr4-probe-matrix.txt` | **new** — the re-measurement: three trees, two axes, nine sections, the digest battery, the masked-site probe and the blast-radius simulation |
| `evidence/cr4-probe-harness/` | **new** — the three probe scripts, preserved so the measurement is re-runnable without this pass's context (`probe_pkg.py`, `probe_base2.py`, `probe_closure.py`) |
| `iterations/088-codex-review.md` | §Dispositions appended to the packet itself (its home, as 079's was for CR3): dispositions, the two measurement corrections, the `/v1/messages` ruling, the capacity answer |
| `iterations/089-assumption-decide.md` | **new** — this record |
| `handoff-context.md` | §Current Status (one CR4 bullet + the ordering consequence + `Next`), §Forward->Yield Retries (new wave, counter 0), §Planned Task Context → `T019.002`, new §Deterministic Closure Ledger — 2026-10-07 (CR4) with `DC-CR4-F-01-r1`, new §Closure-packet routing — 2026-10-07 (CR4) with the ruling, two §Accepted risks lines, one §Elevated to user line; the run-45 narrative moved out verbatim |
| `plan-summary.md` | `T019.002` row (nested immediately after `T019.001`) + its pointer block; the consumed `T019.001` pointer block moved out; footer notes merged — 400 lines in, 400 out |
| `plan-completed.md` | the `T019.001` pointer block, verbatim |
| `handoff-archive.md` | the run-45 assumption-routing narrative, verbatim |
| `roadmap.md` | **§RM-012** — the `/v1/messages` malformed-input posture, owner and trigger named, evidence pinned, and the completeness obligation (three loops, not two) |
| `quality-backlog.md` | `QB:tests/blank-line-at-eof`, one row, with its touch-set owner |
| `assumptions/claims.md`, `assumptions/seams.md`, `invariants/`, `decisions.md`, all production and test bytes | **untouched** — `sha256(seams.md) = e915654cdc3fa9ea1c8bd299e7958e67989281ea4cc7213e7eee78d4f33e45e0`, `sha256(claims.md) = 936d912449fbc16ac13bebe20a23a28163c06db19492cff82f0e3d0df5bea457` at this pass's exit, i.e. this pass moved neither; `git status --porcelain -- invariants` empty before and after |

Every absorbed finding has exactly one ledger row, one task-step landing and one `Planned Task Context` block; every
accepted finding has exactly one §Accepted risks line with its Reach verdict and citation; nothing was marked
`resolved`. Zero `question` dispositions means zero Fork reservations — `FK-011` stands alone.

```text
CLOSURE-ROUTED: CR4-F-01 | absorbed | T019.002 step 1 (nested under T019; one-member wave; DC-CR4-F-01-r1)
CLOSURE-ROUTED: CR4-F-02 | accepted | QB:tests/blank-line-at-eof (next Forward touch of either file)
```
