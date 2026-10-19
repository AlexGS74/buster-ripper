# 031 — forward (T001: per-session pending-record buffer)

## Orchestrator Prompt (verbatim)

> Session ID: 02.00-2026-09-13-message-dump-bulk-flush--9bf7afff5573
> Task: T001

## Increment Brief

- **Task/goal**: T001 — add the per-session pending-record buffer to the dump path: `SessionState` gains
  `pending`/`pending_bytes`, `dump_turn` appends each turn's record unconditionally and accumulates rendered-text
  bytes; purely additive, nothing drained (no trigger exists until T003), nothing removed.
- **Anchors touched**: per-session dump state → `buster_ripper/dump.py:SessionState` → extend (the two fields the
  row names); dump entry point → `buster_ripper/dump.py:dump_turn` → extend (the append the row hosts, at append
  time); dump write buffer → `dump.py:SessionState` (fields on the existing class — explicitly *not* a new type,
  per the row and `E:T001/dump/session-state`).
- **Will NOT create**: no `flush_all`, no format selector (`DUMP_FORMAT`/`DUMP_FORMATS`/`--dump-format`), no trigger
  parameters, no artifact path builder, no record encoders — all T002/T003 authorized surfaces, untouched here.
- **Reuse evidence**: N/A — no `Mode: NEW` anchor is exercised by T001.
- **Inventions: none.** `tests/test_dump.py` is the task's own done-when ("unit coverage in `tests/test_dump.py`"),
  not an invention.

## Pre-flight verification

- **Plan approval gate**: PASSED. Round 18 final approval ("approve, begin", option 1) with the implementable mark;
  the chain completed after it — `forward-red` scoped recheck RED: 9 | 0 AMBIGUOUS (030), the R-A…R-D repairs
  applied (Checkpoint 58) and reconciled (Checkpoint 59), the assumption gate complete (Checkpoint 57: 50 claims,
  6 seams SAFE, routing absorbed). The `## Planned Task Context` blocks and the Deterministic Closure Ledger exist
  on disk — this dispatch is the first `forward`.
- **QA contract consistency**: `qa-plan.md` exists. T001's VP handles: `VP-T001-dump-failure-isolated` (observable
  after T001, validated at MS-001 via TC-001/QAP-MS001 — **authored by forward-qa, not by me**). No integration
  harness run during implementation; no persistent QA gate test authored.
- **Task-specific planned context**: no `<details>` block matches T001 (the five blocks route claim-3/18/19/22/32/38
  to T004/T003/T002/T005/T006) — no `Assumption finding` criteria attach to T001.
- **Invariant registry (narrowed to T001's touch paths)**: read `invariants/buster_ripper/dump/invariants.md`.
  `E:T001/dump/session-state`'s binding resolves and its gloss is exactly this increment (pending list + byte count
  + append-time advancement). The three dump `OI:` blocks are `Decision: preserve` — T001 contradicts none: the
  documented writes are untouched (`output_layout`), `turn`/`prev_text` still advance at append time
  (`sequential_diff_chain`), turn 0's full-body write is untouched (`turn_zero_is_full_body`). The `PI:T002/*`
  entries are `Status: planned` for T002's surfaces; T001 neither implements nor contradicts them. **No protected
  invariant conflict — no stop.**
- **Working tree**: clean on `refactor/package-structure` at `7c59495` (Checkpoint 59) before the first edit.

## Implementation

Two additive edits to `buster_ripper/dump.py` (109 → 115 lines, FL-LNCP Green) and one new test file.

1. **`SessionState` gains the buffer fields** (the anchor row's exact pair):
   ```python
   pending: list[str] = dataclasses.field(default_factory=list)
   pending_bytes: int = 0
   ```
   `default_factory` is required — a bare mutable default would share one list across every session state
   (failure-path row below). A two-line comment records the non-obvious semantics: `pending_bytes` measures
   **rendered-text bytes** (`len(_render_body(body))`, the FK-B volume-trigger measure), not record bytes.
2. **`dump_turn` appends at append time**, in the state-advancement block:
   ```python
   state.pending.append(current_text)
   state.pending_bytes += len(current_text)
   state.prev_text = current_text
   state.turn += 1
   ```
   The append is **unconditional** in this increment (F-21 as corrected by 025/R9: the condition is T002's);
   `turn`/`prev_text` advance exactly as today, now together with the buffer — satisfying
   `OI:.../sequential_diff_chain`'s append-time clause ("a flush never resets the chain" — vacuously true here,
   no flush exists).

**Reading decision — what "the turn's encoded record" is at T001.** The format-specific record builders
(change-focused / self-contained records, summary-line framing) are T002's, extracted by its capacity gate; none
exists yet. The plan's own T001 sentence ties the record to the same value as the byte measure ("appends the turn's
encoded record … and adds `len(current_text)` to `state.pending_bytes`"), so the record appended at this increment
is `current_text` — the turn's `_render_body` rendering, the only encoding that exists today. T002's dispatch will
replace the record body with the format records and add the collapsed-path condition (018/F-21); the seam — the
two statements at the advancement block — is what T002 conditions. Consequence verified in tests:
`pending_bytes == sum(len(r) for r in pending)` holds **only** in this increment (records are renderings); T002's
change-focused records will store diffs, at which point the two measures diverge by design (FK-B).

**T001's REMOVED clause honored**: nothing removed. The three write sites (`:85`, `:98`, `:108` after the edit) and
the `index.txt` append stay live; `dump_turn` still writes every turn to disk exactly as today while also buffering.
The bare `assert config.DUMP_DIR is not None` stays (F-5's real `ValueError` guard is T002's done-when). The module
docstring is untouched (T002 owns the collapsed-artifact docstring clause).

## Failure-Path Evidence

| Operation / condition | Expected outcome and authority | Handling / observation evidence | Disposition |
|---|---|---|---|
| `dump_turn` called with `config.DUMP_DIR is None` | Not applicable at T001 — the only caller guards `if config.DUMP_DIR is not None` (`routes/messages.py:55`); the assert→ValueError repair is T002's (`F-5`) | Caller guard verified unchanged; no new call sites added | not-applicable (guarded upstream; T002 owns the guard) |
| Dump path raises (incl. the new statements) | Request still succeeds — the caller's swallow-all isolates it (`VP-T001-dump-failure-isolated`, observable after T001; owner = caller, gate TC-001 at MS-001) | `routes/messages.py:55-59` `try/except Exception → log.exception`; the two new statements run after `_render_body` succeeded and cannot raise for any body that rendered (in-memory list append / int add) | safe (existing contract; no new raise site) |
| Mutable-default sharing across `SessionState` instances | Each session owns a private pending list | `dataclasses.field(default_factory=list)`; `TestSessionStateDefaults::test_new_state_has_empty_private_buffer` asserts distinct objects | safe |
| Cross-test leakage via module-global `_sessions` / `config.DUMP_DIR` | Tests must not observe prior state (parallel-safe) | `dump_env` fixture sets `config.DUMP_DIR`, clears `dump._sessions` before **and after** each test, restores `DUMP_DIR = None` | safe |
| Buffer drained / flushed | Not applicable — no trigger and no `flush_all` caller exist at T001 (`MS-001` declares no flush; triggers are T003's) | Buffer grows within the process by design; the plan's documented intermediate state | planned (T003) |
| Disk-full / write failure on the documented sites | Unchanged pre-existing surface — same three sites, same isolation via the caller | No write site added, removed, or moved | not-applicable (untouched surface) |

## Capacity gate

Additive production surface: `dump_turn` (+2 executable statements) and `SessionState` (+2 field declarations).
Pre-edit baseline measured this pass (`codify scan path buster_ripper/dump.py --detailed`): `dump_turn`
**L:42 N:1 C:7** (pre-existing hard-L violation + gray C — the plan's own recorded capacity debt), `_render_body`
L:26 N:2 C:9 (pre-existing gray, **not touched**). Post-edit: `dump_turn` **L:44 N:1 C:7** — N and C unchanged; L
grew by exactly the two plan-prescribed statements. The plan's approved capacity constraint (§Goals and
Constraints) schedules the extraction ("state lookup/creation, format dispatch, the documented writer, the two
encodings, summary line, artifact path builder") at **T002** — those concepts do not exist before T002's format
machinery, so the extraction cannot be pulled into T001 without authoring T002's surface early (Plan Adherence
bars that). Debt recorded here for Yield: the hard-L violation worsened 42→44 by the plan's own prescribed
statements; `dump.py` remains FL-Green at 115 lines.

## L→N→C→P compliance

- `SessionState` fields: declarations, not functions — N/A.
- `dump_turn`: pre-existing violation (see Capacity gate); no new metric class entered.
- New test file functions: all at ideal targets; `codify scan path tests/test_dump.py --git --detailed` →
  **"No refactoring issues found."** (the first draft's one gray-L test was refactored — shape literals extracted
  into `_block_shape_body()`/`_plain_shape_body()` — before checkpointing).

## Tests written and coverage

`tests/test_dump.py` (new, 198 lines, 16 tests — no dump tests existed before, per T004's "REMOVED: nothing — no
dump tests exist today"):

- `TestPendingBuffer` — one record per turn (parametrized N ∈ {1,2,3,5}); arrival order (records ==
  rendered texts in call order); `pending_bytes` accumulation (parametrized N ∈ {1,2,5}); bytes-track-records
  (this-increment property); accumulation across every `_render_body` shape (system-as-list, tools,
  content-as-list, plain strings — same system *content*, one session).
- `TestSessionIsolation` — two prompts → two states, independent buffers (per-sid keying,
  `E:T001/dump/session-state`); same prompt → one shared buffer.
- `TestStateAdvanceTiming` — after each call: `turn` advanced, `prev_text` == current rendering, buffer length
  advanced — the append-time co-advancement (`OI:.../sequential_diff_chain`, claim 8).
- `TestDocumentedWritesStayLive` — T001's additivity guard: after 3 turns, `turn_000.json` + 2 `.diff` +
  `index.txt` (3 lines, FULL first line) exist **while** the buffer holds 3 records — "REMOVED: nothing".
- `TestSessionStateDefaults` — fresh states own private empty buffers (mutable-default guard).

**Results**: `uv run pytest` (full suite, checkpoint run) — **143 passed**, total coverage **91.42%** (gate 80%);
`buster_ripper/dump.py` **100%** (was 54 missed statements at 24% per the plan's baseline — this file alone
clears them). New-statement coverage: 2/2 executed on every test path (both sit in `dump_turn`'s straight line).
Plan coverage arithmetic: K = 2 new statements, U = 0 uncovered → `U ≤ 3.2 + 0.2·K` trivially satisfied.

**Property-based note**: `hypothesis` is not in the dev dependency group (pyproject declares pytest,
pytest-aiohttp/asyncio/cov/mock only) and adding a dependency is outside T001's touch set — the parametrized
N-dimension plus the shape-variant test carry the property coverage (append count, order, and byte-sum
arithmetic) within the approved dependency set.

## Dependency claims checked

None encoded. T001 adds no cross-module call, no shape validation, no ordering/freshness assumption on foreign
code — `_render_body`/`session_id` are same-module/same-package reads already exercised today. `claims.md` rows
resting on T001 (3, 4, 8, 11, 12, 13, 17, 45) were read; all were already verified by the assumption gate
(Checkpoint 57) with closures DC-claim-3-r1 etc. — nothing T001 writes re-opens one. No silent corrections.

## Backlog pickup

`Backlog pickup: none match touch-set` — the three `quality-backlog.md` items target `server.py:99`,
`plan-summary.md`'s capacity paragraph, and `config.py:13`/`main.py:13`; T001's touch set is
`buster_ripper/dump.py` + `tests/test_dump.py`. (This report's Capacity-gate section records the per-function
metrics the backlog's item 2 asks planning surfaces to carry — recorded here, not edited there; plan text is not
Forward's to rewrite.)

## Durable signal

- **Overrides exercised**: none — T001 displaces nothing (the session's `refine` direction; all three dump `OI:`
  blocks preserved and their `preserve` decisions untouched).
- **Decision changes**: none. One reading decision documented above (the T001 record = the rendering) — an
  application of the plan's own sentence, not a change to it.
- **Bug-fix rationale**: none (no defect found).
- **Invariant entries changed by this implementation**: none (triad rule — Forward writes no invariant records;
  `E:T001/dump/session-state` already binds and needs no edit: its gloss already describes the landed fields).

## Concerns

- `dump_turn`'s hard-L violation grew 42→44 (plan-prescribed statements; extraction is T002's — see Capacity
  gate). Flagging for Yield's explicit adjudication rather than silently proceeding.
- No user-visible surface changed in this increment (on-disk output is byte-identical to today; the buffer is
  in-memory) — **no visual demo evidence row is owed**; the first evidence-bearing increment is T002 (format
  branch) / T003 (flush observable).
- No `TodoWrite` tool exists in this runtime — progress was tracked in this report instead.

## Questions Asked

None — no materially different admissible outcomes remained (the one reading fork, record content at T001, is
settled by the plan's own T001 sentence naming `len(current_text)` beside the append; T002 owns the divergence).

## Status for Yield

T001 implemented exactly as specified: fields + unconditional append + byte accumulation + unit coverage;
purely additive; full suite 143 passed at 91.42%; scan clean on the test file, `dump_turn` delta fully accounted.
Next planned increment: **T002** (format branch, collapsed artifact, `flush_all`, capacity extraction,
`norecursedirs` collection exclusion) — not clustered with T001 because T002 is a day-plus task on its own.
