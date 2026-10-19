# 002 — Organize

## Orchestrator Prompt (verbatim)

> Session ID: 02.00-2026-09-13-message-dump-bulk-flush--9bf7afff5573
>
> [Remainder of the prompt was the `organize-interactive` skill body (plus the
> `codify-agents`, `critical-thinker`, and `codify-user-prompt` skill bodies) with
> an empty `$ARGUMENTS`. No task-specific WHAT beyond the Session ID. Question
> Mode is `deferred` per session metadata; no active-mode phrase was present.]

## Session Type

New parent session (`02.00`, no `parent_session_uid` in `session-identity.json`).
The parent coordination contract does not apply. No interview reroute — organize
runs on the collect iteration report as its input.

## Changes Since Iteration 001

- VERIFIED: every collect Anchor Map binding against source. All bindings resolve
  to the referent collect named; no wrong-binding correction.
- CORRECTED: two collect details — a Mode refinement (`dump caller` extend →
  reuse) and a citation error (aiohttp graceful-exit catch is `web.py:511`, not
  `web.py:520`).
- RESOLVED: collect's open candidate `dump write buffer → NEW` — the buffer is a
  field on the existing `SessionState`, so the stance is `extend`, not `NEW`.
- NEW: quantified the coverage-gate headroom (**~4 uncovered statements**) and the
  write-reduction ceiling (**~50%**, index-only) — both constrain the plan.
- NEW: identified the loss-window scope mismatch between the user's stated
  acceptance ("in-flight messages") and shutdown-only flushing ("the whole session").
- NEW: confusability finding for the duplicate `main` entry points.

## Anchor Map Verification (re-verified against source)

Every row was re-read at its bound symbol. Detail:

| Term | Collect binding | Verified against | Result |
|---|---|---|---|
| dump entry point | `dump.py:dump_turn` | `dump.py:61-109` | CONFIRMED — the single write site; sync; called only from `routes/messages.py:57` |
| per-session dump state | `dump.py:SessionState` | `dump.py:16-20` | CONFIRMED — dataclass `turn`/`prev_text`/`session_dir`, stored in module-global `_sessions` (`dump.py:24`) |
| dump write buffer | candidates: `dump.py:SessionState`, NEW | `dump.py:16-24`; broad grep | RESOLVED → `extend`. The buffer is per-session pending state; `SessionState` is the per-session state holder. A separate buffer type would be an unnecessary second abstraction over the same key (`sid`). |
| bulk flush | `dump.py` (new symbol) | grep for `flush|buffer|batch|pending|queue|drain|writer` across `buster_ripper/` + `tests/` | CONFIRMED NEW — no existing flush/writer role; only 3 write sites exist package-wide, all in `dump.py` (lines 82, 95, 105) |
| dump dir config | `config.py:DUMP_DIR` | `config.py:16` | CONFIRMED — `Optional[Path] = None`, set at `main.py:88-90` |
| shutdown flush seam | `server.py:on_cleanup` | `server.py:60-72` | CONFIRMED — `on_cleanup` is registered at `server.py:79`; runs on graceful exit |
| signal handling | aiohttp web_runner | `web_runner.py:283-284`, `web.py:467`, `web.py:425`, `web.py:511` | CONFIRMED — but see corrected citations below |
| dump caller | `routes/messages.py:handle_messages` | `routes/messages.py:55-59` | CONFIRMED binding; Mode REFINED to `reuse` — `dump_turn(body) -> None` keeps its signature, so the call site needs no edit |
| dump index | `dump.py` index.txt write | `dump.py:104-106` | CONFIRMED — `index.open("a")` + one write + close, per turn |

### Corrections (headline)

1. **Citation error — aiohttp graceful-exit catch is `web.py:511`, not `web.py:520`.**
   `grep -n "GracefulExit" aiohttp/web.py` → `133`, `262`, `511`. Line 511 is
   `except (GracefulExit, KeyboardInterrupt):  # pragma: no cover`. Collect's
   `web.py:520` points at unrelated shutdown code. The *mechanism* collect described
   is correct; only the citation was wrong. Not a confusability finding — the
   disagreeing artifacts are a report and a vendored dependency, not two codebase
   artifacts.

2. **Mode refinement — `dump caller` is `reuse`, not `extend`.** No plan requirement
   forces a change at the call site: buffering and flushing both happen inside
   `dump.py`, and `dump_turn(body) -> None` keeps its signature and its
   swallow-all-exceptions contract (`routes/messages.py:55-59`). Marking it `extend`
   would invite an unnecessary edit to a file that coverage omits
   (`pyproject.toml:27`), hiding the real change surface.

3. **Resolved candidate — `dump write buffer` is `extend`, not `NEW`.** The pending
   writes are per-session state keyed by `sid`; `SessionState` is already that key's
   holder. A new buffer type would duplicate the keying already done by `_sessions`.

### New verified facts (not in collect)

- **`server.py` has a second `main()`** (`server.py:99-102`) that also calls
  `web.run_app`, in addition to the real console entry point
  (`pyproject.toml:11` → `buster_ripper.main:main`, defined at `main.py:62`). Only
  `main.py:main` configures `config.DUMP_DIR`. See confusability finding below.
- **Write operations are countable and few.** Package-wide there are exactly three write
  sites, all in `dump.py`: `Path.write_text` for the turn file (`dump.py:82`, `:95`)
  and `index.open("a")` for the index (`dump.py:105`). Per intercepted request that
  is **2 write operations** (open+write+close each), plus a one-time `mkdir` per new
  session (`dump.py:72`).
- **The flush hook must not raise.** `runner.cleanup()` runs inside the `finally` of
  `_run_app` (`web.py:425`), and `GracefulExit` is raised out of `main_task` and
  caught at `web.py:511`. If `on_cleanup` raises, it replaces `GracefulExit`, the
  `except (GracefulExit, KeyboardInterrupt)` clause no longer matches, and the process
  exits with a traceback. A flush that can raise on a full disk is a defect, not a
  nuisance.
- **Both `on_shutdown` and `on_cleanup` run within `runner.cleanup()`**
  (`web_runner.py:295-320` → `self.shutdown()` → `AppRunner.shutdown()` →
  `app.shutdown()` at `web_runner.py:386-387`; then `_cleanup_server()` →
  `app.cleanup()`). No custom signal handler is needed and none should be added —
  installing one would race aiohttp's own `add_signal_handler`.
- **Coverage budget is ~4 statements.** `.coverage` reports TOTAL 506 stmts / 98 miss /
  81%; the gate is `--cov-fail-under=80` (`pyproject.toml:19`). Solving
  `408 / (506 + K) ≥ 0.80` gives **K ≤ 4** uncovered new statements before the
  suite fails. `dump.py` alone carries 54 of the 98 missed statements at 24%.
  Practical consequence: the buffer/flush work must ship with tests that cover its new
  `dump.py` lines, and lifting any of the existing 54 missed statements buys headroom.
  `routes/*`, `main.py`, `server.py` are omitted (`pyproject.toml:26-30`), so the
  `on_cleanup` flush call itself is coverage-free — but the flush implementation in
  `dump.py` is not.

## Goal Shape Analysis

| Goal candidate | Source wording | Observation surface | Met when | Failure witness | Boundary / non-goals | Alternate readings | Authority |
|---|---|---|---|---|---|---|---|
| G-001 — bulk write | "do collection into memory and dump to the disk in bulk reducing number of writes" | write operations issued against `DUMP_DIR` per intercepted request, plus the on-disk state of `DUMP_DIR/<sid>/` during and after the session | For a session of N intercepted requests under one flush, the number of file-write operations is strictly less than 2N (today's count) | `dump_turn` still calls `Path.write_text` for the turn file and re-opens `index.txt` in append mode on **every** intercepted request — i.e. the buffer changed only *when* writes happen, never *how many* | Layout under `DUMP_DIR/<sid>/` unchanged (`README.md:146-162`); no new persistence format; `/v1/messages` responses unchanged | (a) "bulk" = batched writes preserving layout → **~50%** reduction (only the index appends collapse; turn files are one-file-per-turn and irreducible); (b) "bulk" = one aggregated file per session → **~100%** reduction but breaks the documented layout contract; (c) the real goal is moving synchronous IO off the request path and "reducing writes" is the user's *mechanism guess*, not the outcome | **OPEN** → decide |
| G-002 — flush on Ctrl-C / SIGTERM | "properly handle ctrl c seek term ... do a flush of that dump on a disk before terminating the process" | `DUMP_DIR/<sid>/` contents after the process receives SIGINT or SIGTERM | After SIGINT or SIGTERM the process exits and every turn buffered before the signal is present on disk, in order, in the documented layout | Process exits cleanly (no traceback, exit 0) but `DUMP_DIR/<sid>/` is missing the turns that were buffered — the flush hook ran and wrote nothing | SIGKILL / hard power loss explicitly out of scope; no custom signal handler (aiohttp already routes both signals) | (a) flush on SIGINT/SIGTERM only → the whole session's buffer is the loss window; (b) also flush on a size/turn threshold → the loss window is bounded | Partially code-grounded (the *seam*: `web_runner.py:283-284` → `web.py:511` → `web.py:425` → `server.py:60`); trigger policy **OPEN** → decide |
| G-003 — proxy unchanged | "make this repository ... to do collection into memory" (implicit: don't break the proxy) | HTTP responses on `/v1/messages` + the existing suite | Existing suite green; a raising dump/flush still yields the upstream response | A request returns non-200, or its latency is dominated by dump/flush work, because the buffer moved IO into the request path | No new endpoints; no normalization changes; `dump_turn` stays failure-isolated (`routes/messages.py:55-59`) | none material | code-grounded (`routes/messages.py:55-59`, `EI:routes.messages/dump_failure_isolated`) |
| Constraint — durability is a non-goal | "we don't care about making this super robust and if for example shutdown will kill the in memory in flight messages it's okay" | n/a — normative constraint on design | The implementation ships no durability machinery (no fsync, no journal, no crash-recovery, no write-ahead log) | The implementation adds fsync/journaling/recovery to satisfy a durability requirement the user explicitly waived | SIGKILL loss is acceptable; do not build for it | **"in flight messages" is ambiguous in magnitude** — (a) a bounded recent window (the current buffer contents); (b) **the entire session** (what shutdown-only flushing actually loses) | **OPEN** → decide |

### Goal-shape findings (for decide)

1. **The write-reduction ceiling is ~50%, not ~100%.** Turn files are one file per
   turn (`turn_NNN.diff`); batching cannot merge them without changing the
   documented layout. Only the per-turn `index.txt` append collapses (N appends → 1
   write per flush). So for N turns: 2N writes today vs N+1 after. If the user's
   intent was a dramatic reduction, only the layout-breaking collapse reading (G-001
   reading b) delivers it — and that reading is exactly the one the README contract
   forbids. **decide must not present "bulk buffering" as achieving the user's stated
   goal without naming this ceiling.**

2. **"in-flight messages" and shutdown-only flushing describe different loss
   magnitudes.** The user's acceptance sentence is scoped to in-flight messages. A
   shutdown-only flush makes the loss window *the entire session*, because nothing is
   ever written before exit. That is not a bigger version of the same thing; it is a
   different quantity, and the user's "it's okay" may not cover it. This is the single
   most important question for decide/interview.

3. **Shutdown-only flushing regresses the live-reading workflow.** `README.md:146-162`
   documents `index.txt` as the intended reading interface ("Reading the index"). Under
   shutdown-only, that file does not exist for the whole session. `dump.py`'s per-turn
   `log.info("dump: %s", summary)` (`dump.py:85`, `:102`) becomes the only live
   signal. Any option that keeps flushing shutdown-only must state this consequence.

## Invariant Reconciliation Prep

**No invariant registry exists in this repository.** `.invariants/` is absent at the repo
root and nowhere else in the tree (verified by directory search). Collect's six `EI:`
entries are therefore **session-local draft invariants authored by collect**, not registry
records. There is nothing to read broadly and nothing to reconcile against a registry.
They are carried forward as session-local, with the statuses below re-assessed.

### Existing invariants in contradiction scope

- `EI:buster_ripper.dump/dump_turn/output_layout` — session dir holds
  `turn_000.json`, `turn_{N:03d}.diff`, `index.txt` (one summary line per turn) |
  Status: **intended** — documented at `README.md:146-162`; code at `dump.py:80-106`
- `EI:buster_ripper.dump/dump_turn/turn_zero_is_full_body` — turn 0 is always a
  full JSON body, never a diff | Status: **intended** — `dump.py:80-85`; `README.md:148`
- `EI:buster_ripper.dump/dump_turn/sequential_diff_chain` — turn N diffs turn N-1's
  rendered text; `prev_text` advances only when a turn is written | Status:
  **intended** — `dump.py:86-108`
- `EI:buster_ripper.routes.messages/handle_messages/dump_failure_isolated` — any dump
  exception is logged and swallowed | Status: **intended** — `routes/messages.py:55-59`
- `EI:buster_ripper.session/session_id/derived_after_normalize` — `session_id` is the
  first 12 hex of SHA-256 of the normalized system prompt | Status: **intended** —
  `routes/messages.py:31-35`, `session.py:93-98`
- `EI:buster_ripper.server/on_cleanup/graceful_shutdown_hook` — `on_cleanup` runs on
  graceful exit including aiohttp's SIGINT/SIGTERM path | Status: **intended** —
  `server.py:60-72`; mechanism re-verified in aiohttp 3.13.3

**No `known_bug` or `likely_accidental` entries.** Nothing in scope is a bug masquerading
as load-bearing behavior. One adjacent item is a latent defect but not an invariant:
`dump.py:63` `assert config.DUMP_DIR is not None` is a bare `assert`, stripped under
`python -O`. It sits inside the file this session rewrites, so decide may fold a real guard
into the same increment — but it is not a behavior to preserve.

### Likely planned invariant themes

- **Buffered append preserves turn order** — the diff chain is computed in arrival order at
  append time, not at flush time; a reordered or concurrent flush corrupts every later diff.
- **Flush preserves the output layout** — same file names, same numbering, same index line
  format as today.
- **Flush is lossy-but-safe** — a flush that raises must not fail the request path, and must
  not escape `on_cleanup` (see the `GracefulExit` replacement hazard above).
- **Flush is idempotent per pending batch** — a threshold flush followed by a shutdown flush
  must not double-write or duplicate index lines.

### Suspected contradiction pairs

- `EI:.../sequential_diff_chain` ↔ *buffered append theme* — if the diff is computed at flush
  time instead of append time, `prev_text` sequencing becomes a cross-flush invariant that no
  single `SessionState` field enforces. Decide which point owns sequencing.
- `EI:.../output_layout` ↔ *G-001 reading (b) (collapse)* — direct contradiction; the
  collapse reading deletes the layout contract. decide must resolve G-001 before any task.
- `EI:.../graceful_shutdown_hook` ↔ *flush in `on_cleanup`* — not a conflict, but the
  hook's existing body (cancel task, close session) runs in the same coroutine; a flush that
  raises changes the process's exit behavior.

### Out-of-scope invariants intentionally dropped

- None. Collect recorded six and all six intersect the planned change (dump layout, diff chain,
  failure isolation, session keying, shutdown hook). Nothing was dropped.

## QA Planning Context

### Regression Test Candidates (from Collect)

| Behavior ID | Description | Risk | Tier | Priority |
|-------------|-------------|------|------|----------|
| B-001 | Dump output layout + index format (`turn_000.json`, `turn_NNN.diff`, `index.txt`) | High | integration (tmp `DUMP_DIR`) | P0 |
| B-002 | Diff chain correctness across turns (turn N vs N-1) | High | integration | P0 |
| B-003 | Dump failure does not break the proxied request | High | integration (aiohttp test client) | P0 |
| B-004 | `on_cleanup` shutdown behavior (cancel task, close session) | Medium | unit (`tests/test_server_hooks.py`) | P1 |
| B-005 | Proxy response passthrough unaffected | High | existing suite | P0 |
| B-006 | Session directory isolation by `session_id` | High | integration | P1 |

Tier names follow the existing project convention: this repo has no `qa-project`
skill, and its two established tiers are (1) pytest unit/integration under `tests/`
and (2) the `tests/test_routes_integration.py` aiohttp-client suite. Tier labels
above map to those.

### New Feature Test Areas

| Feature | Expected Behaviors | Tier | Dependencies |
|---|---|---|---|
| In-memory pending-write buffer | Intercepted requests append to per-session pending state; no turn file appears on disk below the flush threshold | unit (`tests/test_dump.py`, new) | T001 |
| Bulk flush | On flush, all pending turn files are written in order and `index.txt` is written once with one line per turn | unit | T001, T002 |
| Shutdown flush | SIGINT/SIGTERM → `on_cleanup` → flush → all buffered turns on disk | integration (subprocess + real signal) | T001-T003 |
| Flush failure isolation | A raising flush does not fail a request and does not change process exit | unit + integration | T001, T003 |

### Test Scope Recommendations

- **Supporting local proof** (Forward→Yield cycles): buffer append order, flush writes
  exact layout, flush no-ops when `DUMP_DIR is None`, flush clears pending without
  resetting `turn`/`prev_text`, flush idempotence across a threshold-then-shutdown
  sequence, write-operation count for N turns.
- **Phase/milestone**: `MS-001` (buffer + flush observable in the filesystem),
  `MS-002` (shutdown flush observable across a real signal).
- **Final integration**: the existing suite green, plus a real subprocess SIGTERM/SIGINT
  test — no in-process mock can prove the signal path.

### Validation Point Candidates

| Source behavior | Validation point | Observable behavior | Candidate observable after | Candidate validate at | QA author pass | Dependencies | Scope reason |
|---|---|---|---|---|---|---|---|
| B-001 | VP-T002-layout-preserved | After a flush, `DUMP_DIR/<sid>/` contains `turn_000.json`, `turn_NNN.diff`, and `index.txt` with one documented line per turn | T002 | MS-001 | QAP-MS001 | T001, T002 | needs buffer + flush both present |
| B-002 | VP-T002-diff-chain | After a single batched flush of turns 0..2, `turn_001.diff`/`turn_002.diff` each reflect the change from the previous turn, in arrival order | T002 | MS-001 | QAP-MS001 | T001, T002 | ordering is only observable across a multi-turn flush |
| B-003 | VP-T001-dump-failure-isolated | A raising dump/flush still yields the upstream response body | T001 | MS-001 | QAP-MS001 | T001 | observable as soon as the buffer path exists |
| B-004 | VP-T003-shutdown-flush | After SIGINT or SIGTERM, buffered turns are on disk and the process exits without a traceback | T003 | MS-002 | QAP-MS002 | T001-T003 | requires process-level signal delivery; no in-process mock can prove it |
| B-005 | VP-T003-passthrough-regression | Existing `tests/test_routes_integration.py` suite green | T003 | MS-003 | — (existing suite) | T001-T003 | regression gate |
| B-006 | VP-T001-session-isolation | Two distinct system prompts produce two `DUMP_DIR/<sid>/` directories, each flushed independently | T001 | MS-001 | QAP-MS001 | T001 | |
| (new) buffering | VP-T001-writes-deferred | After N intercepted requests below the flush threshold, no turn file for those requests exists on disk | T001 | MS-001 | QAP-MS001 | T001 | the deferral is the observable, black-box form of "buffering" |
| (new) write count | VP-T002-write-count-reduced | Write operations for N turns under one flush ≤ N + 1, versus 2N today | T002 | MS-001 | QAP-MS001 | T001, T002 | structural/instrumented proof — see risk note below |

**Risk note on VP-T002-write-count-reduced.** Counting writes requires
instrumenting `Path.write_text` / `Path.open`, which is a white-box assertion. It
must be paired with a black-box assertion (VP-T001-writes-deferred, VP-T002-layout-
preserved) so the "reduced writes" claim is not satisfied solely by a mock that
mirrors the implementation's own structure.

### Behavior Dispositions

| Behavior | Disposition | Validation point or reason |
|---|---|---|
| B-001 | mapped | VP-T002-layout-preserved |
| B-002 | mapped | VP-T002-diff-chain |
| B-003 | mapped | VP-T001-dump-failure-isolated |
| B-004 | mapped | VP-T003-shutdown-flush |
| B-005 | mapped | VP-T003-passthrough-regression (existing suite) |
| B-006 | mapped | VP-T001-session-isolation |
| (new) buffering deferral | mapped | VP-T001-writes-deferred |
| (new) write-count reduction | mapped | VP-T002-write-count-reduced |

### Candidate Milestones

| Milestone | Constituent tasks | Complete when | Validation points | QA author pass |
|---|---|---|---|---|
| MS-001 | T001, T002 | A buffered session's turns are on disk only after a flush, in the documented layout and order | VP-T001-*, VP-T002-* | QAP-MS001 |
| MS-002 | T003 | After a real SIGINT/SIGTERM, buffered turns are on disk and the process exits cleanly | VP-T003-shutdown-flush | QAP-MS002 |
| MS-003 | T004 | Existing suite green with the new `dump.py` lines covered and total coverage ≥ 80% | VP-T003-passthrough-regression | — (existing suite is the gate) |

### Notes for Interview Agent

- **Q1 (blocking for decide):** shutdown-only flush vs. threshold+shutdown
  flush. Frame it as the *loss-window magnitude*, not "robustness": the user
  waived losing **in-flight** messages; shutdown-only makes the loss **the whole
  session** and leaves `index.txt` absent for the entire session, regressing the
  `README.md:146-162` reading workflow.
- **Q2 (blocking for decide):** the ~50% write-reduction ceiling. If the user
  expects "in bulk" to mean near-zero per-turn writes, the only design that
  delivers it breaks the documented layout. Present the ceiling with the number.
- **Q3:** whether turn 0's full body (potentially hundreds of KB) is buffered too.
  Collect marked this genuinely open and organize confirms it: nothing in the code
  forces an answer.
- **Resolved, not for interview:** scope is the existing `--dump-dir` path on
  `/v1/messages` only (`routes/chat.py`, `count_tokens.py`, `passthrough.py`
  contain no dump call — verified by grep). Collect's deferred question 4 is
  code-grounded, not a user decision.
- **Resolved, not for interview:** "in bulk" does **not** require a background
  flush timer. A threshold check inside `dump_turn` is synchronous and needs no new
  task; reusing the `compaction.poll_kv_cache` task pattern
  (`server.py:25-34`) would add a task to cancel in `on_cleanup` for no benefit.

## Initial Task Identification

Candidate tasks only — decide owns final authoring, ordering, and done-when text.

| Candidate | Task | Why it exists |
|---|---|---|
| T001 | Add per-session pending-write buffer to `dump.py` — a pending field on `SessionState`, appended in `dump_turn`, with `turn`/`prev_text` still advancing at append time | The buffer is the core of G-001; sequencing must stay at append time to protect `EI:.../sequential_diff_chain` |
| T002 | Implement the bulk flush in `dump.py` — write all pending turn files in order, then `index.txt` once; clear pending; no-op when `DUMP_DIR is None` | The flush is the write-reduction mechanism (index appends collapse) and the unit the shutdown hook calls |
| T003 | Wire the flush into `server.on_cleanup`, wrapped so it cannot raise; add the threshold trigger if Q1 resolves that way | The shutdown seam for G-002; the raise-hazard is a correctness requirement, not polish |
| T004 | Tests: new `tests/test_dump.py` covering buffer/flush/layout/diff-chain/isolation/failure, plus an `on_cleanup` flush case; keep total coverage ≥ 80% | Hard gate — only ~4 uncovered statements of headroom exist |

## Compaction Decision

- Prior same-phase files reviewed: none — `001-collect.md` is the collect phase;
  this is the first `*-organize.md` in this planning cycle.
- Superseded: none
- Not superseded: `001-collect.md` (collect phase, different slice)
- Compaction performed: no
- If no: there is no prior same-phase file to compact. Nothing was deleted.

## Confusability Findings

One finding, filed to `quality-backlog.md`:

- `buster_ripper/server.py:99` vs `buster_ripper/main.py:62` — **same-vocabulary
  siblings**. Both define `def main(...)` that call `web.run_app`, with nothing at
  either definition site disambiguating them; only `main.py:main` is the console
  entry point (`pyproject.toml:11`), and only it configures `config.DUMP_DIR`.
  Trigger: resolved candidate ambiguous for a structural reason while verifying the
  signal path for this session's `G-002`. `server.py:main` also lacks every CLI
  option, so `python -m buster_ripper.server` silently runs with defaults
  (`UPSTREAM=http://localhost:30000`, `DUMP_DIR=None`). Remedy: delete
  `server.py:99-112` (the package's `main()` is `main.py:main`), or if it must
  stay, add a docstring naming it as a dev-only launcher. Does **not** directly
  endanger this session's scope (both entry points reach `create_app`/`on_cleanup`),
  so no pre-implementation cleanup task is recommended.

No other findings: no collect binding was a wrong-binding correction, and no other
artifact was misread this session. Speculative sweeps were not run.

## Deferred Questions

Deferred mode (session metadata `Question Mode: deferred`; no active-mode phrase in
the orchestrator prompt). No `AskUserQuestion` used.

- **Flush trigger policy — shutdown-only vs. threshold+shutdown:**
  - Status: DEFERRED for /codify-review — routed to decide (FK owner)
  - Impact: Determines the loss window magnitude (whole session vs. a bounded
    recent window) and whether `index.txt` is readable during the session at all.
    Also determines whether the "reduced writes" claim is measurable per session.
  - Agent analysis: ratification recommendation — **flush at shutdown AND on a bounded
    threshold**. The user waived losing **in-flight** messages; shutdown-only makes the
    loss **the entire session**, which is a different magnitude the user's words do not
    cover, and it regresses the documented live-reading workflow
    (`README.md:146-162`). Evidence: `dump.py` holds all state in a module-global
    `_sessions` with no write until flush.
- **Meaning of "in bulk" — preserve layout vs. collapse into one file:**
  - Status: DEFERRED for /codify-review — routed to decide (FK owner)
  - Impact: Collapsing breaks `EI:.../output_layout` and the diff-reading workflow;
    preserving caps the write reduction at ~50%.
  - Agent analysis: ratification recommendation — **preserve the documented layout**.
    The user asked to reduce *write operations*, not to change the output format, and
    `README.md:146-162` documents the layout as the reading interface. But decide
    must surface the ~50% ceiling, because that number is what the user's goal
    actually buys.
- **Whether turn 0's full body is buffered too:**
  - Status: DEFERRED for /codify-review — routed to decide (FK owner)
  - Impact: Turn 0 can be hundreds of KB; deferring it means a hard shutdown
    leaves a session directory with no baseline.
  - Agent analysis: **none — genuinely open.** Nothing in the code or the user's
    words decides it.
- **"in flight messages" magnitude — recent window vs. entire session:**
  - Status: DEFERRED for /codify-review — routed to decide (FK owner)
  - Impact: Sets the loss window the user already accepted; changes whether a
    periodic flush is required.
  - Agent analysis: ratification recommendation — the user's sentence is naturally read
    as a bounded recent window, so shutdown-only exceeds what was accepted.

## Questions Asked

No questions - deferred mode; no active-mode instruction was present in the
orchestrator prompt or session metadata. All clarifications are recorded above as
deferred questions with agent analysis and routed to their owning author.

## Validation Performed

- Read `dump.py`, `routes/messages.py`, `server.py`, `main.py`, `config.py`,
  `session.py`, `compaction.py`, `tests/test_server_hooks.py`, `pyproject.toml`,
  `README.md` (dump section).
- `grep -rn "dump"` across `buster_ripper/` + `tests/` — one dump entry, one call site.
- `grep -rniE "buffer|batch|pending|flush|queue|drain|accumulat|writer|defer|debounce"`
  — only `X-Accel-Buffering` and an unrelated comment. RB-001 confirmed.
- `grep -rn "signal|SIGTERM|SIGINT|atexit|KeyboardInterrupt|on_shutdown"` — nothing.
- `grep -rn "write_text|\.open("` — exactly three write sites, all in `dump.py`.
- Read aiohttp 3.13.3 `web_runner.py` (`setup` 281-284, `cleanup` 295-320,
  `AppRunner.shutdown` 386-387) and `web.py` (`run_app` `handle_signals=True` at
  467, `_run_app` finally → `runner.cleanup()` at 425, `GracefulExit` catch at 511)
  — the SIGINT/SIGTERM → `on_cleanup` chain is confirmed by source, not assumed.
- `python -m coverage report` on the existing `.coverage` → 506 stmts / 98 miss /
  81%; derived the ≤4-statement headroom.
- Confirmed `.invariants/` does not exist at the repo root or anywhere in the tree.
