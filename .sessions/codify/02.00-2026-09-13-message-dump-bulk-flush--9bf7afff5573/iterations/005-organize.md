# 005 — Organize (REPLAN-INCREMENT: flush-policy cluster)

## Orchestrator Prompt (verbatim)

> Execute the organize-interactive skill for this session. Arguments: $ARGUMENTS
>
> Session ID: 02.00-2026-09-13-message-dump-bulk-flush--9bf7afff5573
> Mode: replan
> Cluster: flush-policy cluster (flush trigger parameters and timer mechanism)
> Reason (from Interview): REPLAN-INCREMENT — the trigger dial expanded beyond what
> decide analysed, so this cluster is not agent-ready and final approval is blocked.
> The user chose to flush on BOTH a time interval and a data-volume threshold, plus
> the shutdown flush — a third position the presented option set did not contain.
> Scope: author successor FKs for the trigger parameters (interval value and basis;
> volume-threshold basis — turns or bytes; whether the timer runs always or only
> with pending data); the timer mechanism plus its complete RB-* Reuse Evidence row
> (it is a Mode: NEW Anchor Map row); re-derive VP-T002-write-count-reduced's
> target in terms of F; re-check the ≤ 4 uncovered new statements coverage
> headroom, which tightens materially with a background task. Settled topics stay
> untouched: T001, T002, the existing Anchor Map bindings, the milestone skeleton.
> FK-002 (payload layout) is independent of the trigger parameters and is not in
> this cluster's scope.
>
> [Remainder of the prompt was the `organize-interactive` skill body plus the
> `codify-agents`, `critical-thinker`, and `codify-user-prompt` skill bodies.
> `$ARGUMENTS` was empty. Question Mode is `deferred` per session metadata; no
> active-mode phrase was present.]

## Session Type / Mode

- New parent session (`02.00`), no `parent_session_uid` → the parent coordination
  contract does not apply. No parent path to read or reconcile.
- **Input is the interview report** (`iterations/004-interview.md`) plus the prior
  collect/organize/decide reports. Per the interview-as-collect-input rule, the
  interview findings are the primary source; they override the stale assumption they
  correct (decide's "no background flush timer is needed").
- Scope is the **flush-policy cluster only**. `T001`, `T002`, the existing Anchor Map
  bindings, and the milestone skeleton are treated as settled and are not re-litigated.

## Context Conflict Detected

None. `decisions.md` (`FK-001` resolved, `FK-002` open), `plan-summary.md`,
`handoff-context.md`, and `iterations/004-interview.md` agree with each other and
with source. The one recorded supersession (decide's "no background flush timer is
needed" superseded by user authority, not falsified) is carried as written in
`FK-001`'s planning note; this pass does not re-derive the exclusion.

## Changes Since Iteration 002

- NEW: `RB-002` authored — the timer entity is `Mode: NEW` and now has a complete
  Reuse Evidence row. The nearest existing peer is the in-repo KV poller; the native
  aiohttp facility (`Application.cleanup_ctx`) was verified in installed source and
  is deliberately not adopted.
- NEW: two Anchor Map rows added (`flush timer`, `flush trigger parameters`); the
  `bulk flush`, `per-turn dump state`, and `shutdown flush seam` rows have their
  Notes updated for the third trigger. No existing binding changed.
- RE-DERIVED: `VP-T002-write-count-reduced`'s target. The general identity is
  `N + F`, not `N + 1`; `N + 1` is only the `F = 1` case. Reduction is
  `N − F` operations, i.e. a fraction `½·(1 − F/N)` — so the interval the user
  chooses sets how much of `G-001` survives.
- RE-CHECKED: the coverage headroom. The budget is **not a fixed 4**; it is
  `U ≤ 3.2 + 0.2·K` uncovered new statements for `K` new statements. The
  binding structural fact is that `server.py` is coverage-omitted and `dump.py` is
  not — placement decides whether the timer is inside the gate.
- NEW: `G-004` (periodic flush / bounded staleness) and `G-005` (volume-triggered
  flush) as separate goals, because a single test cannot prove both and each has a
  distinct failure witness.
- NEW: the timer's raise hazard — an exception inside the timer loop kills the
  interval trigger silently while the process looks healthy. This is the concrete
  `G-004` failure witness and a new correctness requirement.
- RESOLVED-FROM-CODE (not a user question): the interval **basis** (wall-clock
  seconds, matching `KV_POLL_INTERVAL`), the timer idle policy (always-running loop),
  and the injectability mechanism (CLI flag, matching every other config global).
- CONFIRMED as a genuine user decision: the interval **value**, and the volume
  threshold's **basis** ("data volume" reads as bytes; decide's option text said
  turns — the user never confirmed either).

## Anchor Map Verification (re-verified against source)

Every existing row was re-read at its bound symbol. **All bindings still resolve; no
wrong-binding correction this round.** Detail:

| Term | Binding | Re-verified against | Result |
|---|---|---|---|
| dump entry point | `dump.py:dump_turn` | `dump.py:61-109` | CONFIRMED — still the single append point; sync; called only from `routes/messages.py:57` |
| per-session dump state | `dump.py:SessionState` | `dump.py:16-24` | CONFIRMED — `turn`/`prev_text`/`session_dir`; module-global `_sessions` at `:24` |
| dump write buffer | `dump.py:SessionState` (extend) | `dump.py:16-24` | CONFIRMED — unchanged stance |
| bulk flush | `dump.py` (new symbol) | `grep -rn "write_text\|\.open("` → three write sites, all `dump.py:82,95,105` | CONFIRMED NEW (RB-001) — now called from **three** triggers, not one |
| dump dir config | `config.py:DUMP_DIR` | `config.py:16` | CONFIRMED |
| shutdown flush seam | `server.py:on_cleanup` | `server.py:60-72` | CONFIRMED — `:60` cancel KV task, `:72` close session; registered `:79` |
| signal handling | aiohttp web_runner | `web_runner.py:283-284` → `web.py:511` → `web.py:425` → `server.py:60` | CONFIRMED — no custom handler |
| dump caller | `routes/messages.py:handle_messages` | `routes/messages.py:55-59` | CONFIRMED `reuse` |
| dump index | `dump.py` index.txt write | `dump.py:104-106` | CONFIRMED — one open+write+close per turn |

### Rows added this pass

| Term | Binding (file:symbol) | Mode | Invariants | Notes |
|---|---|---|---|---|
| flush timer | buster_ripper/dump.py (new symbol) | NEW | EI:.../sequential_diff_chain, EI:.../graceful_shutdown_hook | RB-002; the periodic loop body, following the in-repo split where the domain loop lives in the domain module (`compaction.poll_kv_cache`) and the app wiring lives in `server.py`. Coroutine; `while True: flush_all(); await asyncio.sleep(config.DUMP_FLUSH_INTERVAL)`. **Must catch per-iteration exceptions and continue** — a raise kills the trigger silently (see G-004 failure witness). Must contain no `await` around the flush itself |
| flush trigger parameters | buster_ripper/config.py:DUMP_FLUSH_INTERVAL, DUMP_FLUSH_TURNS (new globals) | NEW | — | RB-002; the injectability seam. `config.py` is the established home for every tunable (`KV_POLL_INTERVAL` at `:44` is the direct peer) and every global there has a CLI flag in `main.py`; the QA gate needs the interval settable at process start. Spawner wiring (new symbol in `server.py`) is covered by the `shutdown flush seam` row's `extend` stance, not a separate row |

`shutdown flush seam` Notes updated: the hook now cancels **two** tasks, then runs
the final flush, then closes the shared session — in that order.

## Reuse Evidence — RB-002

| Evidence | NEW anchor | Name search | Role/peer search | Native capability | Conclusion |
|---|---|---|---|---|---|
| RB-002 | flush timer + its trigger parameters (`dump.py` periodic coroutine; `config.py:DUMP_FLUSH_INTERVAL`, `DUMP_FLUSH_TURNS`) | `grep -rniE "timer\|schedule\|interval\|periodic\|tick\|deadline\|call_later\|create_task\|asyncio\.sleep"` over `buster_ripper/` + `tests/` (bounded scope). Matches and dispositions: `server.py:25-34 spawn_kv_cache_monitor` — a background-task spawner, wrong concern (spawns the KV poller); `compaction.py:28-57 poll_kv_cache` — `while True: … await asyncio.sleep(config.KV_POLL_INTERVAL)`, nearest structural match, wrong concern (HTTP `/metrics` poll + module-global EMA); `config.py:44 KV_POLL_INTERVAL: int = 10` — an interval constant for a different concern; `tests/test_compaction.py:198-335` — the test pattern for a periodic task (create_task → sleep → cancel), a pattern peer not a code peer; `routes/count_tokens.py:19` — the word "periodically" in prose only. `buster_ripper_legacy.py:164,208,227` is the pre-refactor monolith (1013 lines, tracked, referenced by nothing in `buster_ripper/`); it contains a copy of the same poller and is **excluded from scope** as dead legacy — a reader must not bind to it. No `call_later`, `TimerHandle`, or scheduler use anywhere in the package | No registries or Protocol/ABC contracts exist in this repository. Sibling implementations inspected: `server.py:44-72` (spawn at `on_startup:57`, cancel at `on_cleanup:63-69`) and `compaction.py:28-57`. Why not reuse or extend: `poll_kv_cache` polls an external HTTP endpoint and writes a module-global EMA under a blanket `except Exception: pass` (`compaction.py:55-56`); the flush timer must call a synchronous in-process function on an interval. Extending it would fuse two unrelated concerns into one loop and couple the dump flush to the KV monitor's error handling and lifetime. Call paths inspected: `routes/messages.py:55-59` → `dump.py:61`; `server.py:44-72` | Verified in installed source: aiohttp 3.13.3 exposes `Application.cleanup_ctx` (`aiohttp/web_app.py:172-174`, `:408-409`) — the native async-context-manager lifecycle for background tasks. **Not adopted**: the repository's established lifecycle is `on_startup`/`on_cleanup` with the task keyed in `app` state (`server.py:57`, `:63-69`), and the flush must run *inside* the existing `on_cleanup` before `app["session"].close()`; adopting `cleanup_ctx` would split teardown across two mechanisms. asyncio itself has no periodic scheduler — only `TimerHandle`/`loop.call_later` chaining (verified: `[n for n in dir(asyncio) if 'timer' in n.lower() or 'period' in n.lower() or 'schedul' in n.lower()]` → `['TimerHandle']`). No third-party scheduler installed: `pyproject.toml:5-8` declares only `aiohttp` and `click` | No existing symbol owns the periodic-flush role; a new coroutine is required. What is reusable is the **lifecycle pattern** (spawn → `app`-state key → cancel in `on_cleanup`), not any existing code. The parameter globals are new but follow `KV_POLL_INTERVAL`'s established shape in the same module |

## Coverage Headroom Re-check

Re-derived from the live `.coverage` (re-run this pass, not carried):

```
TOTAL   506 stmts   98 miss   81%     (gate: --cov-fail-under=80, pyproject.toml:19)
```

Covered = 408. With `K` new statements of which `U` are uncovered:

```
(408 + K − U) / (506 + K) ≥ 0.80   →   U ≤ 3.2 + 0.2·K
```

**The budget is not a fixed 4.** It is `3.2 + 0.2·K` uncovered new statements.
The previously recorded "≤ 4" is the `K ≈ 4` case only. For a background task plus
its parameters (`K ≈ 25-40`), the ceiling is `U ≤ 8-11` — *looser*, not tighter, if
the code is roughly covered.

The constraint that actually binds is **placement**, from `pyproject.toml:26-30`:

| Module | Coverage | Consequence for the timer |
|---|---|---|
| `server.py` | **omitted** (`:29`) | a coroutine placed here adds **0** statements to the denominator |
| `dump.py` | **not omitted** | a coroutine placed here adds `K` statements that must be covered |
| `config.py` | not omitted | new module-level globals execute at import → always covered, coverage-neutral |

The plan should place the timer loop body in `dump.py` (domain module, matching the
`compaction.py` precedent) and accept that it is coverage-visible — untested timer code
is exactly what a flaky gate hides, and the loop body is small (`while True` + one
`try/except` + one `sleep`) and testable by the existing pattern in
`tests/test_compaction.py:198-335` (create task → patched/short sleep → cancel).
A placing in `server.py` would keep it coverage-free but untested by the gate.

Two pre-existing facts that shape the arithmetic and were re-verified:
`dump.py` carries 54 of the 98 missed statements at 24% (fixed debt, not new), and
`routes/*`, `main.py`, `server.py` are omitted, so the `on_cleanup` wiring and the
spawner are coverage-free.

## Goal Shape Analysis

Revised/extended from iteration 002. `G-001`'s success criterion and `G-002`'s scope
are **rebuilt** (see the Dropped-Clause Sweep); `G-004`/`G-005` are new.

| Goal candidate | Source wording | Observation surface | Met when | Failure witness | Boundary / non-goals | Alternate readings | Authority |
|---|---|---|---|---|---|---|---|
| G-001 — bulk write | "do collection into memory and dump to the disk in bulk reducing number of writes" | file-write operations issued against `DUMP_DIR` for a session of N turns, and the on-disk state of `DUMP_DIR/<sid>/` | For a session of N intercepted turns flushed F times, total file-write operations `= N + F`, and `F < N` (strictly fewer than today's `2N`); reduction `= N − F` operations | **Witness A**: `dump_turn` still calls `Path.write_text` and re-opens `index.txt` on every turn (`F = N`, so `N + F = 2N`) — buffering changed only *when* writes happen, never *how many*. **Witness B**: buffering and flushing both work, but the timer fires so often that `F → N`, so the measured reduction is ~0 — the mechanism is correct and the goal is still absent | Layout unchanged; no new persistence format; `/v1/messages` responses unchanged | (a) "bulk" = batched writes preserving layout → reduction `½(1 − F/N)`, ≤ ~50%; (b) "bulk" = one aggregated file → ~100% but breaks the layout contract; (c) the real goal is removing blocking IO from the request path and "reducing writes" is the user's mechanism guess | **OPEN** on the interval value, which sets `F` → decide (`FK-002` carries reading (b)) |
| G-002 — flush on Ctrl-C / SIGTERM | "properly handle ctrl c seek term … do a flush of that dump on a disk before terminating the process" | `DUMP_DIR/<sid>/` contents after SIGINT/SIGTERM | After SIGINT or SIGTERM the process exits and every turn buffered before the signal is on disk, in order, in the documented layout | Process exits cleanly (no traceback, exit 0) but `DUMP_DIR/<sid>/` is missing the turns buffered before the signal — the hook ran and wrote nothing | SIGKILL / hard power loss out of scope; no custom signal handler | none material — this is the user's explicit instruction and the shutdown flush is unchanged by `FK-001` | code-grounded (`web_runner.py:283-284` → `web.py:511` → `web.py:425` → `server.py:60`) |
| G-003 — proxy unchanged | "make this repository … to do collection into memory" (implicit) | HTTP responses on `/v1/messages` + the existing suite | Existing suite green; a raising dump/flush still yields the upstream response | A request returns non-200, or its latency is dominated by dump/flush work, because the buffer moved IO onto the request path | No new endpoints; no normalization changes; `dump_turn` stays failure-isolated | none material | code-grounded (`routes/messages.py:55-59`) |
| **G-004 — periodic flush (bounded staleness)** | "We need to have both time-based and data volume-based flash frequencies" | `DUMP_DIR/<sid>/index.txt` (existence, contents, mtime) **while the process runs**, with no shutdown and no signal | With the interval configured to `I`, a turn appended at `t` appears on disk no later than `t + I + ε`, with the process still running | The timer task exists, is spawned, and runs, but `index.txt` never appears for a live session — e.g. the loop's first `await` is skipped, the flush is a no-op against the wrong state, or an exception kills the task on the first iteration and nothing logs it. The process is healthy and staleness is unbounded | No durability guarantee (`US-001`); no fsync; the interval is not a request-latency bound; no promise about *ordering* relative to the volume trigger beyond the append-time sequencing invariant | (a) "time-based" = a fixed wall-clock interval; (b) "time-based" = a per-session idle timer that only arms after data arrives. Both bound staleness; neither is settled by source | code-grounded on the *mechanism*; the **interval value is OPEN** → decide |
| **G-005 — volume-triggered flush** | "We need to have both time-based and data volume-based flash frequencies" | `DUMP_DIR/<sid>/` contents after appending turns with the timer effectively disabled | After enough pending turns/bytes to reach the configured volume threshold, the pending turns are on disk with no shutdown and no timer | The threshold constant and its comparison exist, but turns accumulate past it with nothing on disk until shutdown — the check reads the wrong field, or the comparison is off by one, or it is evaluated before the append rather than after | Threshold basis and value are policy; no durability guarantee; no memory *cap* is promised unless the user picks a byte basis | (a) "data volume" = number of pending turns (the unit the code already counts, `SessionState.turn`); (b) "data volume" = bytes of pending rendered text (a literal reading of the user's words, needs a new accumulator) | **OPEN** → decide |
| Constraint — durability is a non-goal | "we don't care about making this super robust and if for example shutdown will kill the in memory in flight messages it's okay" | n/a — normative constraint on design | The implementation ships no durability machinery (no fsync, no journal, no crash recovery, no write-ahead log) | The implementation adds fsync/journaling/recovery to satisfy a durability requirement the user explicitly waived | SIGKILL loss acceptable; do not build for it | none — resolved by `US-001` and by the `FK-001` resolution | code-grounded (`US-001`) |
| Constraint — write-reduction ceiling | `US-001` "reducing number of writes" + `README.md:146-162` layout contract | measured write operations per session | Reduction `= ½(1 − F/N)`; ~50% only as `F → 1` | The plan states a fixed reduction percentage that `F` can drive to zero | The ceiling is a property of the preserved layout; only `FK-002`'s collapse reading beats it | none | code-grounded |
| Constraint — coverage gate | plan constraint | `--cov-fail-under=80` | Uncovered new statements `≤ 3.2 + 0.2·K`; all new `dump.py` lines covered | New `dump.py` statements ship uncovered and the suite fails | Not a user question | none | code-grounded (`pyproject.toml:19,26-30`) |

### Goal-shape findings (for decide)

1. **The interval is the dial that sets `F`, and `F` sets whether `G-001` exists
   at all.** Reduction is `N − F` operations, i.e. `½·(1 − F/N)` as a fraction. At
   `F = 1` it is ~50%; at `F = N` it is **zero** — `2N` writes, identical to
   today. The user asked for both triggers; the interval therefore controls how much
   of their *other* stated goal survives. `FK-A` must present that trade, not a bare
   number.
2. **`G-004` and `G-005` need separate proofs.** A tiny interval proves the timer
   but masks a broken volume trigger; a huge interval proves the volume trigger but
   masks a dead timer. Each needs a scenario in which the *other* trigger cannot fire.
   `G-004` additionally needs the timer's raise hazard covered: a per-iteration
   exception that kills the loop leaves the process healthy with unbounded staleness.
3. **"data volume" is ambiguous in the user's own words.** `FK-001`'s option text
   said "pending-turn threshold (order of tens of turns)"; the user's words say "data
   volume-based". Bytes is the literal reading; turns is the unit the code already
   holds (`SessionState.turn`). Different flush frequency for large bodies — a real
   product difference, not a naming one.
4. **`G-001`'s criterion is currently inconsistent with the plan's own Options
   table.** `plan-summary.md` §Options already states `2N → N + F`, while the Goal
   Definition Matrix still says "under one flush … < 2N" and
   `VP-T002-write-count-reduced` still says `≤ N + 1`. This pass rebuilds both.

## Invariant Reconciliation Prep

**No invariant registry exists in this repository** (`.invariants/` absent; re-confirmed
this pass). Collect's six `EI:` entries remain session-local drafts. This pass **narrows
to the contradiction surface** for the flush-policy cluster; the other rows are unchanged
from iteration 002 and are not re-listed here.

### Existing invariants in contradiction scope

- `EI:buster_ripper.dump/dump_turn/sequential_diff_chain` — turn N diffs turn N-1's
  rendered text; `prev_text` advances only when a turn is written | Status: **intended**
- `EI:buster_ripper.server/on_cleanup/graceful_shutdown_hook` — `on_cleanup` runs on
  graceful exit including aiohttp's SIGINT/SIGTERM path | Status: **intended**
- `EI:buster_ripper.dump/dump_turn/output_layout` — session dir holds
  `turn_000.json`, `turn_{N:03d}.diff`, `index.txt` | Status: **intended**
  (contradicted only by `FK-002`'s collapse reading, still open)

### Likely planned invariant themes (revised for this cluster)

- **Buffered append preserves turn order** — unchanged; sequencing stays at append time.
  Now load-bearing across **three** callers of the flush, not one.
- **The flush contains no `await`** — the mechanism that makes three triggers safe on
  asyncio's single thread. Introducing an `await` inside the flush creates an
  interleaving window between a timer flush and a volume flush. **This is the
  concurrency invariant of the two-trigger design and must be explicitly protected.**
- **Flush preserves the output layout** — unchanged.
- **Flush cannot raise out of `on_cleanup`** — unchanged (it would replace
  `GracefulExit` at `web.py:511`).
- **Flush is idempotent per pending batch** — unchanged; now also the reason an idle
  timer tick is a no-op rather than a duplicate index line.
- **NEW: the timer loop survives a raising flush** — a per-iteration exception must be
  caught and logged so the trigger is not silently killed. Distinct from the `on_cleanup`
  hazard: there the cost is a traceback at exit, here it is unbounded staleness in a
  healthy-looking process.
- **NEW: the timer task is cancelled before the final flush in `on_cleanup`** — a leaked
  task makes the interval trigger a second concurrent flusher during teardown.

### Suspected contradiction pairs

- `EI:.../sequential_diff_chain` ↔ *buffered append theme* — unchanged from 002, and now
  sharper: a timer flush must not advance `turn`/`prev_text`, or the diff chain resets
  whenever the interval fires.
- `EI:.../graceful_shutdown_hook` ↔ *timer cancellation theme* — not a conflict, but the
  hook's body grows a second cancel branch before the flush; ordering is load-bearing
  (cancel → flush → close session).
- `EI:.../output_layout` ↔ `FK-002` collapse reading — unchanged; still decide's to resolve.
- `G-001` ↔ `G-004` — **a goal-level contradiction, not an invariant one**: a short
  interval satisfies `G-004` and drives `G-001` toward zero. This is the trade `FK-A` exists
  to arbitrate.

### Out-of-scope invariants intentionally dropped

- None. All six `EI:` entries still intersect the planned change.

## Dropped-Clause Sweep

Triggered — this pass **rebuilds** `G-001`'s success criterion and
`VP-T002-write-count-reduced`'s target (wholesale replacement of a definition). One hop:
the direct predecessors are the `G-001` row and the `VP-T002-write-count-reduced` row in
`plan-summary.md`, plus `FK-001`'s `Impact: Plan` clause list in `decisions.md`. No
invariant lineage is cited by either, so there is no second hop.

| Clause (predecessor) | Disposition | Why / where it now lives |
|---|---|---|
| G-001 outcome: "held in memory and written in bulk instead of once per request" | carried | `plan-summary.md` §Goal Definition Matrix, `G-001` unchanged |
| G-001 observation surface: "write operations against `DUMP_DIR`; on-disk state of `DUMP_DIR/<sid>/`" | carried | same row |
| G-001 met-when: "for a session of N intercepted requests **under one flush**, file-write operations `< 2N`" | **superseded** | rebuilt as `= N + F` with `F < N`; "under one flush" was the `F = 1` assumption and is wrong under `FK-001` |
| G-001 failure witness: "`dump_turn` still calls `Path.write_text` and re-opens `index.txt` every request" | carried + extended | witness A kept; witness B (`F → N`, reduction ~0) added |
| G-001 boundary: "layout unchanged; no new persistence format; responses unchanged" | carried | same row |
| G-001 alternate reading (a) "~50% reduction" | carried | same row; now stated as `½(1 − F/N)`, whose maximum is the ~50% ceiling |
| G-001 alternate reading (b) "collapse → ~100%" | carried | same row; lives in `FK-002`, not this cluster |
| G-001 alternate reading (c) "real goal is IO off the request path" | carried | same row |
| `VP-T002-write-count-reduced` observable: "write operations for N turns **under one flush** `≤ N + 1`, versus `2N`" | **superseded** | rebuilt as `= N + F`, versus `2N`; `N + 1` is the `F = 1` case only, and "under one flush" contradicts the two-trigger design |
| `VP-T002-write-count-reduced` observable after `T002` / validate at `MS-001` | carried | same row |
| `VP-T002-write-count-reduced` scope reason: "structural/instrumented proof — must be paired with the black-box VPs so a mock cannot mirror the implementation" | carried + strengthened | now also requires `F` to be **deterministic**, which requires the injectable parameters |
| `VP-T002-write-count-reduced` evidence: "instrumented write-call count + deferred-state observation" | carried + extended | add "a known, injected `F`" |
| `FK-001` Impact: Plan list (Options table, `G-001`, `T003`, `MS-002`, `VP-T002-write-count-reduced`, coverage constraint) | carried | this pass executes every item; `VP-T002` and the coverage constraint are re-derived here, the rest are decide's |
| `FK-001` Impact: "Registry: none — no invariant registry exists" | carried | re-confirmed this pass |
| `FK-001` Resolution normative choice: "both a time interval and a data-volume threshold, plus the shutdown flush" | carried | unchanged; this pass adds no requirement and removes none |

Nothing was found lost. All other clauses carried verbatim.

## QA Planning Context

### Regression Test Candidates (unchanged)

| Behavior ID | Description | Risk | Tier | Priority |
|-------------|-------------|------|------|----------|
| B-001 | Dump output layout + index format | High | integration (tmp `DUMP_DIR`) | P0 |
| B-002 | Diff chain correctness across turns | High | integration | P0 |
| B-003 | Dump failure does not break the proxied request | High | integration | P0 |
| B-004 | `on_cleanup` shutdown behavior | Medium | unit (`tests/test_server_hooks.py`) | P1 |
| B-005 | Proxy response passthrough unaffected | High | existing suite | P0 |
| B-006 | Session directory isolation by `session_id` | High | integration | P1 |

### New Feature Test Areas

| Feature | Expected Behaviors | Tier | Dependencies |
|---------|-------------------|------|--------------|
| In-memory pending-write buffer | Turns append to per-session pending state; nothing appears on disk below both triggers | unit (`tests/test_dump.py`, new) | T001 |
| Bulk flush | On flush, all pending turn files written in order and `index.txt` written once with one line per turn | unit | T001, T002 |
| Volume-triggered flush | Reaching the configured volume threshold flushes pending turns with no timer and no shutdown | unit | T001, T002 |
| Time-triggered flush | With the interval injected to a small value, a pending turn reaches disk with no shutdown | integration (running loop) + unit (loop body) | T001-T003 |
| Timer resilience | A raising flush inside the timer loop is logged and the loop continues; a cancelled timer stops cleanly | unit | T003 |
| Shutdown flush | SIGINT/SIGTERM → `on_cleanup` → cancel timer → flush → all buffered turns on disk | integration (subprocess + real signal) | T001-T003 |
| Flush failure isolation | A raising flush neither fails a request nor changes process exit | unit + integration | T001, T003 |

### Test Scope Recommendations

- **Supporting local proof**: append order; flush writes the exact layout; flush no-ops
  when `DUMP_DIR is None`; flush clears pending without resetting `turn`/`prev_text`;
  flush idempotence across a volume-then-shutdown sequence; a raising flush inside the
  timer loop is swallowed and the loop continues; write-operation count for a known `F`.
- **Phase/milestone**: `MS-001` (buffer + volume trigger + flush observable in the
  filesystem), `MS-002` (time trigger observable on a live process; shutdown flush
  observable across a real signal).
- **Final integration**: existing suite green plus a real subprocess SIGTERM/SIGINT test —
  no in-process mock can prove the signal path.

### Validation Point Candidates

| Source behavior | Validation point | Observable behavior | Candidate observable after | Candidate validate at | QA author pass | Dependencies | Scope reason |
|---|---|---|---|---|---|---|---|
| B-001 | VP-T002-layout-preserved | After a flush, `DUMP_DIR/<sid>/` holds `turn_000.json`, `turn_NNN.diff`, `index.txt` with one documented line per turn | T002 | MS-001 | QAP-MS001 | T001, T002 | needs buffer + flush both present |
| B-002 | VP-T002-diff-chain | After one batched flush of turns 0..2, each diff reflects the change from the previous turn, in arrival order | T002 | MS-001 | QAP-MS001 | T001, T002 | ordering is only observable across a multi-turn flush |
| B-003 | VP-T001-dump-failure-isolated | A raising dump/flush still yields the upstream response body | T001 | MS-001 | QAP-MS001 | T001 | observable as soon as the buffer path exists |
| B-004 | VP-T003-shutdown-flush | After SIGINT or SIGTERM, buffered turns are on disk and the process exits without a traceback | T003 | MS-002 | QAP-MS002 | T001-T003 | requires process-level signal delivery; no in-process mock can prove it |
| B-005 | VP-T003-passthrough-regression | Existing `tests/test_routes_integration.py` suite green | T003 | MS-003 | — (existing suite) | T001-T003 | regression gate |
| B-006 | VP-T001-session-isolation | Two distinct system prompts produce two `DUMP_DIR/<sid>/` dirs, each flushed independently | T001 | MS-001 | QAP-MS001 | T001 | |
| (new) buffering | VP-T001-writes-deferred | After N turns below **both** thresholds and with the timer effectively disabled, no turn file for those turns exists on disk | T001 | MS-001 | QAP-MS001 | T001 | the deferral is the black-box form of "buffering"; now must hold with the timer present |
| (new) write count | **VP-T002-write-count-reduced (re-derived)** | For N turns flushed F times, instrumented file-write operations **= N + F**, versus `2N` today; reduction `= N − F` | T002 | MS-001 | QAP-MS001 | T001, T002 | structural/instrumented proof; must be paired with the black-box VPs so a mock cannot mirror the implementation. **`F` must be deterministic** — the test injects the parameters and disables the timer, so this stays a `MS-001` observation |
| (new) volume trigger | **VP-T002-volume-flush** | With the interval injected large (timer cannot fire), appending turns/bytes to the configured volume threshold puts the pending turns on disk with no shutdown | T002 | MS-001 | QAP-MS001 | T001, T002 | needs the flush + the threshold check; basis-agnostic wording, so it survives the `FK-B` answer |
| (new) time trigger | **VP-T003-interval-flush** | With the interval injected small and the volume threshold set unreachable, a single appended turn appears in `DUMP_DIR/<sid>/index.txt` while the process keeps running | T003 | MS-002 | QAP-MS002 | T001-T003 | the timer only exists once wired; the volume trigger is disabled so only the timer can satisfy it |
| (new) timer resilience | **VP-T003-timer-resilience** | A flush that raises inside the timer loop is logged and the loop keeps flushing on later ticks; a cancelled timer stops without a traceback | T003 | MS-002 | QAP-MS002 | T003 | the `G-004` failure witness is a dead trigger in a healthy process; only a live loop can show it |

**Risk note (unchanged, now sharper).** `VP-T002-write-count-reduced` is a white-box
instrumented assertion and must stay paired with the black-box VPs. Its determinism now
depends on the trigger parameters being injectable — without injection, `F` is not fixed and
the assertion is not repeatable.

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
| (new) volume-triggered flush | mapped | VP-T002-volume-flush |
| (new) time-triggered flush | mapped | VP-T003-interval-flush |
| (new) timer resilience | mapped | VP-T003-timer-resilience |

### Candidate Milestones (skeleton preserved; completion criteria extended)

| Milestone | Constituent tasks | Complete when | Validation points | QA author pass |
|---|---|---|---|---|
| MS-001 | T001, T002 | A buffered session's turns reach disk only after a flush — a volume-triggered flush or a shutdown flush — in the documented layout and arrival order, and the write count is `N + F` with `F` controlled | VP-T001-*, VP-T002-* (incl. VP-T002-volume-flush, VP-T002-write-count-reduced) | QAP-MS001 |
| MS-002 | T003 | After a real SIGINT/SIGTERM, buffered turns are on disk and the process exits cleanly; **and** with the timer running, a pending turn reaches disk without shutdown | VP-T003-shutdown-flush, VP-T003-interval-flush, VP-T003-timer-resilience | QAP-MS002 |
| MS-003 | T004 | Existing suite green with the new `dump.py` lines covered and total coverage ≥ 80% | VP-T003-passthrough-regression | — (existing suite is the gate) |

### Notes for Interview Agent

- **Two successor questions are genuine user decisions** (`FK-A` interval value,
  `FK-B` volume-threshold basis) — decide authors them; interview presents. See
  `## Successor Question Scope` below.
- **Not user questions** (resolved from code this pass): the interval *basis* (seconds),
  the timer idle policy (always-running loop), the injectability mechanism (CLI flag),
  and the placement of the timer loop in `dump.py`.
- **QA setup (Topic 3) gains one item**: the harness must be able to inject both trigger
  parameters at process start, or `VP-T002-write-count-reduced`,
  `VP-T003-interval-flush`, and `VP-T002-volume-flush` cannot be deterministic.

## Successor Question Scope (for decide — decide authors the FKs)

Organize structures; decide authors. Below is the question space with the
code-answerable candidates resolved out.

| Candidate question | Verdict | Basis |
|---|---|---|
| **Interval value** — how long may a buffered turn stay unflushed? | **USER DECISION** (preference) | No source settles it; it trades against `G-001`, the user's other stated goal. Alternate readings produce different success classifications → `OPEN` |
| **Volume-threshold basis** — pending turns, bytes, or both? | **USER DECISION** (preference) | The user's words say "data volume" (bytes); `FK-001`'s option text said turns; the user never confirmed either. Materially different flush behavior for large bodies → `OPEN` |
| Interval **basis** — wall-clock seconds vs. turns | **RESOLVED FROM CODE** — seconds | The in-repo precedent is `config.KV_POLL_INTERVAL: int = 10  # seconds` (`config.py:44`). A turn-based interval *is* the volume threshold, not a time interval; the user asked for both |
| Volume-threshold **value** | **RESOLVED — forward default, not a question** | Cheap and reversible (one constant); it is the volume twin of the interval and must be presented with it, not as a separate fork |
| Timer **idle policy** — always-running vs. only-with-pending | **RESOLVED FROM CODE** — always-running loop | A flush with nothing pending must be a no-op either way, so no user-observable difference exists. Recommend the in-repo shape (`while True: flush_all(); await asyncio.sleep(interval)`, per `compaction.py:34-57`) over re-arm-on-append bookkeeping. Not an escalation-floor question |
| **Injectability mechanism** — CLI flag vs. env var vs. config global | **RESOLVED FROM CODE** — CLI flag setting a `config` global | Every tunable in this repo is a `config` global with a `main.py` click option (`main.py:12-61`, `:70-90`); `serve_buster_ripper.sh` maps env → flags. A flag is sufficient for the QA harness, which starts the process itself |
| Timer **placement** — `dump.py` vs `server.py` | **RESOLVED FROM CODE** — loop body in `dump.py`, spawner in `server.py` | Matches the in-repo split (`compaction.py` holds the loop, `server.py:25-34` the spawner). `server.py` is coverage-omitted (`pyproject.toml:29`), so the spawner is free; the loop body is coverage-visible and must be tested |
| Timer raise policy | **RESOLVED FROM CODE** — catch per iteration, log, continue | A killed trigger is unbounded staleness in a healthy process (`G-004` failure witness); there is no user-observable choice between "die silently" and "keep flushing" |

**The one interaction `FK-A` must carry.** Reduction is `½·(1 − F/N)`. Presenting
an interval without `F` hides that a short interval erases `G-001`. `FK-A`'s `If-wrong`
must name that the reduction can reach zero.

## Compaction Decision

- Prior same-phase files reviewed: `iterations/002-organize.md`.
- Superseded: **none** — `002-organize.md` covers the *initial* planning slice
  (Anchor Map verification, the two-dial option analysis input, the original `G-001`–`G-003`
  rows, the initial task identification). This run covers a **different slice**: the
  flush-policy cluster's trigger parameters, timer mechanism, and re-derived VP/coverage
  figures. Both files remain valid; the new run refines rows `002` authored rather than
  replacing the analysis that produced them.
- Not superseded: `002-organize.md` (different slice).
- Compaction performed: **no**.
- If no: slice-based reason — the two organize files cover different slices of the same
  planning cycle. Compacting `002` would delete the Goal Shape Analysis that produced
  `FK-001`/`FK-002` and the Anchor Map verification record, neither of which this run
  supersedes. The Dropped-Clause Sweep above carries the specific clauses this run
  rebuilds, which is the intended mechanism for that delta.

## Confusability Findings

**No new finding.** One near-miss was examined and rejected as not a finding under the
observed-confusion reliability rule:

- `buster_ripper_legacy.py:164,208,227` — the tracked 1013-line pre-refactor monolith
  contains a copy of the same `while True: await asyncio.sleep(KV_POLL_INTERVAL)` loop,
  so a name search for the timer pattern hits it. I identified it as legacy on first read
  and did not misread it, and it is referenced by nothing under `buster_ripper/`
  (grep: only two session-report mentions). It is self-disambiguating by filename
  (`_legacy`), so it does not meet the observed-confusion bar. **Recorded instead in
  `RB-002`'s name search as an explicitly excluded scope**, which is where a future
  reader's search will actually land. No backlog entry added.

The existing backlog finding (duplicate `main` entry points, `server.py:99` vs
`main.py:62`) is unchanged and remains triaged as not endangering this session — the
timer spawner sits in `on_startup`/`on_cleanup`, which both entry points reach, and the
flush must tolerate `DUMP_DIR is None` regardless.

## Deferred Questions

Deferred mode (`Question Mode: deferred`; no active-mode phrase in the orchestrator
prompt). No `AskUserQuestion` used. The two genuine user decisions are routed to decide
(the FK owner) to author; organize does not present them.

- **Flush interval value:**
  - Status: DEFERRED for /codify-review — routed to decide (`FK-A` owner)
  - Impact: sets `F`, hence both the staleness bound (`G-004`) and how much of the
    write reduction survives (`G-001`). At `F = N` the reduction is zero.
  - Agent analysis: **none — genuine preference.** No source settles it, and the two
    readings of "time-based" (fixed interval vs. per-session idle timer) bound
    staleness equally, so neither can be excluded by code.
- **Volume-threshold basis (turns vs. bytes vs. both):**
  - Status: DEFERRED for /codify-review — routed to decide (`FK-B` owner)
  - Impact: turns leaves buffer memory unbounded (a session of large bodies); bytes
    makes flush frequency depend on body size, so a large turn-0 body can flush
    immediately and erase the reduction. Different flush behavior, not a naming choice.
  - Agent analysis: **none — genuine preference.** The user's own words ("data volume")
    read as bytes; the unit the code already holds is turns (`SessionState.turn`). Bytes
    needs a new accumulator; "both" mirrors the user's own two-trigger answer.

## Questions Asked

No questions - deferred mode (`Question Mode: deferred` in session metadata; no
active-mode phrase in the orchestrator prompt). Organize does not present FKs; the two
genuine user decisions are structured above and routed to their owner.

## Validation Performed

- Re-read `plan-summary.md`, `handoff-context.md`, `decisions.md`,
  `iterations/002-organize.md`, `iterations/003-decide.md`,
  `iterations/004-interview.md` — the interview report is the primary input.
- Read `buster_ripper/dump.py:1-109` — confirmed the three write sites (`:82`,
  `:95`, `:104-106`), `SessionState` (`:16-20`), the module-global `_sessions`
  (`:24`), the bare `assert` (`:63`), and that `prev_text`/`turn` advance at `:108-109`.
- Read `buster_ripper/server.py:1-113` — `on_cleanup` (`:60-72`), registration
  (`:79`), the spawn/cancel lifecycle (`:25-34`, `:57`, `:63-69`).
- Read `buster_ripper/compaction.py:1-91` — the periodic loop shape
  (`:34-57`) and its blanket `except Exception: pass` (`:55-56`).
- Read `buster_ripper/config.py` — `KV_POLL_INTERVAL: int = 10  # seconds`
  (`:44`); `DUMP_DIR: Optional[Path] = None` (`:16`).
- Read `buster_ripper/main.py:1-131` — every config global is set from a click
  option (`:12-67`, `:70-90`); no `--dump-*` tuning flag exists yet.
- Read `pyproject.toml` — `--cov-fail-under=80` (`:19`); `routes/*`, `main.py`,
  `server.py` omitted (`:26-30`); deps are `aiohttp` + `click` only (`:5-8`).
- Read `tests/test_server_hooks.py:1-152` and `tests/test_compaction.py:198-335` —
  the existing periodic-task test pattern (create_task → short sleep → cancel).
- Read `README.md:134-162` — the documented layout and the "Reading the index"
  workflow.
- `grep -rniE "timer|schedule|interval|periodic|tick|deadline|call_later|create_task|asyncio\.sleep"` over `buster_ripper/` + `tests/` — the RB-002 match set.
- `grep -rn "KV_POLL_INTERVAL"` and `grep -rn "buster_ripper_legacy"` — established the
  legacy file is tracked, 1013 lines, and referenced by nothing under `buster_ripper/`.
- `.venv/bin/python -m coverage report` — re-ran: TOTAL 506 stmts / 98 miss / 81%.
- `.venv/bin/python -c "import aiohttp; …"` + `inspect.getsource` on
  `aiohttp.web_app.Application` — verified `cleanup_ctx` in installed 3.13.3
  (`web_app.py:172-174`, `:408-409`), and `dir(asyncio)` periodic helpers →
  `['TimerHandle']` only.
- `grep -rn "dump_turn"` over `buster_ripper/` + `tests/` — one definition
  (`dump.py:61`), one import and one call site (`routes/messages.py:12,57`).

## Initial Task Identification

`T001` and `T002` are settled and untouched. `T003` is **enlarged** (the milestone
skeleton is preserved — `T003` stays the sole constituent of `MS-002`):

| Task | Description | Depends on | Status |
|---|---|---|---|
| T001 | Per-session pending-write buffer on `SessionState`; appended in `dump_turn`; `turn`/`prev_text` still advance at append time | — | settled |
| T002 | Bulk flush in `dump.py`: write pending turn files in order, then `index.txt` once; clear pending; no-op when `DUMP_DIR is None`; must not raise. **Now called from three triggers.** | T001 | settled |
| T003 | Wire the flush into `server.on_cleanup` wrapped so it cannot raise; **enlarged**: add the volume-threshold check in the append path, the periodic timer coroutine (loop body in `dump.py`, spawner + `app`-state key + cancel branch in `server.py`), the `config`/CLI injectable parameters, and the timer's per-iteration raise guard | T002 | **enlarged — decide finalizes** |
| T004 | Tests: `tests/test_dump.py` (buffer, flush, layout, diff chain, session isolation, failure isolation, volume trigger, timer loop + resilience) plus `on_cleanup` and signal cases; keep all new `dump.py` lines covered | T001-T003 | enlarged |

**Verdict for the REPLAN-INCREMENT:** the flush-policy cluster is now structured to
agent-ready depth. What remains is decide's authorship of `FK-A`/`FK-B` (the two genuine
user decisions) and the PLAN-SELECTED task-out that carries the re-derived
`VP-T002-write-count-reduced`, the new VPs, and the enlarged `T003`. Final approval
stays blocked until those land.
