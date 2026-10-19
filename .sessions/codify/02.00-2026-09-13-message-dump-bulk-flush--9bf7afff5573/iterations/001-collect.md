# 001 — Collect

## Orchestrator Prompt (verbatim)

> Session ID: 02.00-2026-09-13-message-dump-bulk-flush--9bf7afff5573
> Initial Goal: Change buster-ripper's message interception and dumping mode so captured messages are collected in memory and dumped to disk in bulk, reducing the number of write operations. Robustness/durability is explicitly not a concern for in-flight data — if a hard shutdown kills buffered in-memory messages, that is acceptable. Additionally, handle Ctrl-C / SIGTERM properly: flush the buffered dump to disk before terminating the process.
> User Context: The user invoked CODIFY explicitly for this change. Working directory is the buster-ripper repository (/Users/alexhubitski/Projects/buster-ripper), currently on branch refactor/package-structure with 31 unpushed local commits and some uncommitted .sessions/ files. The user stated: "make this repository when it runs in message interception and dumping mode to do collection into memory and dump to the disk in bulk reducing number of writes we don't care about making this super robust and if for example shutdown will kill the in memory in flight messages it's okay so at the same time please properly handle ctrl c seek term if i'm not mistaken and do a flush of that dump on a disk before terminating the process". "seek term" is the user's rendering of SIGTERM; confirm during collection what signal handling actually exists today.

## Session Type

New parent session (`02.00`). No `parent_session_uid` in `session-identity.json` — the parent coordination contract does not apply.

## Scope

Collect the current state of the message-interception/dumping path in
`buster-ripper` and the process shutdown path, in preparation for a plan that
(a) buffers captured messages in memory and writes them to disk in bulk, and
(b) flushes that buffer on Ctrl-C / SIGTERM before exit.

## What Was Collected

### 1. Dump path (the write surface)

- `buster_ripper/dump.py` (109 lines, `Mode` candidate: extend) — module-level
  `_sessions: dict[str, SessionState]`; `SessionState(turn, prev_text, session_dir)`.
- `dump_turn(body)` is the single entry point, called from exactly one place:
  `buster_ripper/routes/messages.py:57` inside `handle_messages`.
- `dump_turn` is **synchronous** and performs **direct, unbuffered disk IO per
  call**:
  - turn 0: `session_dir/turn_000.json` ← `write_text(json.dumps(body, indent=2))`
  - turn N: `session_dir/turn_{N:03d}.diff` ← `write_text(unified_diff(...))`
  - every turn: `session_dir/index.txt` opened in append mode, one line written, closed
  - i.e. **2 file writes per intercepted request** (turn file + index append),
    plus `mkdir(parents=True)` on first turn of a session.
- Diff chain is **sequential**: turn N diffs `state.prev_text` (rendered text of
  turn N-1) against the current render. `prev_text` is updated in place at the
  end of each call. Turn numbering is `state.turn` per in-memory session.
- `_render_body(body)` is a pure renderer (tools → system → messages, joined by
  an 80-char rule).
- Dump failures are isolated: `routes/messages.py:55-59` wraps `dump_turn` in
  `try/except Exception` and only logs — a dump error never fails the proxied
  request.
- Dump is gated on `config.DUMP_DIR is not None` (set from `--dump-dir` in
  `main.py:88-90`, `mkdir(parents=True, exist_ok=True)` at startup).
- `dump_turn` asserts `config.DUMP_DIR is not None` (bare `assert`, would be
  stripped under `-O`).

### 2. What is NOT dumped

- `/v1/chat/completions` (`routes/chat.py`) does not import or call `dump_turn`.
- `/v1/messages/count_tokens` and the passthrough catch-all do not either.
- So "message interception and dumping mode" today == the `--dump-dir` path on
  `/v1/messages` only.

### 3. Signal handling today (the user's "ctrl c seek term")

- **No signal handling exists in the repository.** Grep for
  `signal|SIGTERM|SIGINT|atexit|KeyboardInterrupt` across `buster_ripper/` and
  `tests/` returns nothing. The legacy single-file `buster_ripper_legacy.py`
  has none either.
- Signals *are* handled, but only implicitly, by the framework:
  `main.py:126` calls `aiohttp.web.run_app(app, ...)` with default
  `handle_signals=True`. `aiohttp/web_runner.py:283-284` installs
  `loop.add_signal_handler(SIGINT, ...)` and `(SIGTERM, ...)` →
  `_raise_graceful_exit` → `GracefulExit`, caught in `aiohttp/web.py:520`.
  The runner's `cleanup()` (`web_runner.py:295-320`) then runs the app's
  `on_shutdown` and `on_cleanup` hooks.
- Therefore **SIGINT and SIGTERM already reach `server.on_cleanup`**, and
  `on_cleanup` (`server.py:60-72`) currently only cancels the KV monitor task
  and closes the shared aiohttp session. It does **not** touch the dump buffer.
  There is no `on_shutdown` hook registered.
- Consequence: today, in-memory dump state (`_sessions`) is simply lost at exit
  — but since every turn is already written to disk immediately, no captured
  turn is lost either. Once buffering is introduced, this becomes the flush
  seam.
- SIGKILL / hard shutdown is not interceptable; the user has stated that losing
  buffered in-flight messages in that case is acceptable.

### 4. Test + coverage state (a real constraint)

- Baseline: `127 passed`, total coverage **80.63%** against the
  `--cov-fail-under=80` gate configured in `pyproject.toml`.
- `buster_ripper/dump.py` is at **24% coverage** (71 stmts, 54 missed). There is
  no `test_dump.py` and no dump coverage in `test_routes_integration.py`.
- `pyproject.toml` omits `routes/*`, `main.py`, `server.py` from coverage but
  **does not omit `dump.py`**. Any new uncovered lines in `dump.py` therefore
  push the suite toward the 80% floor — bulk-flush work needs tests to stay
  above the gate. `tests/test_server_hooks.py` covers `on_cleanup` and is the
  natural home for flush-hook tests.
- A `RuntimeWarning: coroutine 'start_kv_cache_monitoring' was never awaited`
  appears in the baseline run (from `test_server_hooks.py` mocking
  `asyncio.create_task`); pre-existing, not introduced here.

### 5. No existing buffering abstraction

Grep for `buffer|batch|pending|flush` across `buster_ripper/` and `tests/`
finds only `X-Accel-Buffering: no` (a response header) and an unrelated
comment. There is nothing to extend — a buffering/flush concept is new.

### 6. Deployment / invocation

- `serve_buster_ripper.sh` runs `exec buster-ripper ...` with
  `DUMP_DIR=${HOME}/mllm/prompt-diffs` by default. `exec` means the process
  receiving SIGTERM/SIGINT is the Python process directly.
- No systemd unit or service file in the repo; no documented signal semantics
  in `README.md` or `docs/`.
- `README.md:134-162` documents the dump output layout and the `index.txt`
  summary format as a user-facing contract.

## Existing Invariants In Scope

### EI:buster_ripper.dump/dump_turn/output_layout
- **Kind**: compatibility
- **Location**: `buster_ripper/dump.py:81-106`, `README.md:146-162`
- **Statement**: A session directory contains `turn_000.json` (full body),
  `turn_{N:03d}.diff` for N ≥ 1, and `index.txt` with one human-readable
  summary line per turn.
- **Why it is load-bearing**: This layout is documented user-facing output;
  the index line format is the intended reading interface.
- **Evidence**: `README.md:146-162`; code at `dump.py:81-106`.
- **Status**: intended
- **Can conflict with**: any bulk-write redesign that changes file names,
  numbering, or the index format.

### EI:buster_ripper.dump/dump_turn/turn_zero_is_full_body
- **Kind**: behavioral
- **Location**: `buster_ripper/dump.py:80-85`
- **Statement**: Turn 0 is always written as a full JSON body, never as a diff.
- **Why it is load-bearing**: every later diff is relative to turn 0; a missing
  full baseline makes the whole session directory unreadable.
- **Evidence**: `dump.py:80-85`; `README.md:148`.
- **Status**: intended
- **Can conflict with**: buffering that defers turn 0's write, or drops it if
  the session ends before flush.

### EI:buster_ripper.dump/dump_turn/sequential_diff_chain
- **Kind**: behavioral
- **Location**: `buster_ripper/dump.py:86-108`
- **Statement**: Turn N diffs the rendered text of turn N-1 against turn N;
  `state.prev_text` advances only when a turn is written, and turn numbering is
  per in-memory session state.
- **Why it is load-bearing**: buffering must preserve per-session arrival order;
  out-of-order or dropped flushes silently corrupt every subsequent diff.
- **Evidence**: `dump.py:87-108`.
- **Status**: intended
- **Can conflict with**: concurrent flushes, per-request async offloading, or
  batching that reorders turns.

### EI:buster_ripper.routes.messages/handle_messages/dump_failure_isolated
- **Kind**: behavioral
- **Location**: `buster_ripper/routes/messages.py:55-59`
- **Statement**: Any exception from the dump path is logged and swallowed; it
  must never fail or alter the proxied request.
- **Why it is load-bearing**: the proxy's primary function is serving Claude
  Code; dumping is a side channel.
- **Evidence**: `routes/messages.py:55-59`.
- **Status**: intended
- **Can conflict with**: a flush design that raises out of the request handler
  or that blocks the event loop on synchronous IO.

### EI:buster_ripper.session/session_id/derived_after_normalize
- **Kind**: structural
- **Location**: `buster_ripper/routes/messages.py:31-35`, `session.py:93-98`
- **Statement**: `session_id` = first 12 hex chars of SHA-256 of the system
  prompt text, computed **after** normalization (the billing nonce changes
  every request and must be stripped first for stability).
- **Why it is load-bearing**: it keys both the dump directory and in-memory
  state; ordering changes fragment or merge sessions.
- **Evidence**: comment + code at `routes/messages.py:31-35`.
- **Status**: intended
- **Can conflict with**: nothing planned; recorded because buffering keys on it.

### EI:buster_ripper.server/on_cleanup/graceful_shutdown_hook
- **Kind**: behavioral
- **Location**: `buster_ripper/server.py:60-72`, `web_runner.py:283-320`
- **Statement**: `on_cleanup` is the app's shutdown hook; it runs on graceful
  exit including aiohttp's SIGINT/SIGTERM path. It currently cancels the KV
  monitor task and closes the shared client session.
- **Why it is load-bearing**: it is the only hook guaranteed to run on Ctrl-C
  and SIGTERM — the flush must be wired here (or into an added `on_shutdown`).
- **Evidence**: `server.py:60-72`; aiohttp `web_runner.py:283-320`.
- **Status**: intended
- **Can conflict with**: a flush implementation that assumes a custom signal
  handler is required (it is not) or that runs after the event loop is gone.

## Behaviors to Protect (Regression Testing)

### B-001: Dump output layout and index format
- **Location**: `buster_ripper/dump.py:80-106`, `README.md:146-162`
- **Current Behavior**: Session dir holds `turn_000.json`, `turn_NNN.diff`,
  `index.txt` with one summary line per turn in the documented format.
- **User Impact**: The dump directory is the product of this mode; users read it.
- **Risk if Broken**: high
- **Verification Approach**: unit/integration on `dump_turn` with a temp
  `DUMP_DIR`; assert file names, contents, and index lines.
- **Observable Boundary**: filesystem state under `DUMP_DIR/<session_id>/`.
- **Evidence Available**: temp-dir file reads.
- **Testable Flow**:
  - Given: `config.DUMP_DIR` set to a temp dir
  - When: `dump_turn(body)` is called for turns 0, 1, 2
  - Then: `turn_000.json`, `turn_001.diff`, `turn_002.diff`, `index.txt` exist
    with the documented content shape

### B-002: Diff chain correctness across turns
- **Location**: `buster_ripper/dump.py:87-95`
- **Current Behavior**: Turn N's diff is computed against turn N-1's rendered text.
- **User Impact**: The dump's whole purpose is showing what changed between turns.
- **Risk if Broken**: high
- **Verification Approach**: feed two bodies differing by a known line; assert
  the diff contains exactly that change.
- **Observable Boundary**: contents of `turn_001.diff`.
- **Evidence Available**: temp-dir file read.
- **Testable Flow**:
  - Given: turn 0 body and a turn 1 body with one message changed
  - When: both are dumped
  - Then: `turn_001.diff` contains the added/removed line

### B-003: Dump failure does not break the proxied request
- **Location**: `buster_ripper/routes/messages.py:55-59`
- **Current Behavior**: A raising dump is logged; the request still proxies.
- **User Impact**: Dumping must never take down the proxy.
- **Risk if Broken**: high
- **Verification Approach**: mock `dump_turn` to raise, assert 200 from route.
- **Observable Boundary**: HTTP response status.
- **Evidence Available**: aiohttp test client.
- **Testable Flow**:
  - Given: `DUMP_DIR` set and `dump_turn` patched to raise
  - When: POST `/v1/messages`
  - Then: response is still the upstream success response

### B-004: on_cleanup shutdown behavior
- **Location**: `buster_ripper/server.py:60-72`, `tests/test_server_hooks.py`
- **Current Behavior**: `on_cleanup` cancels the KV monitor task and closes the
  shared session; tolerates a missing task and a cancelled task.
- **User Impact**: Clean shutdown; no leaked tasks/connections.
- **Risk if Broken**: medium
- **Verification Approach**: existing `TestOnCleanup` tests; extend for flush.
- **Observable Boundary**: mock call assertions on session/task.
- **Evidence Available**: existing tests at `tests/test_server_hooks.py:85-151`.
- **Testable Flow**:
  - Given: an app dict with a mocked session
  - When: `on_cleanup(app)` runs
  - Then: session closes and the dump buffer is flushed (after this change)

### B-005: Proxy response passthrough unaffected
- **Location**: `buster_ripper/routes/messages.py:73-268`
- **Current Behavior**: streaming and non-streaming proxying, context-overflow
  retry, header filtering all behave as tested.
- **User Impact**: Core function of the tool.
- **Risk if Broken**: high
- **Verification Approach**: existing `tests/test_routes_integration.py`.
- **Observable Boundary**: HTTP responses from the test client.
- **Evidence Available**: existing integration tests.
- **Testable Flow**: existing suite stays green.

### B-006: Session directory isolation by session_id
- **Location**: `buster_ripper/session.py:93-98`, `dump.py:70-75`
- **Current Behavior**: Distinct system prompts yield distinct session dirs;
  the same prompt reuses one in-memory session state.
- **User Impact**: Prevents cross-session dump contamination.
- **Risk if Broken**: high
- **Verification Approach**: dump two bodies with different system prompts;
  assert two directories.
- **Observable Boundary**: directory listing under `DUMP_DIR`.
- **Evidence Available**: temp-dir listing.
- **Testable Flow**:
  - Given: two bodies with different `system` text
  - When: both dumped
  - Then: two distinct `<sid>/` directories exist

## Anchor Map (draft)

| Term | Binding (file:symbol) | Mode | Invariants | Notes |
|---|---|---|---|---|
| dump entry point | buster_ripper/dump.py:dump_turn | extend | EI:.../output_layout, EI:.../turn_zero_is_full_body, EI:.../sequential_diff_chain | becomes buffer-append + conditional flush |
| per-session dump state | buster_ripper/dump.py:SessionState | extend | EI:.../sequential_diff_chain | extended with buffered pending writes |
| dump write buffer | candidates: buster_ripper/dump.py:SessionState | NEW | — | RB-001; no buffering abstraction exists in repo |
| bulk flush | buster_ripper/dump.py (new symbol) | NEW | EI:.../output_layout, EI:.../turn_zero_is_full_body | RB-001; must preserve layout |
| dump dir config | buster_ripper/config.py:DUMP_DIR | reuse | — | already the mode switch |
| shutdown flush seam | buster_ripper/server.py:on_cleanup | extend | EI:.../graceful_shutdown_hook | runs on aiohttp SIGINT/SIGTERM graceful exit |
| signal handling | aiohttp web_runner (SIGINT/SIGTERM → GracefulExit) | reuse | EI:.../graceful_shutdown_hook | native; no custom handler needed |
| dump caller | buster_ripper/routes/messages.py:handle_messages | extend | EI:.../dump_failure_isolated | only interception site today |
| dump index | buster_ripper/dump.py index.txt append | extend | EI:.../output_layout | currently one append per turn |

## Reuse Evidence (draft)

| Evidence | NEW anchor | Name search | Role/peer search | Native capability | Conclusion |
|---|---|---|---|---|---|
| RB-001 | dump write buffer + bulk flush | `grep -rni "buffer\|batch\|pending\|flush"` over `buster_ripper/` and `tests/` → only `X-Accel-Buffering` header and an unrelated comment; no buffering/batching module, class, or function | inspected `dump.py`, `session.py`, `server.py`, `routes/*` call paths and module state — no queue, no writer abstraction, no existing flush role to extend | Python stdlib has no project-fit facility for "session-keyed pending write buffer"; `io.BufferedWriter` is OS-level and does not own session-keyed turn semantics. Attempted operation N/A — this is repository-domain behavior, not a host facility | no candidate owns the required role; a new buffer/flush concept in `dump.py` is required |

## Deferred Questions

- **Flush trigger policy**: is the buffer flushed only at shutdown, or also on a
  size/turn-count threshold during normal operation? "Dump to disk in bulk"
  could mean one write at exit or periodic batches.
  - Status: DEFERRED for /codify-review
  - Impact: Determines whether a long-running session ever produces readable
    dump files before exit, and how much is lost on SIGKILL. Also affects
    whether a "reduced writes" claim is measurable.
  - Agent analysis: ratification recommendation — flush at shutdown **and** on a
    bounded threshold (e.g. buffered bytes or turn count) so a long session is
    not an all-or-nothing buffer; evidence: user explicitly accepts loss on hard
    shutdown, which only makes sense if flushing is otherwise deferred.
- **Meaning of "in bulk"**: preserve the current per-file layout and merely
  batch the writes in one flush, or collapse each session's turns into a single
  appended/aggregated file?
  - Status: DEFERRED for /codify-review
  - Impact: Collapsing files would break the documented layout
    (`EI:.../output_layout`) and the diff-reading workflow.
  - Agent analysis: ratification recommendation — keep the documented layout
    (`turn_000.json`, `turn_NNN.diff`, `index.txt`) and batch the writes; the
    user asked to reduce *write operations*, not to change the output format.
- **Should buffering also cover turn 0's full body** (potentially hundreds of KB)?
  - Status: DEFERRED for /codify-review
  - Impact: Buffering turn 0 defers the baseline; losing it on hard shutdown
    leaves a session directory with no baseline.
  - Agent analysis: none — genuinely open; the user accepted losing in-flight
    data, which may or may not extend to the turn-0 baseline.
- **Should buffering extend to `/v1/chat/completions` / other routes?**
  - Status: DEFERRED for /codify-review
  - Impact: Scope boundary for "message interception and dumping mode".
  - Agent analysis: ratification recommendation — out of scope; only
    `/v1/messages` dumps today, so "dumping mode" is that path only.

## Conflicts / Concerns

- Coverage gate: `dump.py` at 24% with an 80% total gate means new dump code
  needs tests or the suite fails. Flag for the plan, not a blocker.
- The `assert config.DUMP_DIR is not None` in `dump_turn` is a bare assert
  (stripped under `python -O`); a buffer/flush refactor is a natural point to
  replace it with a real guard, but that is a decision for organize/decide.

## Questions Asked

No questions - deferred mode (no explicit active-mode instruction in the
orchestrator prompt); all clarifications recorded above as deferred questions
with agent analysis.

## Validation Performed

- `pytest -q` baseline: 127 passed, coverage 80.63% (gate 80%).
- Grep verification for signal handling, dump callers, buffering abstractions.
- Read aiohttp `web_runner.py` / `web.py` to confirm the SIGINT/SIGTERM →
  `on_cleanup` path rather than assuming it.
