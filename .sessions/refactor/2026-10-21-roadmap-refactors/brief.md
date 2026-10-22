# Restructure the three remaining refactor rows

- Goal: take RM-001, RM-002 and RM-025 as one behavior-preserving refactor session.
- Target: RM-001, RM-002, RM-025 (freed for this session from the stale `origin/fix/tool-reference-scrub` branch, last commit 2026-09-17).
- In scope: `buster_ripper/routes/messages.py`, `buster_ripper/routes/chat.py`, `buster_ripper/dump.py`, `tests/test_compaction.py`
- Importers: `buster_ripper/routes/messages.py`, `buster_ripper/routes/stream_relay.py`, `buster_ripper/routes/chat.py`, `buster_ripper/dump.py`, `buster_ripper/dump_render.py`, `tests/test_stream_relay_commit.py`, `tests/test_preserved_thinking.py`, `tests/test_dump.py`, `tests/test_dump_records.py`, `tests/test_dump_triggers.py`, `tests/test_compaction.py`, `tests/test_compaction_poll.py`, `tests/test_compaction_ema.py`
- Out of scope: `buster_ripper_legacy.py`, the `tests/qa/` QA gate modules, the RM-010/RM-011/RM-012 design passes, and the RM-013/RM-014/RM-015/RM-022/RM-023 future-scope rows; no behavior change.
- Verify: `.venv/bin/python -m pytest tests/test_messages_routes.py tests/test_chat_routes.py tests/test_messages_request_bytes.py tests/test_chat_request_bytes.py tests/test_stream_relay_commit.py tests/test_header_parity.py tests/test_context_overflow.py tests/test_context_overflow_retry.py tests/test_upstream_answer_relay.py tests/test_upstream_error_relay.py tests/test_transport_parity.py tests/test_preserved_thinking.py tests/test_proxy_errors.py tests/test_image.py tests/test_dump.py tests/test_dump_flush.py tests/test_dump_records.py tests/test_dump_triggers.py tests/test_dump_write_count.py tests/test_server_hooks.py tests/test_compaction.py -o addopts= && .venv/bin/python -m pytest tests/qa/test_qap_ms001.py tests/qa/test_qap_ms002.py -o addopts=`
- Final verify: `uv run pytest`
- Worktree: /Users/alexhubitski/Worktrees/buster-ripper/refactor-roadmap

## Decided
- 2026-10-21: "Free the three refactor rows (Recommended)" — RM-001, RM-002 and RM-025 are freed for this session. The stale `origin/fix/tool-reference-scrub` branch is left untouched.
## Collected
| # | Candidate | Why | Recorded as | State |
|---|---|---|---|---|
| 1 | buster_ripper/routes/messages.py | 341 lines (review: <=300); 19 issues in 7 functions — High Complexity 1, Deep Nesting 2, Long Function 2, Medium Complexity 2, Medium Length 4, Medium Nesting 3, Medium Parameters 5; `handle_messages` L:28 N:2 C:12 (High Complexity), `_non_stream_single_request` L:34 N:4 C:8 P:4 (Long Function, Deep Nesting), `handle_streaming` L:25 N:3 C:8 P:4 (Deep Nesting), `_stream_single_request` L:35 N:2 C:4 P:4 (Long Function); ahead of RM-010 · transport-error envelope (`messages.py:289-334`), RM-011 · content-coding posture (`:156,299`), RM-012 · route-input posture; freed by this session from `origin/fix/tool-reference-scrub` (stale since 2026-09-17) | RM-001 | free |
| 2 | buster_ripper/dump.py | 319 lines (review: <=300) — the row records 316, so its size note is stale; 1 issue in 1 function — Medium Complexity 1: `_render_body` L:12 N:1 C:6 P:1; ahead of RM-013 · dump-format default (`:174,186,196`), RM-014 · machine record, RM-015 · dump viewer, RM-022 · restart design (`:147-158`, `:295-305`), RM-023 · append rollback (`:195-200`, `:300-307`); freed by this session from `origin/fix/tool-reference-scrub` (stale since 2026-09-17) | RM-002 | free |
| 3 | buster_ripper/routes/chat.py | no file-level size warning — the row's 299/300 capacity note is not re-raised by the scan; 14 issues in 8 functions — Medium Complexity 4, Medium Nesting 6, Medium Length 2, Medium Parameters 1; `handle_chat` L:26 N:2 C:6 (Medium Length, Medium Nesting, Medium Complexity), `_apply_request_transforms` L:20 N:2 C:8 (Medium Complexity), `_make_chat_request` L:15 N:2 C:3 P:5 (Near Tolerance Nesting Parameters); ahead of RM-010 · transport-error envelope (`chat.py:134-152`), RM-011 · content-coding posture (`:143`), RM-012 · route-input posture; freed by this session from `origin/fix/tool-reference-scrub` (stale since 2026-09-17) | RM-001 | free |
| 4 | tests/test_compaction.py | 451 lines (review: <=300); 2 issues in 2 functions — Medium Length 2: `test_poll_kv_cache_updates_global_cache` L:20 N:0 C:2 P:1, `test_poll_kv_cache_uses_session_get` L:17 N:0 C:2 P:1; ahead of RM-025 · drift-helper split (no other open row names the file); freed by this session from `origin/fix/tool-reference-scrub` (stale since 2026-09-17) | RM-025 | free |
- Verify: `.venv/bin/python -m pytest tests/test_messages_routes.py tests/test_chat_routes.py tests/test_messages_request_bytes.py tests/test_chat_request_bytes.py tests/test_stream_relay_commit.py tests/test_header_parity.py tests/test_context_overflow.py tests/test_context_overflow_retry.py tests/test_upstream_answer_relay.py tests/test_upstream_error_relay.py tests/test_transport_parity.py tests/test_preserved_thinking.py tests/test_proxy_errors.py tests/test_image.py tests/test_dump.py tests/test_dump_flush.py tests/test_dump_records.py tests/test_dump_triggers.py tests/test_dump_write_count.py tests/test_server_hooks.py tests/test_compaction.py -o addopts= && .venv/bin/python -m pytest tests/qa/test_qap_ms001.py tests/qa/test_qap_ms002.py -o addopts=`
- Final verify: `uv run pytest`
- Tool gaps: `quality-backlog.md` is absent from the tree (deleted by the 02.00 squash `3323551`), so its `## Refactoring` section was read from `git show 3323551^:.sessions/codify/02.00-2026-09-13-message-dump-bulk-flush--9bf7afff5573/quality-backlog.md`. No CODIFY session at HEAD carries an Anchor Map (the only one, 02.00, is complete and squashed), so no `planned:` files could be derived. The scan reports no file-level line count for `buster_ripper/routes/chat.py` because it raises no size warning, so RM-001's 299/300 capacity note could not be re-measured from the scan alone.

## Safety net
- Ran: `.venv/bin/python -m pytest tests/test_messages_routes.py tests/test_chat_routes.py tests/test_messages_request_bytes.py tests/test_chat_request_bytes.py tests/test_stream_relay_commit.py tests/test_header_parity.py tests/test_context_overflow.py tests/test_context_overflow_retry.py tests/test_upstream_answer_relay.py tests/test_upstream_error_relay.py tests/test_transport_parity.py tests/test_preserved_thinking.py tests/test_proxy_errors.py tests/test_image.py tests/test_dump.py tests/test_dump_flush.py tests/test_dump_records.py tests/test_dump_triggers.py tests/test_dump_write_count.py tests/test_server_hooks.py tests/test_compaction.py -o addopts= && .venv/bin/python -m pytest tests/qa/test_qap_ms001.py tests/qa/test_qap_ms002.py -o addopts=` -> 364 passed + 7 passed, 0 skipped (the worktree had no `.venv`; `uv sync` built it once, then the brief's exact command ran unchanged)
- Scan: `codify scan path buster_ripper/routes/messages.py buster_ripper/routes/chat.py buster_ripper/dump.py tests/test_compaction.py` -> `buster_ripper/routes/messages.py` — file 341 lines (review: <=300); noise decomposition 36 trimmable full-line comments; 19 issues in 7 functions: Deep Nesting 2, High Complexity 1, Long Function 2, Medium Complexity 2, Medium Length 4, Medium Nesting 3, Medium Parameters 5; top: `_stream_single_request` L:35 N:2 C:4 P:4 (Long Function, Medium Nesting, Medium Parameters), `_non_stream_single_request` L:34 N:4 C:8 P:4 (Long Function, Deep Nesting, Medium Complexity, Medium Parameters), `handle_messages` L:28 N:2 C:12 P:1 (Medium Length, Medium Nesting, High Complexity), `handle_streaming` L:25 N:3 C:8 P:4 (Medium Length, Deep Nesting, Medium Complexity, Medium Parameters), `handle_non_streaming` L:21 N:2 C:5 P:4 (Medium Length, Medium Nesting, Medium Parameters); the detailed tree also flags `_relay_after_commit` L:9 N:1 C:3 P:4 (Medium Parameters). `buster_ripper/routes/chat.py` — 299 lines (no file-size finding); 14 issues in 8 functions: Medium Complexity 4, Medium Length 2, Medium Nesting 6, Medium Parameters 1; one near-tolerance (Nesting and Parameters); top: `handle_chat` L:26 N:2 C:6 P:1 (Medium Length, Medium Nesting, Medium Complexity), `_apply_request_transforms` L:20 N:2 C:8 P:2, `_transform_eval_response` L:15 N:2 C:7 P:4, `_make_chat_request` L:15 N:2 C:3 P:5 (Near Tolerance Nesting Parameters), `_apply_response_transforms` L:12 N:2 C:5 P:2 (Medium Nesting). `buster_ripper/dump.py` — file 319 lines (review: <=300); noise decomposition 5 trimmable full-line comments; 1 issue in 1 function: Medium Complexity 1; `_render_body` L:12 N:1 C:6 P:1. `tests/test_compaction.py` — file 451 lines (review: <=300); noise decomposition 15 trimmable full-line comments; 2 issues in 2 functions: Medium Length 2; `test_poll_kv_cache_updates_global_cache` L:20 N:0 C:2 P:1, `test_poll_kv_cache_uses_session_get` L:17 N:0 C:2 P:1
- Sizes: buster_ripper/routes/messages.py: 341, buster_ripper/dump.py: 319, tests/test_compaction.py: 451 (buster_ripper/routes/chat.py is 299, under the bound, and the scan prints no file-size finding for it)
- Public paths kept: none (no module in scope defines `__all__`; `handle_messages`/`handle_chat` are registered by `buster_ripper/server.py`, which stays an in-repository importer, and `dump_turn`/`flush_all`/`poll_flush` stay in `buster_ripper/dump.py`)

## Map

### U0 · comments
| # | Kind | What | From | To |
|---|---|---|---|---|
| 1 | shrink | `_relay_after_commit` docstring (17 lines: prepare/commit, the second head, the catch class, `CancelledError`) to 4 lines; the `_request_payload` docstring (6) to 3; the `handle_streaming` docstring (4) to 3; `_stream_single_request`'s no-deadline comment (5) to 3 and its redirect comment (2) to 1; `_non_stream_single_request`'s lenient-decode comment (5) to 2 and its token-fallback comment (2) to 1; `handle_messages`'s normalize-order comment (3) to 2; `_relay_after_commit`'s position-not-cause comment (2) to 1 | buster_ripper/routes/messages.py | kept reasons: tests/test_stream_relay_commit.py's module docstring (one head/one terminal, the post-commit catch); the rest stay as one- or two-line comments |
| 2 | shrink | module docstring (13 lines: the three formats, their artifacts and flush triggers) to 4 lines; the `flush_all` docstring (4) to 3; the `poll_flush` docstring (3) to 2 | buster_ripper/dump.py | kept reasons: README.md §Dump mode + §Collapsed dump formats already carry the three formats, their artifacts and the flush triggers |
| 3 | shrink | `_TickSeam` docstring (7 lines) to 3; the `_kv_answer` docstring (10) to 4; the `_drive_ticks` docstring (9) to 4; `test_ema_first_reading_is_raw`'s docstring (4) to 2; `test_ema_smoothing_reduces_bounces`'s docstring (7) to 3 | tests/test_compaction.py | kept reasons: each witness's mutation stays in its docstring; the composed recipe stays in the `CO:` record at invariants/index.md:303 |
- Narrow tests: tests/test_messages_routes.py, tests/test_stream_relay_commit.py, tests/test_context_overflow.py, tests/test_context_overflow_retry.py, tests/test_dump.py, tests/test_dump_records.py, tests/test_dump_triggers.py, tests/test_compaction.py
- Green after U0: no file drops a unit — messages.py 341 -> ~318, dump.py 319 -> ~308 and test_compaction.py 451 -> ~430 all stay over 300, so every split unit below still runs; chat.py (299) has no over-long comment and no split unit.

### U1 · buster_ripper/routes/messages.py: move the post-commit SSE relay into its own module
| # | Kind | What | From | To |
|---|---|---|---|---|
| 1 | move | `_relay_after_commit`, `_note_message_start` | buster_ripper/routes/messages.py | buster_ripper/routes/stream_relay.py |
- Importers to update: buster_ripper/routes/messages.py (import block only: `from .stream_relay import _relay_after_commit`; `find_message_start` leaves with the relay, `time`, `update_session_stats` and the rest stay)
- Witnesses to re-point: none (`invariants/` names neither function; the only records naming them are the squashed session records)
- Tests with import lines or patch targets to update: tests/test_stream_relay_commit.py (`messages_routes._relay_after_commit` at `:216` and `:233` -> the new module's function; the `messages_routes.update_session_stats` patch target at `:139` and `:252` -> the new module's `update_session_stats`)
- Narrow tests: tests/test_stream_relay_commit.py

### U2 · buster_ripper/routes/messages.py: reshape the two flagged handlers and dedupe the two answer arms
| # | Kind | What | From | To |
|---|---|---|---|---|
| 1 | reshape | `handle_messages`: extract `_prepare_body(request, body) -> str` — normalize, compute the session id, strip/cache images, inject the preserved-thinking pair, log the changes; returns the session id | buster_ripper/routes/messages.py | buster_ripper/routes/messages.py |
| 2 | reshape | `handle_streaming`: extract `_stream_overflow(result)` — the `isinstance(..., web.Response) and status >= 400` probe that decodes the relayed bytes and parses the overflow; the loop keeps its `continue`/400/`return result` arms | buster_ripper/routes/messages.py | buster_ripper/routes/messages.py |
| 3 | reshape | `_stream_single_request`: extract `_upstream_answer(resp)` for the identical redirect and error answer blocks, and take one `if resp.status >= 300:` arm instead of two | buster_ripper/routes/messages.py | buster_ripper/routes/messages.py |
| 4 | reshape | `_non_stream_single_request`: extract `_raise_context_overflow(content, input_tokens)` — the `status >= 400` probe and its token fallback — and `_note_non_stream_usage(request, sid, input_tokens, elapsed_ms)` — the usage record and its verbose line | buster_ripper/routes/messages.py | buster_ripper/routes/messages.py |
- Importers to update: none
- Witnesses to re-point: none
- Tests with import lines or patch targets to update: none (the tests drive the route; only `dump_turn` is patched by path and it stays)
- Calls today: `handle_messages` by tests/test_messages_routes.py (`TestMessagesRoute`, `TestDumpFailureIsolated`); `handle_streaming` by tests/test_context_overflow_retry.py (`TestStreamingOverflowRetry`); `_stream_single_request` by tests/test_stream_relay_commit.py (`TestPostCommitFaults`, `TestRelayBehaviour`) and tests/test_context_overflow_retry.py (`TestStreamingOverflowRetry`); `_non_stream_single_request` by tests/test_context_overflow_retry.py (`TestNonStreamingOverflowRetry`) and tests/test_messages_routes.py (`TestMessagesRoute`)
- Narrow tests: tests/test_messages_routes.py, tests/test_context_overflow.py, tests/test_context_overflow_retry.py, tests/test_stream_relay_commit.py, tests/test_transport_parity.py

### U3 · buster_ripper/routes/chat.py: reshape the handler and the eval transforms
| # | Kind | What | From | To |
|---|---|---|---|---|
| 1 | reshape | `handle_chat`: extract `_chat_with_overflow_retry(request, body, headers, profile)` — the backend dispatch and SGLang retry loop — leaving `handle_chat` the parse, the request transforms, the headers and the call | buster_ripper/routes/chat.py | buster_ripper/routes/chat.py |
| 2 | reshape | `_apply_request_transforms`: extract `_inject_eval_kwargs(body)` — the `profile.inject_chat_template_kwargs` block | buster_ripper/routes/chat.py | buster_ripper/routes/chat.py |
| 3 | reshape | `_make_chat_request`: derive `session = request.app["session"]` inside the function and drop the `session` parameter (P:5 -> P:4); `_chat_with_overflow_retry` does the same, so `handle_chat` no longer reads the session | buster_ripper/routes/chat.py | buster_ripper/routes/chat.py |
| 4 | test | add `test_eval_response_transforms_today` to tests/test_preserved_thinking.py — calls `_transform_eval_response` directly with the glm47 profile and pins today's `(content, fixed)` on the think-split, reasoning-copy and code-fence arms, before the reshape | tests/test_preserved_thinking.py | tests/test_preserved_thinking.py |
| 5 | reshape | `_transform_eval_response`: drive the enabled transforms from a step list, each step `(msg, content) -> (content, changed)`, so the changed/fixed bookkeeping appears once; `_copy_reasoning_to_content` takes the ignored `content` parameter to share that shape | buster_ripper/routes/chat.py | buster_ripper/routes/chat.py |
- Importers to update: none
- Witnesses to re-point: none
- Tests with import lines or patch targets to update: none (`chat_routes._prepare_headers` and `chat_routes._relay_chat_stream` stay in the module)
- Calls today: `handle_chat` by tests/test_chat_routes.py (`TestChatRoute`); `_apply_request_transforms` by tests/test_preserved_thinking.py (`TestEvalChatTemplateKwargsGuard`); `_make_chat_request` by tests/test_chat_routes.py (`TestChatRoute`) and tests/test_stream_relay_commit.py (`TestChatPostCommit`); `_transform_eval_response` by tests/test_chat_routes.py (`test_chat_nonstream_compatible_json_transforms`) and tests/test_preserved_thinking.py, both through the route
- Narrow tests: tests/test_chat_routes.py, tests/test_preserved_thinking.py, tests/test_context_overflow_retry.py, tests/test_upstream_answer_relay.py, tests/test_header_parity.py

### U4 · buster_ripper/dump.py: move the body rendering into its own module
| # | Kind | What | From | To |
|---|---|---|---|---|
| 1 | move | `_render_tools`, `_system_text`, `_content_text`, `_render_body` | buster_ripper/dump.py | buster_ripper/dump_render.py |
- Importers to update: buster_ripper/dump.py (import block only: `from .dump_render import _render_body`; `json` and `difflib` both stay, so `CL:buster_ripper.dump/import-topology` still holds)
- Witnesses to re-point: invariants/index.md:144 and :145 (`repo:buster_ripper.dump#_render_body` -> `repo:buster_ripper.dump_render#_render_body`); invariants/buster_ripper/assumption-claims.md:122 and :136 (Subject), :125 and :139 (Source path), :128 and :142 (Invalidates-On path) -> `buster_ripper/dump_render.py`
- Tests with import lines or patch targets to update: tests/test_dump.py:18, tests/test_dump_records.py:18 (`from buster_ripper.dump import _render_body` -> `from buster_ripper.dump_render import _render_body`); tests/test_dump_triggers.py (`dump._render_body` at `:110`, `:121`, `:140`, `:167` -> the new module's `_render_body`)
- Narrow tests: tests/test_dump.py, tests/test_dump_records.py, tests/test_dump_triggers.py, tests/test_dump_write_count.py

### U5 · buster_ripper/dump_render.py: extract the message section
| # | Kind | What | From | To |
|---|---|---|---|---|
| 1 | reshape | `_render_body`: extract `_render_message(index, msg)` — the role read, the content text and the `[message i / role]` section | buster_ripper/dump_render.py | buster_ripper/dump_render.py |
- Importers to update: none
- Witnesses to re-point: none (the `_render_body` records re-pointed in U4 already name this module)
- Tests with import lines or patch targets to update: none
- Calls today: `_render_body` directly by tests/test_dump.py and tests/test_dump_records.py (`_render_body` imported at `:18` and called in the record and diff assertions)
- Narrow tests: tests/test_dump.py, tests/test_dump_records.py, tests/test_dump_triggers.py

### U6 · tests/test_compaction.py: split the poll and EMA-drift witnesses into their own modules
| # | Kind | What | From | To |
|---|---|---|---|---|
| 1 | move | `test_poll_kv_cache_uses_session_get`, `test_poll_kv_cache_handles_metrics_error`, `test_poll_kv_cache_updates_global_cache`, `test_poll_kv_cache_uses_timeout`, `TestPollKvCacheWithSession` | tests/test_compaction.py | tests/test_compaction_poll.py |
| 2 | move | `_TickSeam`, `_kv_answer`, `_settle`, `_drive_ticks`, `test_ema_first_reading_is_raw`, `test_ema_smoothing_reduces_bounces` | tests/test_compaction.py | tests/test_compaction_ema.py |
- Importers to update: none (both new modules; the poll module pulls `asyncio`, `aiohttp`, `pytest`, `compaction`, the EMA module pulls `asyncio`, `contextlib`, `pytest`, `compaction`, `config`)
- Witnesses to re-point: invariants/index.md:303 (the `CO:` producer parenthetical), invariants/buster_ripper/assumption-claims.md:3215 (Boundary), :3219 (Source; the poll citations split to tests/test_compaction_poll.py and the drift citations to tests/test_compaction_ema.py), :3223 and :3225 (Invalidates-On) -> tests/test_compaction_ema.py
- Tests with import lines or patch targets to update: tests/test_compaction.py's import block (drops `asyncio`, `contextlib`, `aiohttp`, `pytest` and `config`; keeps `compact_token_count` and `should_nudge_compact`)
- Narrow tests: tests/test_compaction.py, tests/test_compaction_poll.py, tests/test_compaction_ema.py (the two new modules are not selected by the brief's Verify command — run them explicitly)

## U0 Forward
- Moved: nothing
- Reshaped: nothing
- Files: buster_ripper/routes/messages.py, buster_ripper/dump.py, tests/test_compaction.py
- Tests edited: none
- Ran: `.venv/bin/python -m pytest tests/test_messages_routes.py tests/test_stream_relay_commit.py tests/test_context_overflow.py tests/test_context_overflow_retry.py tests/test_dump.py tests/test_dump_records.py tests/test_dump_triggers.py tests/test_compaction.py -o addopts=` -> 144 passed (after each row; 50, 73 and 21 passed for rows 1-3)
- Ran: brief `Verify` -> 364 passed, 3 pre-existing RuntimeWarnings + 7 passed
- Scan: `buster_ripper/routes/messages.py` — 318 lines (review: <=300), 30 trimmable full-line comments, 19 issues in 7 functions; top: `_stream_single_request` L:35 N:2 C:4 P:4, `_non_stream_single_request` L:34 N:4 C:8 P:4, `handle_messages` L:28 N:2 C:12 P:1, `handle_streaming` L:25 N:3 C:8 P:4, `handle_non_streaming` L:21 N:2 C:5 P:4. `buster_ripper/dump.py` — 308 lines (review: <=300), 5 trimmable full-line comments, 1 issue in 1 function: `_render_body` L:12 N:1 C:6 P:1
- Sizes: buster_ripper/routes/messages.py: 318, buster_ripper/dump.py: 308, tests/test_compaction.py: 434
- Noticed: messages.py's 19 function-level issues are unchanged pre-existing debt — the unit's U2 rows reshape them
  Ask: no — U2 reshapes them
- Noticed: dump.py's `_render_body` Medium Complexity is pre-existing — the unit's U5 row reshapes it
  Ask: no — U5 reshapes it
- Noticed: tests/test_server_hooks.py's three never-awaited-coroutine RuntimeWarnings are unchanged test noise
  Ask: no — test hygiene only, no user-visible wrong result

## U0 Yield
- Verdict: ACCEPT
- Ran: `git status` + `git diff -M -C` -> only `buster_ripper/routes/messages.py`, `buster_ripper/dump.py`, `tests/test_compaction.py` and the brief are changed; every changed line in the three code files is a comment, docstring line or blank line
- Ran: `python -` (AST compare, docstrings stripped, working tree vs `HEAD`) for the three files -> identical AST for all three; every docstring that existed at `HEAD` still exists
- Ran: `.venv/bin/python -m pytest tests/test_messages_routes.py tests/test_chat_routes.py tests/test_messages_request_bytes.py tests/test_chat_request_bytes.py tests/test_stream_relay_commit.py tests/test_header_parity.py tests/test_context_overflow.py tests/test_context_overflow_retry.py tests/test_upstream_answer_relay.py tests/test_upstream_error_relay.py tests/test_transport_parity.py tests/test_preserved_thinking.py tests/test_proxy_errors.py tests/test_image.py tests/test_dump.py tests/test_dump_flush.py tests/test_dump_records.py tests/test_dump_triggers.py tests/test_dump_write_count.py tests/test_server_hooks.py tests/test_compaction.py -o addopts= && .venv/bin/python -m pytest tests/qa/test_qap_ms001.py tests/qa/test_qap_ms002.py -o addopts=` -> 364 passed, 3 pre-existing RuntimeWarnings + 7 passed (Safety net: 364 + 7, 0 skipped)
- Scan: `buster_ripper/routes/messages.py` — 318 lines, 19 issues in 7 functions, same functions and metrics as the Safety net (`_stream_single_request` L:35 N:2 C:4 P:4, `_non_stream_single_request` L:34 N:4 C:8 P:4, `handle_messages` L:28 N:2 C:12 P:1, `handle_streaming` L:25 N:3 C:8 P:4, `handle_non_streaming` L:21 N:2 C:5 P:4); `buster_ripper/dump.py` — 308 lines, 1 issue in 1 function `_render_body` L:12 N:1 C:6 P:1; `tests/test_compaction.py` — 434 lines, 2 issues in 2 functions `test_poll_kv_cache_updates_global_cache` L:20 and `test_poll_kv_cache_uses_session_get` L:17. Nothing new, nothing worse
- Sizes: buster_ripper/routes/messages.py: 318, buster_ripper/dump.py: 308, tests/test_compaction.py: 434
- Findings: none
- Noticed: tests/test_compaction.py ends at 434, 4 lines over the map's ~430 estimate (its `_TickSeam`/`_kv_answer` shrink targets were 7->3 and 10->4, delivered as 4 and 5, and row 1's token-fallback comment was already one line at `HEAD`); all three files stay over 300, so the split units are unaffected
  Ask: no — the estimate was approximate and no unit is dropped
- Noticed: messages.py's 19 function-level issues and dump.py's `_render_body` Medium Complexity are unchanged pre-existing debt — the unit's U2/U5 rows reshape them
  Ask: no — later units own them
- Noticed: tests/test_server_hooks.py's three never-awaited-coroutine RuntimeWarnings are unchanged test noise
  Ask: no — test hygiene only, no user-visible wrong result

## U1 Forward
- Moved: `_relay_after_commit`, `_note_message_start` -> buster_ripper/routes/stream_relay.py (cut and pasted byte-identical, AST-compared against `HEAD`; new module: one-line docstring, `time`, `config`, `update_session_stats`, `find_message_start`)
- Reshaped: nothing
- Files: buster_ripper/routes/messages.py, buster_ripper/routes/stream_relay.py (new), tests/test_stream_relay_commit.py
- Tests edited: tests/test_stream_relay_commit.py: import `from buster_ripper.routes import messages as messages_routes` -> `from buster_ripper.routes import stream_relay` (it had no remaining use); patch targets `messages_routes.update_session_stats` at :139 and :252 and `messages_routes._relay_after_commit` at :216 and :233 -> `stream_relay.*` — no assertion changed
- Ran: `.venv/bin/python -m pytest tests/test_stream_relay_commit.py -o addopts=` -> 16 passed
- Ran: brief `Verify` -> 364 passed, 3 pre-existing RuntimeWarnings + 7 passed
- Scan: `buster_ripper/routes/messages.py` — 17 issues in 5 functions (Deep Nesting 2, High Complexity 1, Long Function 2, Medium Complexity 2, Medium Length 3, Medium Nesting 3, Medium Parameters 4); top: `_stream_single_request` L:35 N:2 C:4 P:4, `_non_stream_single_request` L:34 N:4 C:8 P:4, `handle_messages` L:28 N:2 C:12 P:1, `handle_streaming` L:25 N:3 C:8 P:4, `handle_non_streaming` L:21 N:2 C:5 P:4; no file-size finding (was 318, now 280). `buster_ripper/routes/stream_relay.py` — 2 issues in 2 functions (Medium Length 1, Medium Parameters 1): `_note_message_start` L:16 N:1 C:4 P:3 (Medium Length), `_relay_after_commit` L:9 N:1 C:3 P:4 (Medium Parameters)
- Sizes: buster_ripper/routes/messages.py: 280, buster_ripper/routes/stream_relay.py: 45
- Noticed: the two issues the scan prints in stream_relay.py are the same two the moved functions carried at `HEAD` (`_note_message_start` Medium Length, `_relay_after_commit` Medium Parameters) — debt that moved, and no unit row reshapes it
  Ask: no — text-only move, later rows own the reshapes
- Noticed: messages.py's 17 remaining function issues are the pre-existing debt U2 reshapes
  Ask: no — U2 owns them
- Noticed: tests/test_transport_parity.py:17's module docstring names `_relay_after_commit` without a module path — prose, not an import line or patch target, so left alone
  Ask: no — a comment cannot change behaviour
- Noticed: tests/test_server_hooks.py's three never-awaited-coroutine RuntimeWarnings are unchanged test noise
  Ask: no — test hygiene only, no user-visible wrong result

## U1 Yield
- Verdict: ACCEPT
- Ran: `git status` + `git diff -M -C --no-ext-diff` -> only `buster_ripper/routes/messages.py`, the new `buster_ripper/routes/stream_relay.py`, `tests/test_stream_relay_commit.py` and the brief changed; `messages.py`'s diff is one removed import (`find_message_start`, which moved), one added import (`from .stream_relay import _relay_after_commit`) and the two removed definitions, nothing else
- Ran: `git show HEAD:buster_ripper/routes/messages.py` vs `buster_ripper/routes/stream_relay.py` (byte diff + AST dump of both definitions) -> `_relay_after_commit` and `_note_message_start` byte-identical and AST-identical; the new file's only added lines are its one-line docstring and its imports; `_relay_after_commit`'s sole call site (`messages.py:175`) is unchanged
- Ran: `git diff -- tests/test_stream_relay_commit.py` -> five changed lines, all module-path updates (the `messages_routes` import and the four `messages_routes.*` references at :139, :216, :233, :252); no assertion, no test body changed
- Ran: `.venv/bin/python -m pytest tests/test_stream_relay_commit.py -o addopts=` -> 16 passed
- Ran: `.venv/bin/python -m pytest tests/test_messages_routes.py tests/test_chat_routes.py tests/test_messages_request_bytes.py tests/test_chat_request_bytes.py tests/test_stream_relay_commit.py tests/test_header_parity.py tests/test_context_overflow.py tests/test_context_overflow_retry.py tests/test_upstream_answer_relay.py tests/test_upstream_error_relay.py tests/test_transport_parity.py tests/test_preserved_thinking.py tests/test_proxy_errors.py tests/test_image.py tests/test_dump.py tests/test_dump_flush.py tests/test_dump_records.py tests/test_dump_triggers.py tests/test_dump_write_count.py tests/test_server_hooks.py tests/test_compaction.py -o addopts= && .venv/bin/python -m pytest tests/qa/test_qap_ms001.py tests/qa/test_qap_ms002.py -o addopts=` -> 364 passed, 2 pre-existing RuntimeWarnings + 7 passed (Safety net: 364 + 7, 0 skipped)
- Ran: `codify knowledge sync-registry` (preview) -> `current`, no pending sync; `codify knowledge check --profile strict` -> 5 diagnostics, all pre-existing (`invariants/` is untouched by this unit)
- Scan: `buster_ripper/routes/messages.py` — 17 issues in 5 functions (Deep Nesting 2, High Complexity 1, Long Function 2, Medium Complexity 2, Medium Length 3, Medium Nesting 3, Medium Parameters 4): `_stream_single_request` L:35 N:2 C:4 P:4, `_non_stream_single_request` L:34 N:4 C:8 P:4, `handle_messages` L:28 N:2 C:12 P:1, `handle_streaming` L:25 N:3 C:8 P:4, `handle_non_streaming` L:21 N:2 C:5 P:4; `buster_ripper/routes/stream_relay.py` — 2 issues in 2 functions (Medium Length 1, Medium Parameters 1): `_note_message_start` L:16 N:1 C:4 P:3, `_relay_after_commit` L:9 N:1 C:3 P:4; no file-size finding on either. Together 19 issues in 7 functions, the same set and metrics as the Safety net — `_relay_after_commit`'s L:9 N:1 C:3 P:4 is unchanged, and nothing is printed that the Safety net did not already print
- Sizes: buster_ripper/routes/messages.py: 280, buster_ripper/routes/stream_relay.py: 45
- Findings: none
- Noticed: `invariants/buster_ripper/assumption-claims.md:2850` names `_relay_after_commit` in the `CO:client-response-to-relay-consumer/payload-and-header-pairing` `Counterexample:` prose with `messages.py:170-197`; that range was already stale at `HEAD` (the function then sat at :196-213) and now names a file it has left, but no `Subject:`/`Source:`/`Invalidates-On:` line names either moved function, the sync preview is `current`, and the check adds no diagnostic
  Ask: no — documentation prose only, no runtime or write path affected
- Noticed: `codify knowledge check --profile strict` reports 5 pre-existing diagnostics (three `missing-version` records; `## Handle Index` and `## Entity Index` curated-section errors in `invariants/index.md`) — none in a file this unit touched
  Ask: no — pre-existing registry debt outside this unit's files

## U2 Forward
- Moved: nothing
- Reshaped: `handle_messages`: the normalize/session-id/image-strip/preserved-thinking/log-changes block moved verbatim to `_prepare_body(request, body) -> str`, which returns the session id; dump, headers and dispatch stay. `handle_streaming`: the `isinstance(..., web.Response) and status >= 400` probe moved to `_stream_overflow(result)`, which returns the parsed overflow or None; the loop keeps its `continue`/400/`return result` arms. `_stream_single_request`: the identical redirect and error relay blocks became one `if resp.status >= 300:` arm calling `_upstream_answer(resp)`. `_non_stream_single_request`: the `>= 400` probe and its token fallback moved to `_raise_context_overflow(content, input_tokens)`; the usage record and its verbose line moved to `_note_non_stream_usage(request, sid, input_tokens, elapsed_ms)`
- Files: buster_ripper/routes/messages.py
- Tests edited: none
- Ran: `.venv/bin/python -m pytest tests/test_messages_routes.py tests/test_context_overflow.py tests/test_context_overflow_retry.py tests/test_stream_relay_commit.py tests/test_transport_parity.py -o addopts=` -> 60 passed at baseline and after each of rows 1-4
- Ran: brief `Verify` -> 364 passed, 3 pre-existing RuntimeWarnings + 7 passed
- Scan: `buster_ripper/routes/messages.py` — 17 total issues in 8 functions (Medium Complexity 1, Medium Length 5, Medium Nesting 5, Medium Parameters 4); two near tolerance (Complexity and Nesting: `_prepare_body`; Nesting and Parameters: `_note_non_stream_usage`); top: `_stream_single_request` L:25 N:2 C:3 P:4, `_non_stream_single_request` L:24 N:2 C:3 P:4, `handle_streaming` L:23 N:2 C:6 P:4, `handle_non_streaming` L:21 N:2 C:5 P:4, `handle_messages` L:16 N:1 C:5 P:1; no file-size finding
- Sizes: buster_ripper/routes/messages.py: 300
- Noticed: the reshaped functions' metrics all fell (`handle_messages` L:28 N:2 C:12 P:1 -> L:16 N:1 C:5 P:1, `handle_streaming` L:25 N:3 C:8 -> L:23 N:2 C:6, `_stream_single_request` L:35 N:2 C:4 -> L:25 N:2 C:3, `_non_stream_single_request` L:34 N:4 C:8 -> L:24 N:2 C:3), but the extracted blocks carry their nesting into three new functions: `_prepare_body` L:14 N:2 C:8 P:2 (Medium Complexity and Nesting), `_note_non_stream_usage` L:7 N:2 C:3 P:4 (Medium Nesting and Parameters), `_raise_context_overflow` L:5 N:2 C:4 P:2 (Medium Nesting)
  Ask: no — debt that moved with the extracted text; the map's rows end at these extractions and no later unit reshapes them
- Noticed: messages.py ends at exactly 300 lines, the review bound; to stay there without touching a code line the two relay-arm comments were merged into one and `_note_non_stream_usage`'s parameter annotations were dropped
  Ask: no — comment/annotation text only, no behavior or caller changed
- Noticed: `handle_messages` is still printed for Medium Length (L:16) and `handle_non_streaming` is unchanged at L:21 N:2 C:5 P:4; no unit row reshapes either
  Ask: no — pre-existing threshold debt outside the map's rows
- Noticed: tests/test_server_hooks.py's three never-awaited-coroutine RuntimeWarnings are unchanged test noise
  Ask: no — test hygiene only, no user-visible wrong result

## U2 Yield
- Verdict: ACCEPT
- Ran: `git status` + `git diff -M -C --no-ext-diff` -> only `buster_ripper/routes/messages.py` and the brief changed; the file's 92 added and 72 removed lines all sit inside the four reshape rows — no test, invariant or other production file touched
- Ran: `git show HEAD:buster_ripper/routes/messages.py` vs the working tree (AST per top-level function) -> `_inject_preserved_thinking`, `_request_payload` and `handle_non_streaming` unchanged; only `handle_messages`, `handle_streaming`, `_stream_single_request` and `_non_stream_single_request` changed, and the five added names are exactly the map's helpers (`_prepare_body`, `_stream_overflow`, `_upstream_answer`, `_raise_context_overflow`, `_note_non_stream_usage`)
- Ran: block-by-block text compare against `HEAD` -> row 1's block verbatim in `_prepare_body` plus `return sid`; row 2's decode/parse probe verbatim in `_stream_overflow`, with the loop's `continue`/400/`return result` arms textually unchanged; row 3's two identical `web.Response(...)` constructions verbatim in `_upstream_answer`, behind one `if resp.status >= 300:` arm (`300 <= status < 400` plus `status >= 400` is the same set); row 4's overflow probe and token fallback verbatim in `_raise_context_overflow`, the usage record and verbose line verbatim in `_note_non_stream_usage`. `normalize` returns the same dict it mutates, so `_prepare_body`'s rebinding cannot drop the caller's body
- Ran: `.venv/bin/python -m pytest tests/test_messages_routes.py tests/test_chat_routes.py tests/test_messages_request_bytes.py tests/test_chat_request_bytes.py tests/test_stream_relay_commit.py tests/test_header_parity.py tests/test_context_overflow.py tests/test_context_overflow_retry.py tests/test_upstream_answer_relay.py tests/test_upstream_error_relay.py tests/test_transport_parity.py tests/test_preserved_thinking.py tests/test_proxy_errors.py tests/test_image.py tests/test_dump.py tests/test_dump_flush.py tests/test_dump_records.py tests/test_dump_triggers.py tests/test_dump_write_count.py tests/test_server_hooks.py tests/test_compaction.py -o addopts= && .venv/bin/python -m pytest tests/qa/test_qap_ms001.py tests/qa/test_qap_ms002.py -o addopts=` -> 364 passed, 2 pre-existing RuntimeWarnings + 7 passed (Safety net: 364 + 7, 0 skipped)
- Ran: `codify knowledge sync-registry` (preview) -> `current`, no pending sync; `codify knowledge check --profile strict` -> the same 5 pre-existing diagnostics U1 recorded (`invariants/` is untouched by this unit)
- Scan: `codify scan path buster_ripper/routes/messages.py` -> 17 issues in 8 functions (Medium Complexity 1, Medium Length 5, Medium Nesting 5, Medium Parameters 4; near tolerance 2): `_stream_single_request` L:25 N:2 C:3 P:4, `_non_stream_single_request` L:24 N:2 C:3 P:4, `handle_streaming` L:23 N:2 C:6 P:4, `handle_non_streaming` L:21 N:2 C:5 P:4, `handle_messages` L:16 N:1 C:5 P:1; no file-size finding. Every function the Safety net printed improved or held (`handle_non_streaming` unchanged; the other four lower L, N or C), 17 issues against the Safety net's 19, and the only names printed that the Safety net did not are the map's own helpers: `_prepare_body` L:14 N:2 C:8 P:2 (near tolerance Complexity and Nesting), `_note_non_stream_usage` L:7 N:2 C:3 P:4 (near tolerance Nesting and Parameters), `_raise_context_overflow` L:5 N:2 C:4 P:2; `_stream_overflow` and `_upstream_answer` are clean
- Sizes: buster_ripper/routes/messages.py: 300
- Findings: none
- Noticed: `messages.py` ends at exactly 300 lines, the review bound; the two relay-arm comments were merged into one and `_note_non_stream_usage`'s parameter annotations were dropped to stay there without touching a code line
  Ask: no — comment and annotation text only, no behavior or caller changed
- Noticed: the three map-named helpers the scan prints carry the nesting of the blocks they came from; the map's rows end at these extractions and no later unit reshapes them
  Ask: no — debt that moved with the extracted text, and the map has no further row
- Noticed: the `handle_messages/dump-failure-isolation` assumption record's `Source: buster_ripper/routes/messages.py:55-59` no longer matches the dump block, now at `:37-41`; the range was already stale at `HEAD` (`:57-61`, after U0's comment shrink), the `Subject:` path is unchanged, and the sync preview stays `current`
  Ask: no — a stale line reference in prose, no runtime or write-path effect
- Noticed: `codify knowledge check --profile strict` reports 5 pre-existing diagnostics (three `missing-version` records; `## Handle Index` and `## Entity Index` curated-section errors in `invariants/index.md`) — none in a file this unit touched
  Ask: no — pre-existing registry debt outside this unit's files
