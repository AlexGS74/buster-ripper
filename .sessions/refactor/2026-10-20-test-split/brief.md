# Split the header-parity helpers and migrate the routes-integration tests

- Goal: split `tests/test_header_parity.py`'s helper/census code out of the parity tests, and migrate `tests/test_routes_integration.py` onto the shared `tests/upstream_stub.py` + `conftest.py::proxy` fixtures, with behavior unchanged.
- Target: roadmap `RM-017`, scoped to `tests/test_header_parity.py` and `tests/test_routes_integration.py`.
- In scope: `tests/test_header_parity.py`, `tests/test_routes_integration.py`
- Importers: `tests/test_header_parity.py`, `tests/test_header_helpers.py`
- Out of scope: the other six modules named by `RM-017`; they will be minted as a new roadmap row.
- Note: `tests/conftest.py` is taken by `origin/fix/tool-reference-scrub` (stale since 2026-09-17); its `proxy` fixture is used read-only, no edit planned.
- Verify: `.venv/bin/python -m pytest tests/test_header_parity.py tests/test_header_helpers.py tests/test_messages_routes.py tests/test_chat_routes.py tests/test_count_tokens_routes.py tests/test_passthrough_routes.py -o addopts=`
- Final verify: `uv run pytest`
- Worktree: /Users/alexhubitski/Worktrees/buster-ripper/refactor-test-split

## Collected
| # | Candidate | Why | Recorded as | State |
|---|---|---|---|---|
| 1 | tests/test_routes_integration.py | 521 lines ("File is 521 lines (ideal: <=500)"; RM-017 records 495 at HEAD 54001b6, +26 in `9f23018`); "Summary: 14 total issues found in 14 functions" — Medium Length: 5, Medium Nesting: 1, Medium Parameters: 7; top `test_chat_endpoint_accepts_json` L:21 N:0 C:1 P:3; its four longest tests call `_create_mock_response` L:14 N:1 C:4 P:5 ("Medium Parameters", 12 call sites) and a local `client` fixture — the duplicate pair RM-017 names; ahead of RM-017 · migrate onto `tests/upstream_stub.py` + `conftest.py::proxy` and delete the local pair (`tests/conftest.py` itself is taken by `origin/fix/tool-reference-scrub`, stale since 2026-09-17; read-only, no edit planned) | RM-017 (`QB:test-routes/duplicate-upstream-stub`, `QB:test-routes/file-size`) | free |
| 2 | tests/test_header_parity.py | 444 lines ("File is 444 lines (review: <=300)"); "Summary: 14 total issues found in 14 functions" — Medium Parameters: 13, "Issues Near Tolerance (with only two medium severity factors): Length and Parameters: 1"; top `test_duplicate_spellings_never_lose_the_credential` L:19 N:1 C:5 P:5 ("Near Tolerance Length Parameters"); ahead of RM-017 · helpers/census split (`TestSharedHelpers`/`TestSiteCensus` out to a helper-level module) | RM-017 (`QB:test-header-parity/file-size`) | free |
- Tool gaps: `quality-backlog.md` is absent from the tree (deleted by the 02.00 squash `3323551`), so RM-017's `QB:` rows were read from `git show 191e676:.sessions/codify/02.00-2026-09-13-message-dump-bulk-flush--9bf7afff5573/quality-backlog.md` §Refactoring. None of the six CODIFY sessions whose Progress Overview shows implementation not started carries an Anchor Map (the only session with one, 02.00, is complete and squashed), so no `planned:` files could be derived.

## Safety net
- Ran: `.venv/bin/python -m pytest tests/test_header_parity.py tests/test_routes_integration.py -o addopts=` -> 121 passed, 0 skipped (the worktree shipped no `.venv`; the same command under `uv run python -m pytest ...` created it, then the brief's exact command was re-run)
- Scan: `tests/test_header_parity.py — File is 444 lines (review: <=300); Summary: 14 total issues found in 14 functions — Medium Parameters: 13; Issues Near Tolerance: Length and Parameters: 1 (top: test_duplicate_spellings_never_lose_the_credential L:19 N:1 C:5 P:5)`. `tests/test_routes_integration.py — File is 521 lines (ideal: <=500); Summary: 14 total issues found in 14 functions — Medium Length: 5, Medium Nesting: 1, Medium Parameters: 7; Issues Near Tolerance: Length and Parameters: 1 (top: test_chat_endpoint_accepts_json L:21 N:0 C:1 P:3)`
- Sizes: tests/test_header_parity.py: 444; tests/test_routes_integration.py: 521
- Public paths kept: none

## Map

### U0 · comments
| # | Kind | What | From | To |
|---|---|---|---|---|
| 1 | shrink | module docstring (21 lines) and `TestRepeatedHeaderOccurrences` class docstring (13 lines) to their point | tests/test_header_parity.py | none — the base-parity contract stays in the two shrunk docstrings |
| 2 | shrink | docstrings of `_create_mock_response` (7 lines), `test_count_tokens_returns_token_count` (4), `test_a_billing_nonce_body_finds_the_stats_messages_recorded` (4), `test_count_tokens_inflates_on_compaction_trigger` (8), `test_chat_nonstream_compatible_json_transforms` (4) to their point | tests/test_routes_integration.py | `.sessions/codify/02.00-2026-09-13-message-dump-bulk-flush--9bf7afff5573/chronicle.md` already records CR3-F-02 and 066-01-yield P1 |
- Narrow tests: tests/test_header_parity.py, tests/test_routes_integration.py

### U1 · tests/test_header_parity.py: split the site tables, wire helpers and helper/census tests into a support module and a helper test module
| # | Kind | What | From | To |
|---|---|---|---|---|
| 1 | move | `SIX_NAMES`, `ALL_NAMES`, `END_TO_END`, `_CHAT_BODY`, `_MESSAGES_BODY`, `_CHAT_JSON`, `INBOUND_MARKER`, `UPSTREAM_MARKER`, `CHAT_SITE`, `MESSAGES_SITE`, `PASSTHROUGH_SITE`, `EGRESS_SITES`, `INGRESS_SITES`, `NON_CHAT_SITES`, `_REAL_TOKEN`, `_EMPTY_SPELLINGS`, `_BLANK`, `_REPEAT`, `_REPEAT_PAIR` | tests/test_header_parity.py | tests/header_sites.py |
| 2 | move | `_repeated_head`, `_values`, `_wire_headers`, `_spy_helpers`, `_egress_capture`, `_ingress_relay` | tests/test_header_parity.py | tests/header_sites.py |
| 3 | move | `TestSharedHelpers`, `TestSiteCensus` | tests/test_header_parity.py | tests/test_header_helpers.py |
| 4 | reshape | replace the moved blocks with `from tests.header_sites import (...)`; drop `json`, `Path`, `messages_routes`, `passthrough_routes`, `forward_headers`, `_HOP_BY_HOP`, which no longer have a use | tests/test_header_parity.py | tests/test_header_parity.py |
- Importers to update: tests/test_header_parity.py imports the moved names from `tests.header_sites`; tests/test_header_helpers.py is new and imports `_values` from `tests.header_sites`, and `forward_headers`, `response_headers`, `_HOP_BY_HOP` from `buster_ripper.utils`
- Witnesses to re-point: none (no invariant record names tests/test_header_parity.py)
- Tests with import lines or patch targets to update: tests/test_header_parity.py (new `tests.header_sites` import), tests/test_header_helpers.py (new file, imports `_values` from `tests.header_sites`)
- Narrow tests: tests/test_header_parity.py, tests/test_header_helpers.py
- Sizes after the unit: tests/test_header_parity.py ≈ 271, tests/header_sites.py ≈ 112, tests/test_header_helpers.py ≈ 56; `tests/test_upstream_answer_relay.py:10`'s "`test_header_parity.py`'s ingress sites" stays true (the ingress sites stay put)

### U2 · tests/test_routes_integration.py: migrate onto `StubUpstream` + `proxy` and split by endpoint
| # | Kind | What | From | To |
|---|---|---|---|---|
| 1 | move | `_MESSAGES_BODY`, `TestMessagesRoute`, `TestSessionPooling`, `TestDumpFailureIsolated`, `TestErrorHandling::test_messages_handles_upstream_error` | tests/test_routes_integration.py | tests/test_messages_routes.py |
| 2 | move | `TestChatRoute`, `TestErrorHandling::test_chat_handles_upstream_error` | tests/test_routes_integration.py | tests/test_chat_routes.py |
| 3 | move | `_sid`, `TestCountTokensRoute` | tests/test_routes_integration.py | tests/test_count_tokens_routes.py |
| 4 | move | `TestPassthroughRoute` | tests/test_routes_integration.py | tests/test_passthrough_routes.py |
| 5 | reshape | delete `_create_mock_response`; every call becomes `StubUpstream`: `(mocker, json_data=D, headers=H)` -> `StubUpstream(body=json.dumps(D).encode(), headers=H)`, `(mocker, status=S, headers=H)` -> `StubUpstream(status=S, headers=H)`, `(mocker, status=S)` -> `StubUpstream(status=S)`; its unused `text_data` arm goes with it | tests/test_routes_integration.py | the four new modules |
| 6 | reshape | delete the local `client` fixture; the tests take `proxy` from `tests/conftest.py`; `client.server.app` -> `proxy.server.app`, `client.post/get` -> `proxy.post/get`; drop the `create_app`/`TestClient`/`TestServer` imports | tests/test_routes_integration.py | the four new modules |
| 7 | reshape | mock await pins become spy await counts: `upstream.read.assert_awaited_once()` -> `mocker.spy(upstream, "read")` + `assert read_spy.await_count == 1`; `upstream.json.assert_not_awaited()` -> `mocker.spy(upstream, "json")` + `assert json_spy.await_count == 0` | tests/test_routes_integration.py | tests/test_chat_routes.py |
| 8 | reshape | `upstream.content.iter_any = chunks` -> `StubStream(headers=H, chunks=(payload[:4], payload[4:]))`; `upstream.read.return_value = payload` -> `StubUpstream(..., body=payload)`; keep the declared upstream bytes with `body=json.dumps(D).encode()` where `_create_mock_response` derived them from `json_data` | tests/test_routes_integration.py | tests/test_chat_routes.py |
- Importers to update: none import the module; `tests/test_upstream_answer_relay.py:213` names a witness (`test_chat_upstream_undecodable_json_is_upstream_error`) that this file no longer holds — a historical note, not a live import or patch target, left as written
- Witnesses to re-point: `invariants/buster_ripper/assumption-claims.md:433-434` (claim `CL:tests/upstream-mocking`) -> `tests/test_messages_routes.py`, `tests/test_chat_routes.py`, `tests/test_count_tokens_routes.py`, `tests/test_passthrough_routes.py`, with the cited line spans re-derived in the same commit; `:438-439` (`Invalidates-On`) names the same paths — its "the integration suite moves to a different module" clause fires on this move, so that claim needs re-verification by the assumption lane, not a path edit; `.sessions/codify/roadmap/2026-10-18-message-dump-bulk-flush.md:51` (RM-017 scope) -> the four new paths, 495 -> their sizes
- Tests with import lines or patch targets to update: the four new modules (fresh imports of `json`, `pytest`, `StubUpstream`/`StubStream`, `backend`, `config`, `normalize`, `session`, `session_id`, `compaction`); `tests/test_upstream_answer_relay.py` unchanged (docstring only)
- Narrow tests: tests/test_messages_routes.py, tests/test_chat_routes.py, tests/test_count_tokens_routes.py, tests/test_passthrough_routes.py
- Note: the local fixture patched `get_backend_state` -> `"sglang"`, `tests/conftest.py::proxy` patches it -> `"vllm"`; no migrated assertion reads that value, and the one test whose subject is the backend state (`test_chat_endpoint_rejects_a_non_object_json_root`) keeps its own per-parameter patch. `config.STRIP_IMAGES` is additionally pinned False by `proxy`; no migrated test sends an image.
- Sizes after the unit: tests/test_messages_routes.py ≈ 173, tests/test_chat_routes.py ≈ 166, tests/test_count_tokens_routes.py ≈ 116, tests/test_passthrough_routes.py ≈ 28; tests/test_routes_integration.py is gone (its classes moved whole)

## U1 Forward
- Moved: `SIX_NAMES`, `ALL_NAMES`, `END_TO_END`, `_CHAT_BODY`, `_MESSAGES_BODY`, `_CHAT_JSON`, `INBOUND_MARKER`, `UPSTREAM_MARKER`, `CHAT_SITE`, `MESSAGES_SITE`, `PASSTHROUGH_SITE`, `EGRESS_SITES`, `INGRESS_SITES`, `NON_CHAT_SITES`, `_REAL_TOKEN`, `_EMPTY_SPELLINGS`, `_BLANK`, `_REPEAT`, `_REPEAT_PAIR` -> tests/header_sites.py
- Moved: `_repeated_head`, `_values`, `_wire_headers`, `_spy_helpers`, `_egress_capture`, `_ingress_relay` -> tests/header_sites.py
- Moved: `TestSharedHelpers`, `TestSiteCensus` -> tests/test_header_helpers.py
- Reshaped: nothing
- Files: tests/test_header_parity.py, tests/header_sites.py (new), tests/test_header_helpers.py (new)
- Tests edited: tests/test_header_parity.py: new `from tests.header_sites import (...)` block; `json`, `Path`, `messages_routes`, `passthrough_routes`, `_HOP_BY_HOP`, `forward_headers` imports dropped; tests/test_header_helpers.py: new file importing `ALL_NAMES`, `_values` from `tests.header_sites` and `_HOP_BY_HOP`, `forward_headers`, `response_headers` from `buster_ripper.utils`
- Ran: `.venv/bin/python -m pytest tests/test_header_parity.py -o addopts=` -> 81 passed, 0 skipped (steps 1-2); `.venv/bin/python -m pytest tests/test_header_parity.py tests/test_header_helpers.py -o addopts=` -> 81 passed, 0 skipped (steps 3-4)
- Ran: `.venv/bin/python -m pytest tests/test_header_parity.py tests/test_routes_integration.py -o addopts=` -> 99 passed, 0 skipped (the 22 moved tests now collect from tests/test_header_helpers.py; all three files together: 121 passed)
- Ran: `uv run pytest` -> 575 passed, 0 skipped (coverage 96.83%)
- Scan: `tests/test_header_parity.py — Summary: 12 total issues found in 12 functions — Medium Parameters: 11; Issues Near Tolerance: Length and Parameters: 1 (top: test_duplicate_spellings_never_lose_the_credential L:19 N:1 C:5 P:5)`. `tests/header_sites.py — Summary: 2 total issues found in 2 functions — Medium Parameters: 2 (top: _egress_capture L:9 N:0 C:1 P:4; _ingress_relay L:6 N:0 C:1 P:4)`. `tests/test_header_helpers.py — No refactoring issues found!`
- Sizes: tests/test_header_parity.py: 297; tests/header_sites.py: 113; tests/test_header_helpers.py: 56
- Noticed: tests/test_header_parity.py ends at 297 against the map's ≈271; the 26-line difference is the 27-line `from tests.header_sites import (...)` block that row 4 itself mandates, and the file is now under its 300 review threshold.
  Ask: no — shortening the import would depart from row 4 and trim, not move.
- Noticed: `tests/header_sites.py`'s `_egress_capture` (P:4) and `_ingress_relay` (P:4) are printed by the scan; their text only moved from tests/test_header_parity.py and no reshape row covers them.
  Ask: no — pre-existing debt the map did not schedule.
- Noticed: the brief's Verify command now collects 99 rather than the Safety net's 121, because the 22 helper/census tests collect from the new tests/test_header_helpers.py; all three files together are 121.
  Ask: no — the map's own split moves them; the narrow set covers the new file.

## U1 Yield
- Verdict: ACCEPT
- Ran: `.venv/bin/python -m pytest tests/test_header_parity.py tests/test_routes_integration.py -o addopts=` -> 99 passed, 0 skipped (the 22 moved tests collect from tests/test_header_helpers.py); `.venv/bin/python -m pytest tests/test_header_parity.py tests/test_header_helpers.py tests/test_routes_integration.py -o addopts=` -> 121 passed, 0 skipped, matching the Safety net's 121
- Scan: `tests/test_header_parity.py — Summary: 12 total issues found in 12 functions — Medium Parameters: 11; Issues Near Tolerance: Length and Parameters: 1 (top: test_duplicate_spellings_never_lose_the_credential L:19 N:1 C:5 P:5)`. `tests/header_sites.py — Summary: 2 total issues found in 2 functions — Medium Parameters: 2 (top: _egress_capture L:9 N:0 C:1 P:4; _ingress_relay L:6 N:0 C:1 P:4)`. `tests/test_header_helpers.py — No refactoring issues found!` — the HEAD scan of tests/test_header_parity.py printed the same `_egress_capture` L:9 N:0 C:1 P:4 and `_ingress_relay` L:6 N:0 C:1 P:4 inside its 14, so 12 + 2 accounts for all 14 and none got worse
- Sizes: tests/test_header_parity.py: 297, tests/header_sites.py: 113, tests/test_header_helpers.py: 56
- Findings: none
- Noticed: tests/test_header_parity.py ends at 297 against the map's ≈271; the 26-line difference is the 27-line `from tests.header_sites import (...)` block that row 4 itself mandates, and the file is now under its 300 review threshold
  Ask: no — shortening the import would depart from row 4 and trim, not move
- Noticed: tests/header_sites.py's `_egress_capture` (P:4) and `_ingress_relay` (P:4) are printed by the scan; both were printed at HEAD with identical metrics, so this is pre-existing debt no reshape row covers
  Ask: no — pre-existing debt the map did not schedule
- Noticed: the brief's Verify command collects 99 rather than the Safety net's 121 because the 22 helper/census tests collect from the new tests/test_header_helpers.py; all three files together are 121
  Ask: no — the map's own split moves them and the unit's narrow set covers the new file

## U0 Forward
- Moved: nothing
- Reshaped: nothing
- Files: tests/test_header_parity.py, tests/test_routes_integration.py
- Tests edited: none
- Ran: `.venv/bin/python -m pytest tests/test_header_parity.py tests/test_routes_integration.py -o addopts=` -> 121 passed, 0 skipped
- Scan: `tests/test_header_parity.py — File is 430 lines (review: <=300); Summary: 14 total issues found in 14 functions — Medium Parameters: 13; Issues Near Tolerance: Length and Parameters: 1 (top: test_duplicate_spellings_never_lose_the_credential L:19 N:1 C:5 P:5)`. `tests/test_routes_integration.py — File is 511 lines (ideal: <=500); Summary: 14 total issues found in 14 functions — Medium Length: 5, Medium Nesting: 1, Medium Parameters: 7; Issues Near Tolerance: Length and Parameters: 1 (top: test_chat_endpoint_accepts_json L:21 N:0 C:1 P:3)`
- Sizes: tests/test_header_parity.py: 430; tests/test_routes_integration.py: 511
- Noticed: `tests/test_routes_integration.py` remains 511 lines (ideal <=500); U2's split brings it under.
  Ask: no — U2's map rows move the classes out and delete the duplicate helper pair

## U0 Yield
- Verdict: ACCEPT
- Ran: `.venv/bin/python -m pytest tests/test_header_parity.py tests/test_routes_integration.py -o addopts=` -> 121 passed, 0 skipped
- Scan: `tests/test_header_parity.py — File is 430 lines (review: <=300); Summary: 14 total issues found in 14 functions — Medium Parameters: 13; Issues Near Tolerance: Length and Parameters: 1 (top: test_duplicate_spellings_never_lose_the_credential L:19 N:1 C:5 P:5)`. `tests/test_routes_integration.py — File is 511 lines (ideal: <=500); Summary: 14 total issues found in 14 functions — Medium Length: 5, Medium Nesting: 1, Medium Parameters: 7; Issues Near Tolerance: Length and Parameters: 1 (top: test_chat_endpoint_accepts_json L:21 N:0 C:1 P:3)` — same issue counts as the Safety net, both files smaller
- Sizes: tests/test_header_parity.py: 430, tests/test_routes_integration.py: 511
- Findings: none
- Noticed: map row 2 calls the `test_chat_nonstream_compatible_json_transforms` edit a docstring; the 4 shrunk lines there are a `#` comment, which the shrink rule allows and the row names the test — no action
  Ask: no — a description wording fix in an already-committed map row changes nothing in the code or the test outcome

## U2 Forward
- Moved: `_MESSAGES_BODY`, `TestMessagesRoute`, `TestSessionPooling`, `TestDumpFailureIsolated`, `TestErrorHandling::test_messages_handles_upstream_error` -> tests/test_messages_routes.py
- Moved: `TestChatRoute`, `TestErrorHandling::test_chat_handles_upstream_error` -> tests/test_chat_routes.py
- Moved: `_sid`, `TestCountTokensRoute` -> tests/test_count_tokens_routes.py
- Moved: `TestPassthroughRoute` -> tests/test_passthrough_routes.py
- Reshaped: `_create_mock_response`: deleted; `json_data=D, headers=H` -> `StubUpstream(body=json.dumps(D).encode(), headers=H)`, `status=S, headers=H` -> `StubUpstream(status=S, headers=H)`, `status=S` -> `StubUpstream(status=S)`, `read.return_value = payload` -> `StubUpstream(..., body=payload)`, `content.iter_any = chunks` -> `StubStream(headers=H, chunks=(payload[:4], payload[4:]))`; unused `text_data` arm gone
- Reshaped: `client` fixture: deleted; every test takes `proxy` from tests/conftest.py; `client.server.app` -> `proxy.server.app`, `client.post/get` -> `proxy.post/get`; `create_app`/`TestClient`/`TestServer` imports dropped
- Reshaped: `upstream.read.assert_awaited_once()` -> `mocker.spy(upstream, "read")` + `assert read_spy.await_count == 1`; `upstream.json.assert_not_awaited()` -> `mocker.spy(upstream, "json")` + `assert json_spy.await_count == 0` (tests/test_chat_routes.py)
- Files: tests/test_messages_routes.py (new), tests/test_chat_routes.py (new), tests/test_count_tokens_routes.py (new), tests/test_passthrough_routes.py (new), tests/test_routes_integration.py (deleted), invariants/buster_ripper/assumption-claims.md, .sessions/codify/roadmap/2026-10-18-message-dump-bulk-flush.md
- Tests edited: tests/test_messages_routes.py, tests/test_chat_routes.py, tests/test_count_tokens_routes.py, tests/test_passthrough_routes.py: fresh imports of `json`, `pytest`, `StubUpstream`/`StubStream`, `backend`, `config`, `normalize`, `session_id`; tests/test_upstream_answer_relay.py unchanged (its `:213` note names the old module as history, left as written)
- Ran: `.venv/bin/python -m pytest tests/test_header_parity.py tests/test_routes_integration.py -o addopts=` -> `ERROR: file or directory not found: tests/test_routes_integration.py` (0 collected; the map deletes the module)
- Ran: `.venv/bin/python -m pytest tests/test_header_parity.py tests/test_messages_routes.py tests/test_chat_routes.py tests/test_count_tokens_routes.py tests/test_passthrough_routes.py -o addopts=` -> 99 passed, 0 skipped
- Ran: `.venv/bin/python -m pytest tests/test_messages_routes.py tests/test_chat_routes.py tests/test_count_tokens_routes.py tests/test_passthrough_routes.py -o addopts=` -> 40 passed, 0 skipped (same 40 the Safety net counted in the single file)
- Ran: `uv run pytest` -> 575 passed, 22 warnings (coverage 96.83%)
- Scan: `tests/test_messages_routes.py — 4 total issues found in 4 functions — Medium Length: 2, Medium Parameters: 2 (test_messages_endpoint_normalizes_request L:20 N:0 C:1 P:3; test_messages_endpoint_accepts_json L:19 N:0 C:1 P:3; test_raising_dump_still_yields_upstream_response L:13 N:0 C:1 P:4; test_messages_endpoint_rejects_undecodable_json L:7 N:0 C:1 P:4)`. `tests/test_chat_routes.py — 6 total issues found in 6 functions — Medium Length: 2, Medium Parameters: 4 (top: test_chat_endpoint_accepts_json L:22 N:0 C:1 P:3)`. `tests/test_count_tokens_routes.py — 1 total issue found in 1 function — Issues Near Tolerance: Length and Parameters (test_count_tokens_inflates_on_compaction_trigger L:16 N:0 C:1 P:5)`. `tests/test_passthrough_routes.py — No refactoring issues found!`
- Sizes: tests/test_messages_routes.py: 177; tests/test_chat_routes.py: 167; tests/test_count_tokens_routes.py: 113; tests/test_passthrough_routes.py: 26
- Noticed: the brief's Verify command names tests/test_routes_integration.py, which the map deletes; pytest reports it as not found and collects 0. The same tests run from the four new modules plus tests/test_header_parity.py: 99 passed.
  Ask: no — the map's own split deletes the module and the narrow set covers the four new paths.
- Noticed: the scan prints 11 issues in 11 functions across the four modules against HEAD's 14 in 14 in the single file (Medium Length 5 -> 4, Medium Nesting 1 -> 0, Medium Parameters 7 -> 6); every printed function's text only moved, except three that grew one line from row 5's `json.dumps(D).encode()` wrapping (`test_chat_endpoint_accepts_json` 21 -> 22, `test_messages_endpoint_accepts_json` 18 -> 19, `test_messages_endpoint_normalizes_request` 19 -> 20).
  Ask: no — the one-line growth is the map's own reshape and the aggregate shape improved.
- Noticed: `invariants/buster_ripper/assumption-claims.md:436-437` still carries the pre-move fingerprints, which now hash a tree the move changed; the map routes `:438-439`'s fired clause to the assumption lane and does not name the fingerprint lines.
  Ask: no — the record's own `Invalidates-On` clause fires on this move and tells the assumption lane to re-verify.
- Noticed: `tests/test_upstream_answer_relay.py:213` still names `test_routes_integration.py` in a historical docstring; the map says leave it as written.
  Ask: no — it is a history note, not a live import or patch target.
- Noticed: the local fixture pinned `backend.get_backend_state` to "sglang" and `proxy` pins "vllm"; no migrated assertion reads that value, and `test_chat_endpoint_rejects_a_non_object_json_root` keeps its own per-parameter patch.
  Ask: no — the map's Note covers it and no test outcome changed.

## U2 Yield
- Verdict: ACCEPT
- Ran: `.venv/bin/python -m pytest tests/test_header_parity.py tests/test_routes_integration.py -o addopts=` -> `ERROR: file or directory not found: tests/test_routes_integration.py`, 0 collected (the map deletes the module); `.venv/bin/python -m pytest tests/test_header_parity.py tests/test_header_helpers.py tests/test_messages_routes.py tests/test_chat_routes.py tests/test_count_tokens_routes.py tests/test_passthrough_routes.py -o addopts=` -> 121 passed, 0 skipped, matching the Safety net's 121
- Scan: `tests/test_messages_routes.py — 4 total issues found in 4 functions — Medium Length: 2, Medium Parameters: 2 (test_messages_endpoint_normalizes_request L:20 N:0 C:1 P:3; test_messages_endpoint_accepts_json L:19 N:0 C:1 P:3; test_raising_dump_still_yields_upstream_response L:13 N:0 C:1 P:4; test_messages_endpoint_rejects_undecodable_json L:7 N:0 C:1 P:4)`. `tests/test_chat_routes.py — 6 total issues found in 6 functions — Medium Length: 2, Medium Parameters: 4 (top: test_chat_endpoint_accepts_json L:22 N:0 C:1 P:3)`. `tests/test_count_tokens_routes.py — 1 total issue found in 1 function — Issues Near Tolerance: Length and Parameters (test_count_tokens_inflates_on_compaction_trigger L:16 N:0 C:1 P:5)`. `tests/test_passthrough_routes.py — No refactoring issues found!` — 11 issues in 11 functions against the Safety net's 14 in 14 (Medium Length 5 -> 4, Medium Nesting 1 -> 0, Medium Parameters 7 -> 6); the deleted `_create_mock_response` printed L:14 N:1 C:4 P:5, accounting for one Length, the only Nesting and one Parameters
- Sizes: tests/test_messages_routes.py: 177; tests/test_chat_routes.py: 167; tests/test_count_tokens_routes.py: 113; tests/test_passthrough_routes.py: 26
- Findings: none
- Noticed: `test_chat_endpoint_accepts_json` is printed at L:22 against the Safety net's L:21, and `test_messages_endpoint_accepts_json` / `test_messages_endpoint_normalizes_request` likewise gained one line, because row 5's `body=json.dumps(D).encode()` was wrapped across three lines; the same reshape on one line would keep the old counts.
  Ask: no — this is the map's own reshape and a formatting choice, and the aggregate went 14 issues to 11.
- Noticed: tests/test_messages_routes.py ends at 177 against the map's ≈173 and tests/test_chat_routes.py at 167 against ≈166; the excess is the same `json.dumps(...).encode()` wrapping, and all four files are far under 500.
  Ask: no — the map's sizes are approximate and the split is what the unit was for.
- Noticed: `TestErrorHandling` was dissolved rather than duplicated, so `test_messages_handles_upstream_error` now sits in `TestDumpFailureIsolated` (tests/test_messages_routes.py:157, whose docstring names VP-T001) and `test_chat_handles_upstream_error` in `TestChatRoute` (tests/test_chat_routes.py:144); their bodies and assertions are unchanged and their node ids move with them.
  Ask: no — the class header is a layout choice no live gate or record keys on; `TestErrorHandling` has no reference outside `.sessions/`.
- Noticed: the invariant's `Actual`/`Source` lines were rewritten to describe the four modules and their `StubUpstream`/`StubStream`/`proxy` homes, adding `tests/conftest.py:36-50` and `tests/upstream_stub.py:38-75` as corroborating citations rather than only re-pointing the four module paths the map named.
  Ask: no — the claim's proposition is unchanged and the record now points at real code instead of `_create_mock_response`.
- Noticed: `invariants/buster_ripper/assumption-claims.md:438` still says `tests/test_routes_integration.py no longer patches ...`, a module this unit deletes; the map sends that fired clause to the assumption lane, not to a path edit.
  Ask: no — the record's own `Invalidates-On` clause fires and re-verification is the assumption lane's job.
- Noticed: the RM-017 action cell still says "migrate `test_routes_integration.py` onto ..." though the module is gone, and its target cell still lists `tests/test_header_parity.py` (444), stale since U1; the map only authorized the RM-017 path list and sizes.
  Ask: no — a roadmap row is a historical record and the map named only the path list and the 495 -> four sizes edit.
