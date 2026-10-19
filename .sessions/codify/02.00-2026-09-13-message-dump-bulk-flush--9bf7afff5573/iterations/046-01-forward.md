## Orchestrator Prompt (verbatim)

> You are a subagent spawned by another session.
> Session ID: 02.00-2026-09-13-message-dump-bulk-flush--9bf7afff5573
> Task ID: T008.001

## Increment Brief
- T008.001: preserve the upstream media type along with status and raw bytes on chat stream errors.
- Anchor touched: chat stream relay → `buster_ripper/routes/chat.py:_relay_chat_stream` → extend (T008 handoff binding).
- Will NOT create: another relay, parser, or changes to messages peer; reuse the existing stream branch.
- Reuse evidence: no NEW anchor touched. Inventions: none.

## Failure-Path Evidence
| Operation / condition | Expected outcome and authority | Handling / observation evidence | Disposition |
|---|---|---|---|
| Stream error with non-JSON Content-Type | Preserve status, bytes, and declaration; T008.001 context | Parameterized 429 plain-text and problem+json route tests assert all three plus no `json()`; original failure reproduced | safe |
| Stream error without Content-Type | Do not assert JSON without upstream declaration; S9 | Parameterized absent-header regression asserts aiohttp's octet-stream fallback rather than JSON | safe |
| Successful SSE and nonstream compatible +json | Preserve current relay and transformations; T008.001 context | Existing focused `TestChatRoute` SSE and +json tests: 14 passed | safe |
| Transport failure before response | Existing 502 client error path; no error body to relay | Unchanged `_make_chat_request` catch | not-applicable |
| Cancellation, downstream disconnect, retries | No new lifecycle, stream-writing or retry policy introduced by header change | Existing relay and caller unchanged | not-applicable |

Capacity gate: not applicable - no additive production surface (replace existing constant header in `_relay_chat_stream`).
Backlog pickup: none match touch-set for this narrow replacement; QB:routes-chat/handler-lncp and QB:routes-chat/request-lncp concern untouched functions; QB:test-routes/file-size remains yellow and below 500 after focused regression.

## Implementation and validation
- `buster_ripper/routes/chat.py`: error branch forwards upstream Content-Type verbatim when present via explicit response header; omits it when absent, allowing aiohttp's byte-body default. No JSON decode, body transform, or changes to successful stream and nonstream paths.
- `tests/test_routes_integration.py`: parameterized error regression for `text/plain; charset=utf-8`, `application/problem+json`, and no upstream declaration; asserts 429, exact raw bytes, media type and no upstream JSON call. Fail-first: all three failed on the old forced `application/json`; targeted chat class then 14 passed after replacement (`-o addopts=''` to avoid irrelevant global coverage threshold during narrow run).
- Source contract check: installed aiohttp `web_response.py:628-702` accepts a Content-Type header without `content_type=`, and for an absent header and byte body uses its native default; `ClientResponse.headers` exposes the upstream declaration. S9 is scoped to this task, not a shared claim requiring cross-task routing. No silent plan decision change.
- Task VP handles: T008.001 has a handoff-specific S9 regression rather than a new VP; MS-002's existing QAP-MS002 gates and re-verdict remain separate.
- LNCP: no new production functions; `_relay_chat_stream` is a one-expression substitution, `codify scan path buster_ripper/routes/chat.py --git --detailed` clean. Changed test L11/N0/C2/P4 (one gray metric); raw file 279 → 280 lines; test file 424 → 433 (yellow, below 500). Test-file scan reports only this single-gray function and the already-yellow file size.
- Coverage: route module is excluded by `pyproject.toml`; all new production behavior is exercised by three route tests, with all chat branch regressions passing. Full suite `uv run pytest`: 250 passed, 23 pre-existing warnings, 93.87% total coverage (threshold 80%).
- Demo evidence: [terminal capture of real aiohttp test server](../evidence/t008-001-chat-error.txt), indexed in [evidence/INDEX.md](../evidence/INDEX.md); no pause.
- Assumption impact: scoped:S9, implementation submitted to Yield; do not mark S9 resolved until a fresh SAFE composition. No invariant edits or override entries.
- Questions asked: none. Next: Yield replays error/SSE/nonstream branches, then S9 recomposition and MS-002 process QA.
