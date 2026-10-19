## Orchestrator Prompt (verbatim)

> You are a subagent spawned by another session.
> Session ID: 02.00-2026-09-13-message-dump-bulk-flush--9bf7afff5573
> Task ID: T003.001

## Increment Brief
- T003.001: return typed client errors for undecodable JSON bytes across three request routes while retaining syntax fallback and upstream attribution.
- Anchors touched: dump caller → `routes/messages.py:handle_messages` → reuse (parse boundary); existing `routes/chat.py:handle_chat` and `routes/count_tokens.py:handle_count_tokens` from task handoff; no new referent.
- Will NOT create: shared parser, endpoint, or QA gate; preserve existing handlers and upstream response parsing.
- Reuse evidence: no touched NEW anchor; existing three route parse boundaries own handling.
- Inventions: none.

## Failure-Path Evidence

| Operation / condition | Expected outcome and authority | Handling / observation evidence | Disposition |
|---|---|---|---|
| Client JSON, invalid UTF-8 | Typed 4xx on all three client routes (T003.001/S7) | Nine fail-first byte/route cases in `test_routes_integration.py` went 500→400, upstream spy not called for messages/chat | safe |
| Client JSON, invalid syntax after valid UTF-8 | Messages/chat 400; count_tokens empty-body fallback (handoff S7) | Existing syntax tests plus new count_tokens `{` fallback (200) | safe |
| Upstream chat JSON decode failure | 502, not client 400 (handoff S7) | Fail-first upstream UnicodeDecodeError case 500→502; distinct `resp.json()` handler | safe |
| Valid JSON | Existing dispatch and response behavior | Existing route integration suite: 24/24 focused | safe |
| Dependency failure, cancellation, retry, duplicate, recovery | No new dependency/state/lifecycle introduced by parse exception handling; existing route behavior unchanged | Existing handlers | not-applicable |

## Capacity gate
- Additive surfaces: three existing route parse exception handlers. Pre-edit files: messages.py 268, chat.py 260, count_tokens.py 76 lines. `codify scan path --detailed` baseline: `handle_messages` L31/N2/C11/P1; `handle_chat` L29/N2/C6/P1; `handle_count_tokens` L42/N1/C8/P1. First inline count_tokens fix worsened L/C to 44/9; extracted its existing parse-boundary block into module-private `_read_body` (same existing concept, no new component), reducing the handler to L39/N1/C7. Focused route suite passed 24/24 before and after extraction. Final ratchet verified by `--git` scan; no other touched function metric worsened.

Backlog pickup: none match touch-set.

## Implementation and evidence
- `messages.py` / `chat.py` catch only JSON syntax and UTF-8 decoding failures from `request.json()` as the existing typed 400 JSON error; `count_tokens.py` extracts its existing parse block into `_read_body`, retains syntax fallback `{}` and raises typed 400 for only undecodable bytes. `chat.py` separately maps undecodable upstream JSON to its existing upstream-origin 502 response. No dump or signal code changed.
- `tests/test_routes_integration.py` adds hostile byte variants (invalid leading, truncated multibyte, invalid after BOM) for each route, no upstream dispatch for messages/chat, a count_tokens syntax-fallback witness, and an upstream-attribution witness. Fail-first 9/9 byte cases returned 500; upstream witness also failed 500 before fix. After fix, focused route suite: 24 passed (`-o addopts=` to avoid whole-project coverage threshold on narrow suite).
- Dependency check: installed `aiohttp/web_request.py:690,694` decodes bytes during `request.json()` before JSON parsing, so `UnicodeDecodeError` escapes the `JSONDecodeError` handlers; independent exception branches verified in claims 68–72. No shared claim changed; claim-52 malformed wording is not used as premise.
- Assumption finding `seam-S7` / DC-seam-S7-r1: implemented for Yield replay, not marked resolved; claim-52 remains malformed for assumption owner re-extraction. Assumption impact: scoped:S7; claim 68–72 single task.
- Override entries: none newly exercised. Decision reversals: none. Invariant changes: none. Durable bug rationale: distinguish syntax fallback on count_tokens from charset failure, and upstream parse error from client origin.
- No persistent QA gate test inspected or authored. QAP-MS002 R1 belongs to milestone Yield Phase. Hermetic suite `uv run pytest`: 238 passed, 92.88% total coverage (routes excluded by project config). Existing `test_server_hooks.py` unawaited-coroutine warnings and aiohttp deprecations remain; no test failure.
- Visual HTTP behavior: actual aiohttp TestServer terminal capture at [evidence/t003-001-http-response.txt](../evidence/t003-001-http-response.txt), indexed in [evidence/INDEX.md](../evidence/INDEX.md); all three routes show `HTTP 400 {"error": "Invalid JSON"}`. Disposition intentionally blank for user review.

## L→N→C→P and test gate
- New private extraction `_read_body`: L≈5/N1/C3/P1, local to the already-bound count_tokens parse boundary. Modified `handle_messages` L31/N2/C11/P1 (baseline equal); `handle_chat` L29/N2/C6/P1 (equal); `_make_chat_request` L23/N4/C6/P5 (equal); `handle_count_tokens` L39/N1/C7/P1 (improved from L42/C8). Existing violations not worsened. `tests/test_routes_integration.py` two parametrized functions P4 gray only; new test bodies all below L15. Full suite passed 238/238; coverage 92.88%.
- Scope sizing: messages 268→268, chat 260→260, count_tokens 76→85, route test module 341→387 (yellow, tests-only; no production file crossed 300). The one private extraction is a same-concept capacity repair; no new public API.
- Status: ready for Yield's independent review and MS-002 QA re-verdict; no new roadmap scope. Next task after milestone remains T004. Questions asked: none.
- Questions Asked: none. No unranked implementation choice.
