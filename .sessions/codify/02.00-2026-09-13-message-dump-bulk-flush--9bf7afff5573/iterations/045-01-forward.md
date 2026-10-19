## Orchestrator Prompt (verbatim)

> You are a subagent spawned by another session.
> Session ID: 02.00-2026-09-13-message-dump-bulk-flush--9bf7afff5573
> Task ID: T008

## Increment Brief
- T008: restore request-flag-selected byte-preserving chat SSE relay without changing JSON transforms.
- Anchors touched: chat route `buster_ripper/routes/chat.py:handle_chat` (existing route, extend); no new load-bearing production referent.
- Will NOT create: another chat handler, format selector, upstream client, or milestone QA gate.
- Reuse evidence: no touched `NEW` anchor; the existing `/v1/messages` streaming route supplies the relay pattern.
- Inventions: none; `_relay_chat_stream` and `_raise_chat_overflow` are private decomposition within existing chat route.

## Failure-Path Evidence
| Operation / condition | Expected outcome and authority | Handling / observation evidence | Disposition |
|---|---|---|---|
| Stream=true, finite SSE chunks including non-JSON bytes | Exact bytes, no JSON parse (T008 and S9) | Parameterized route tests assert body, SSE header and no JSON await | safe |
| Stream=true, upstream HTTP error | Preserve status and error bytes (messages streaming peer) | 429 regression reads response body and asserts no JSON await | safe |
| Stream=false, compatible `+json` | Existing JSON transformation (claim-63/S9) | Route test asserts think removal and JSON await | safe |
| Upstream transport error, cancellation or client disconnect | Existing aiohttp error mapping; do not swallow cancellation (route precedent) | ClientError catch remains outside context; no new catch around stream writes | safe |
| Malformed client JSON | Typed 400, no upstream call (T003.001) | Existing hostile-byte route tests unchanged and green | safe |
| Retry/duplicate delivery | Nonstream SGLang overflow retry retained; stream skips JSON-only overflow parsing | Existing overflow tests green; streaming dispatch occurs before `resp.json()` | safe |
| Recovery/competing updates | No persistent state on this route | No new state or deferred mutations | not-applicable |

## Implementation and evidence
- `buster_ripper/routes/chat.py`: dispatches on the request body `stream` flag inside the existing upstream context; relays success as streaming chunks with SSE headers, preserving chunk bytes. Upstream streaming errors retain their status/body. JSON parse and SGLang overflow logic remain on nonstream path; overflow inspection extracted into a private helper to reduce the existing hard nesting.
- `tests/test_routes_integration.py`: four combinations of SSE bytes/content-type proving flag authority and byte identity, 429 error, nonstream `application/problem+json` transform. The first SSE tests failed against old unconditional JSON path (500 with mock JSON), then passed after implementation. Existing parse and overflow tests remain green.
- Assumption finding `claim-63`, `claim-66`, `seam-S9`: implemented for Yield; not marked resolved. `claim-66` is extraction wording, not a production choice. No shared dependency premise changed. No invariant edit or override exercised.
- QA boundary: T008 is MS-002 remediation, no new approved VP handle or persistent QAP authoring. MS-002 process QA/re-verdict remains pending.

## Capacity gate and backlog pickup
- Capacity gate: additive `chat.py` 260 raw lines before edit; `_make_chat_request` L23 N4 C6 P5, `handle_chat` L29 N2 C6 P1. Triggered by hard N4. Existing chat and context-overflow characterization tests passed before extraction (19/19); overflow extraction ran before streaming edit and the same focused tests passed after extraction (19/19). Final `chat.py` 279 lines, `_make_chat_request` L18 N2 C4 P5: no hard violation, decreased length/nesting/complexity; `_raise_chat_overflow` L7 N2 C4 P2; `_relay_chat_stream` within ideal metrics. No >500 file or worsening metric.
- Backlog pickup: QB:routes-chat/request-lncp partially improved (hard N4 removed); QB:routes-chat/handler-lncp remains baseline, since restructuring its unrelated retry control would expand this remediation. Both stay unchecked for later quality pass. Test file 424 lines review-band, previously already review-band; no hard limit crossed.
- `codify scan path ... --git --detailed`: no new hard violation; `_make_chat_request` retains baseline gray L/N/P but each improved or unchanged. New test has only P4 gray. No lint/type tooling declared.

## Validation
- `uv run pytest tests/test_routes_integration.py::TestChatRoute tests/test_context_overflow.py -q -o addopts=''`: 29 passed.
- Final `uv run pytest`: 248 passed, total coverage 93.87% (routes omitted by project coverage config); 23 existing/unrelated coroutine/deprecation warnings. New route branch combinations are exercised by direct integration assertions, not measurable by package coverage omission.
- `git diff --check`: clean. Visual demo evidence not captured from a standalone live proxy; route-level in-process HTTP assertions are the local witness. Yield should not infer full-process acceptance from this.

## Durable signal / handoff
- Regression rationale: chat previously unconditionally parsed SSE as JSON; request `stream` flag now selects relay before parsing, regardless of upstream media type. No new decision, roadmap deferral or invariant change.
- Status: ready for local Yield. Next: MS-002 process QA/re-verdict by its owner, then planned tasks.
