# Answer an upstream timeout and relay a redirect the way base did

- Session: 02.00-2026-09-13-message-dump-bulk-flush--9bf7afff5573
- Findings: PR #1 review comments 4309720765 (`passthrough.py:52`), 4309720790 (`passthrough.py:34`)
- Goal: A fired upstream deadline reaches each route's own JSON 502 envelope, and a 3xx from the upstream reaches the client as the upstream sent it.
- Done when: (a) a fake upstream raising `asyncio.TimeoutError` yields `502 {"error": "Upstream request failed"}` at the passthrough, non-stream messages and chat arms, with no generated 504; (b) a 302 with `Location` and a body reaches the client as 302 with `Location` and the upstream's body on the passthrough, messages non-stream, messages stream and chat calls; both witnesses are red before the fix.
- Wrong witness: only the passthrough arm catches `TimeoutError` and the other two still emit a generated 504; or `allow_redirects=False` is set only on the passthrough and the three POST calls still follow a redirect; or a 3xx on a streaming POST is relayed as a 200/SSE answer with no `Location`.
- In scope: `buster_ripper/routes/passthrough.py`, `buster_ripper/routes/messages.py`, `buster_ripper/routes/chat.py`; `tests/test_routes_integration.py`, `tests/test_upstream_answer_relay.py`.
- Out of scope: `compaction.py`'s poll and `backend.py`'s probe (they already catch `Exception`); the response-direction content coding (`RM-008`); `S17-N4`'s per-request ceiling; the chat relay-fidelity brief's body/header arms.
- Agrees with: `RM-010`'s whole-arm scope; every repo-authored client-facing failure's `{"error": …}` envelope; base's `httpx` defaults (`follow_redirects=False`); `G-003`.
- Verify: `uv run pytest tests/test_routes_integration.py tests/test_upstream_answer_relay.py -q`; `uv run pytest -q`.
- Capacity: `chat.py` is 300/300; this brief adds at most one raw line there (the redirect operand) and lands after the chat relay-fidelity brief. `messages.py` (334) and `passthrough.py` (56) stay under 500.

## Forward
- Changed: `buster_ripper/routes/passthrough.py` — `allow_redirects=False` on the upstream call; the 502 arm now also catches `TimeoutError`.
- Changed: `buster_ripper/routes/messages.py` — `allow_redirects=False` on both POSTs; both 502 arms catch `TimeoutError`; the stream arm relays a 3xx with its status, `Location` and body instead of preparing a 200 SSE head. 344/500.
- Changed: `buster_ripper/routes/chat.py` — `allow_redirects=False` on the POST (inlined to hold 300/300); the 502 arm catches `TimeoutError`; `_relay_chat_stream` sends a 3xx to `_chat_answer` rather than an SSE head.
- Changed: `tests/test_upstream_answer_relay.py` — new `TestUpstreamTimeout` (3 arms) and `TestUpstreamRedirect` (5 arms), plus a `TP-1` paragraph in the module docstring.
- Ran: new witnesses against `git stash push -- buster_ripper/routes` -> 8 failed (504 vs 502 ×3, `KeyError: 'Location'` ×5)
- Ran: `uv run pytest tests/test_routes_integration.py tests/test_upstream_answer_relay.py -q --no-cov` -> 78 passed
- Ran: `uv run pytest tests/test_routes_integration.py tests/test_upstream_answer_relay.py -q` -> 78 passed, exit 1 only on the subset coverage gate (47.27%: route modules are omitted from coverage, so this two-file subset cannot reach 80%)
- Ran: `uv run pytest -q` -> 527 passed (96.72% coverage)
- Noticed: `compaction.py`'s poll and `backend.py`'s probe still follow redirects and name no `TimeoutError`; the brief puts both out of scope and they catch `Exception`. Ask: no — neither answers a client.
- Noticed: the messages stream arm still names no `TimeoutError`; it arms `ClientTimeout(total=None)`, so no deadline can fire there. Ask: no — no deadline exists to catch.
- Noticed: a fired deadline logs `str(TimeoutError())`, which is empty, so the operator line reads "Upstream error: " with no cause; `_relay_after_commit` already names the type. Ask: no — the client-visible envelope is right and no data is lost.
- Noticed: `response_headers` keeps `Content-Encoding` over bytes aiohttp already decoded; that is the roadmap-tracked `RM-008` posture, out of scope. Ask: no — RM-008 owns it.

## Yield
- Verdict: ACCEPT
- Ran: `uv run pytest tests/test_routes_integration.py tests/test_upstream_answer_relay.py -q` -> 78 passed; exit 1 only on the two-file coverage gate (47.27%, route modules omitted from coverage)
- Ran: `uv run pytest -q` -> 527 passed, 96.72% coverage, 80% gate reached
- Ran: `uvx --offline ruff@0.16.10 check buster_ripper/ tests/` -> All checks passed!
- Ran: `wc -l` -> chat.py 300, messages.py 344, passthrough.py 57 (chat.py net 0 lines, as the brief's capacity clause requires)
- Ran: read every branch the change wraps: `passthrough.handle_passthrough` (:37 operand, :49 arm); `messages._stream_single_request` (:157 operand, :161 3xx branch, :191 arm); `messages._non_stream_single_request` (:310 operand, :342 arm); `messages.handle_streaming`'s overflow retry loop (302 fails `status >= 400`, so the redirect is returned, not retried); `chat._make_chat_request` (:144 operand, :150 arm); `chat._relay_chat_stream` (:189 3xx branch -> `_chat_answer` -> `_chat_response`, which keeps `Location` through `response_headers`); `chat._chat_answer` (`json.loads` fails on `b"moved"`, and a JSON 3xx parses with `status < 400` so `_raise_chat_overflow` returns early)
- Wrong witness: ruled out. (1) all three client-facing arms now name `TimeoutError` — `passthrough.py:49`, `messages.py:191/342`, `chat.py:150` — and installed aiohttp's `web_protocol._handle_request` turns a handler's `asyncio.TimeoutError` into `504` (verified in the installed source), so the old arms produced the 504 the tests rule out; the tests assert `502` plus `{"error": "Upstream request failed"}` and pass. (2) `allow_redirects=False` sits on all four upstream calls (`passthrough.py:37`, `messages.py:157/310`, `chat.py:144`), and each of the five redirect arms asserts the kwarg on the recorded call, so the passthrough-only state is excluded. (3) both streaming arms relay a 3xx as its own status and headers — `messages.py:161` returns `web.Response(status=resp.status, headers=response_headers(...))`, `chat.py:189` hands it to `_chat_answer` — so no 200 SSE head is prepared over a 3xx.
- Wrong witness (fail-first, read against the old code): the old arms caught only `aiohttp.ClientError`, so `asyncio.TimeoutError` escaped to aiohttp's 504 and the `502` assertion was red; the old streaming arms gated on `>= 400`, so a 302 prepared a `200` SSE head with no `Location` (red for both stream arms); no call passed `allow_redirects`, so `call.call_args.kwargs["allow_redirects"]` raised `KeyError` on the three non-stream arms (red).
- Findings: none
- Noticed: Forward's `Noticed` line says the messages stream arm "still names no `TimeoutError`", but the diff adds it at `messages.py:191`; the arm is inert because `ClientTimeout(total=None)` arms no deadline, so nothing client-visible changes. Ask: no — it is a stale description of the diff, not a behavior gap.
- Noticed: the brief's two-file `Verify` exits 1 on the subset coverage gate (route modules are omitted from coverage); the whole-suite run reaches 96.72%. Ask: no — Forward disclosed it and the repo's gate is whole-suite.
- Noticed: `_chat_answer` still applies response transforms to a JSON 3xx body, so a redirect whose body happened to carry `choices` would be rewritten; real redirects carry no such body and the chat relay-fidelity brief owns that arm. Ask: no — the brief's Done-when body is plain and no real redirect body is a chat payload.
- Noticed: the parent session's records (`handoff-context.md`, `roadmap.md`, `roadmap/2026-10-18-message-dump-bulk-flush.md`, `iterations/124-assumption-decide.md`) are dirty from the PR round's bookkeeping, not from this change; left unstaged. Ask: no — they are the orchestrator's records, and the protocol stages only the listed paths.
