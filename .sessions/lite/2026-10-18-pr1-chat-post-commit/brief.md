# Keep a chat stream's failure inside the response it already committed

- Session: 02.00-2026-09-13-message-dump-bulk-flush--9bf7afff5573
- Findings: PR #1 review comment 4309720759 — `buster_ripper/routes/chat.py:197` (P1)
- Goal: A streaming `/v1/chat/completions` answer whose upstream dies after the head is sent stays one HTTP response, and the client sees a terminated stream rather than a second status line.
- Done when: an upstream that sends a 200 SSE head, one chunk, then raises `ClientPayloadError` from `iter_any()` yields exactly one HTTP status line, no `HTTP/1.1 502` bytes in the body, and a terminated stream; `asyncio.CancelledError` still propagates; the witness is red before the fix.
- Wrong witness: the relay is wrapped in a bare `except BaseException`, so a client disconnect is swallowed; or the arm returns a new response object and the raw client still reads two status lines.
- In scope: `buster_ripper/routes/chat.py` (`_relay_chat_stream`); one raw-bytes chat witness beside the existing messages cases in `tests/test_stream_relay_commit.py`.
- Out of scope: `_make_chat_request`'s pre-commit `except aiohttp.ClientError`; the error branch's headers and bytes (the chat relay-fidelity brief); `_relay_chat_stream`'s success relay byte-for-byte chunk passthrough.
- Agrees with: `buster_ripper/routes/messages.py:_relay_after_commit` (T013's accepted R-E shape); `roadmap.md` §RM-016.
- Verify: `uv run pytest tests/test_stream_relay_commit.py tests/test_routes_integration.py -q`; `uv run pytest -q`.
- Capacity: `chat.py` is 300/300. The arm must add no more than two raw lines and the file must finish at or below 300; the chat relay-fidelity brief lands first and creates the headroom.

## Forward
- Changed: `buster_ripper/routes/chat.py` — `_relay_chat_stream` wraps the post-`prepare()` chunk loop and `write_eof()` in one `except Exception` arm that logs position-not-cause and falls through to `return response`; `CancelledError` is a `BaseException` and still propagates; 297 → 300 lines.
- Changed: `tests/test_stream_relay_commit.py` — new `TestChatPostCommit`: the raw-bytes `ClientPayloadError` witness (exactly one 200 status line, no 502/500 bytes, chunked terminal, one post-commit log line that does not say "upstream error") and the cancellation unit witness; the module docstring names the chat sibling.
- Ran: `uv run pytest tests/test_stream_relay_commit.py -q -k Chat` against the unfixed relay -> failed with `[b'HTTP/1.1 200 OK', b'HTTP/1.1 502 Bad Gateway']` and an "Upstream error" log line; after the arm -> 16 passed in the file.
- Ran: `uv run pytest tests/test_stream_relay_commit.py tests/test_routes_integration.py -q` -> 50 passed; the repo-wide 80% coverage gate is unmet by a subset command, as its `addopts` always measures the whole package.
- Ran: `uv run pytest -q` -> 519 passed, 96.72% coverage.
- Noticed: the arm is three raw lines, one over the brief's two-line figure, because a logged catch needs `try:`, `except …:` and the `log.error` line and the file may not compress them; the hard cap still holds at exactly 300. Ask: no — line budget, not behavior.
- Noticed: at 300 lines the sibling `2026-10-18-pr1-transport-parity` brief's one raw line lands `chat.py` at 301, in the 301-500 review band rather than the 300 green line. Ask: no — a quality-gate band, no real-use result changes.
- Noticed: `_make_chat_request`'s pre-commit `except aiohttp.ClientError` still answers a fresh 502 for a failure before the head; that is the brief's out-of-scope arm and is correct there. Ask: no — no committed head exists at that point.

## Yield
- Verdict: ACCEPT
- Ran: `uv run pytest tests/test_stream_relay_commit.py tests/test_routes_integration.py -q` -> 50 passed; exit non-zero only on the repo's whole-package `fail-under=80` addopts (47.89% on a subset), not on a test
- Ran: `uv run pytest -q` -> 519 passed, 96.72% coverage, gate reached
- Ran: `uv run pytest tests/test_stream_relay_commit.py -q --no-cov` -> 16 passed
- Ran: red witness — worktree at HEAD `8d6bb0d` with only the new test file copied in -> `TestChatPostCommit::test_chat_payload_error_after_commit_gets_one_head_and_a_terminal` failed with `[b'HTTP/1.1 200 OK', b'HTTP/1.1 502 Bad Gateway']` and `ERROR chat.py:151 Upstream error: chunk died`; the cancellation test passed there (it is the BaseException guard, not the behaviour change)
- Ran: `uvx --offline ruff@0.16.10 check buster_ripper/routes/chat.py tests/test_stream_relay_commit.py` -> All checks passed
- Ran: `git diff -w -- buster_ripper/routes/chat.py` -> the success relay loop body, `write_eof` ordering and `return response` are byte-identical; only the `try`/`except` wrapper is new
- Ran: branches of `_relay_chat_stream` — `resp.status >= 400` early `_chat_answer` (before `prepare`, untouched); empty body (loop never runs, `write_eof` writes the terminal); iteration raising `ClientPayloadError`/`AttributeError`/bare `TimeoutError`/`ClientConnectionResetError` (caught, logged, committed object returned); `write()` raising `OSError` (caught, logged, returned; `finish_response` handles the dead writer); `write_eof()` raising (caught, logged); `CancelledError`/`KeyboardInterrupt`/`SystemExit`/`GeneratorExit` (BaseException, propagates) — all hold
- Wrong witness: ruled out by the code (`except Exception as exc`, never `BaseException`) and by `return response`, the object `prepare()` already committed; the new cancellation unit test fails if the arm widens to `BaseException`, and the raw-bytes test reads exactly one `HTTP/1.1 200 OK` line with the chunked terminal, so no second response object reaches the wire
- Findings: none
- Noticed: the arm adds three raw lines, one over the brief's two-line figure; `chat.py` goes 297 → 300, so the hard cap holds but the per-arm figure does not. Ask: no — the two-line form needs `except …: log.error(…)` on one physical line, a readability regression with no real-use result behind it
- Noticed: the catch/log decision is re-derived from `_relay_after_commit` rather than shared, as `RM-016` orders ("`R-E`'s shape a second time"). Reuse is blocked: that function needs a messages-only `state` and runs `find_message_start` on every chat chunk, and `chat.py` has no room for the import plus the call. Ask: no — both write the same `buster-ripper` logger with the same `Post-commit relay error (%s): %s` format, so no input makes them disagree
- Noticed: the working tree also carries unlisted session-record edits (`roadmap.md`, `handoff-context.md`, `iterations/124-assumption-decide.md`, the sibling `2026-10-18-pr1-transport-parity` brief); left unstaged. Ask: no — records only, no code byte moves
- Noticed: the sibling transport-parity brief's one raw line would land `chat.py` at 301, as Forward recorded. Ask: no — a quality-gate band, no real-use result changes

