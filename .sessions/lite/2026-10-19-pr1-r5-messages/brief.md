# messages request-body fields: retry budget and preserved-thinking backend gate

- Session: 02.00-2026-09-13-message-dump-bulk-flush--9bf7afff5573
- Findings: [4318096159](https://github.com/AlexGS74/buster-ripper/pull/1#discussion_r4318096159), [4318096166](https://github.com/AlexGS74/buster-ripper/pull/1#discussion_r4318096166)
- Goal: an SGLang `/v1/messages` overflow retry reduces the generation budget the endpoint actually reads, and the preserved-thinking flag does not advertise an effect the SGLang `/v1/messages` endpoint discards.
- Done when:
  - an SGLang streaming overflow reissues attempts 2 and 3 with `max_tokens` reduced and no `max_completion_tokens` added; the non-streaming retry does the same;
  - an SGLang `/v1/messages` request with `PRESERVED_THINKING` on is forwarded without an injected `chat_template_kwargs` (a client-sent one is forwarded unchanged);
  - a vLLM or unknown backend still gets the injected pair exactly as today;
  - the chat retry still carries `max_completion_tokens`, and the chat `chat_template_kwargs` still carries `enable_thinking`/`thinking_budget`;
  - `README.md` §Serve script options, the `--preserved-thinking` help and the `config.PRESERVED_THINKING` comment all state the SGLang v0.5.9 `/v1/messages` limitation and the vLLM Anthropic support.
- Wrong witness: the retry writes `max_tokens` but the first attempt is pre-reduced; or the SGLang skip also skips vLLM; or the docs are corrected but the code still injects on SGLang.
- In scope: `buster_ripper/routes/messages.py`, `buster_ripper/config.py`, `buster_ripper/main.py`, `README.md`, `tests/test_context_overflow.py`, `tests/test_upstream_answer_relay.py`.
- Out of scope: `buster_ripper/dump.py` (the 1% record `PR1-R5-1`), `routes/chat.py`, `routes/count_tokens.py`, `routes/passthrough.py`, `normalize.py`, `image.py`, `server.py`.
- Agrees with: `FK-011` Option A (the default stays on); `T023`'s README section (corrected, not reverted); `T019.001`'s error-text probe (byte-unchanged).
- Verify: `uv run pytest tests/test_context_overflow.py tests/test_upstream_answer_relay.py tests/test_routes_integration.py -q --no-cov`; `uv run pytest -q`; `wc -l buster_ripper/routes/messages.py`; `git diff --check`.

## Forward
- Changed: `buster_ripper/routes/messages.py` — both retry loops now reduce `max_tokens` (the field `/v1/messages` reads) instead of writing `max_completion_tokens`; preserved-thinking injection is skipped when the backend is SGLang.
- Changed: `buster_ripper/config.py` — `PRESERVED_THINKING` comment states the SGLang v0.5.9 `/v1/messages` discard and the vLLM Anthropic support.
- Changed: `buster_ripper/main.py` — `--preserved-thinking` help states the backend gate.
- Changed: `README.md` — §Serve script options states the SGLang limitation and vLLM support.
- Changed: `tests/test_context_overflow.py` — streaming test asserts `max_tokens` and no `max_completion_tokens`; added the non-streaming retry test.
- Changed: `tests/test_upstream_answer_relay.py` — extended the absent-field test over vLLM/unknown; added the SGLang skip and client-mapping-unchanged tests.
- Ran: `uv run pytest tests/test_context_overflow.py tests/test_upstream_answer_relay.py tests/test_routes_integration.py -q --no-cov` -> 118 passed
- Ran: `uv run pytest -q` -> 575 passed
- Ran: `wc -l buster_ripper/routes/messages.py` -> 341
- Ran: `git diff --check` -> clean
- Noticed: `buster_ripper/utils.py:145` docstring still says `max_completion_tokens` though the helper now also sizes `max_tokens`; out of scope. Ask: no — docstring wording only, no wrong result.
- Noticed: `serve_buster_ripper.sh:30` comment names the flag's pair without the SGLang limitation; out of scope and the README/help carry it. Ask: no — env-value comment only, no behavior.

## Yield
- Verdict: ACCEPT
- Ran: `uv run pytest tests/test_context_overflow.py tests/test_upstream_answer_relay.py tests/test_routes_integration.py -q --no-cov` -> 118 passed
- Ran: `uv run pytest -q` -> 575 passed, 96.83% coverage, 80% gate reached
- Ran: `wc -l buster_ripper/routes/messages.py` -> 341; `git diff --check` -> clean (exit 0)
- Ran: `uvx --offline ruff@0.16.10 check` on the five changed source/test files -> All checks passed
- Ran: red check in a detached worktree at HEAD `05dac32` with only the two modified test files copied in -> the four new witnesses fail: streaming and non-streaming both `assert 10 == 256` at `test_context_overflow.py:255`/`:330` (pre-fix wrote `max_completion_tokens`, leaving `max_tokens` at 10), `test_sglang_gets_no_injected_pair` (the key was injected, so the absence assertion fails), `test_a_client_sent_mapping_is_forwarded_unchanged` (`{'clear_thinking': False, 'enable_thinking': False}` vs `{'enable_thinking': False}`); the extended vLLM/unknown arm stays green there, as a regression guard should
- Ran: `git status --porcelain --untracked-files=all` -> only the six listed files plus this brief, nothing from `Out of scope` moved (`routes/chat.py`, `routes/count_tokens.py`, `routes/passthrough.py`, `normalize.py`, `image.py`, `server.py`, `dump.py` untouched)
- Ran: branches of the gated code — `handle_messages`: flag off (no write), `sglang` (skip, body byte-unchanged), `vllm`/`unknown`/`None` (inject as before); `_inject_preserved_thinking`: absent field, non-mapping (replaced), client mapping (setdefault keeps client values); `handle_streaming`: non-SGLang early return, overflow with attempts left (writes `max_tokens`, reissues), overflow at the terminal attempt (loop's own 400), non-overflow 4xx (relayed once), transport 502 (crosses the probe); `handle_non_streaming`: non-SGLang early return, `ContextOverflowError` with attempts left (writes `max_tokens`, reissues), terminal attempt (400), non-overflow answer relayed — all hold
- Wrong witness: ruled out on all three. First attempt pre-reduced -> `current_body = dict(body)` is copied before the loop and only written inside the retry arm; `sent[0]["max_tokens"]` reads 10 on both loops. SGLang skip also skipping vLLM -> gate is `!= "sglang"`, and the vLLM/unknown parametrisation asserts the injected pair; `None` (unclassified) also injects, matching `server.on_startup`'s awaited detection before serving. Docs corrected but code still injecting on SGLang -> `messages.py:48` gates the call, and `test_sglang_gets_no_injected_pair` asserts the key absent. Chat arm untouched -> `routes/chat.py` unmodified and `TestChatStreamingOverflowRetry` still asserts `max_completion_tokens` on reissues 2 and 3.
- Findings: none
- Noticed: `buster_ripper/config.py:48-50` adds three comment lines over the brief's two-line figure; a two-line wording cannot carry both required facts at this file's width. Ask: no — documentation only, no behavior.
- Noticed: `buster_ripper/main.py:176` still prints `enabled (enable_thinking=true, clear_thinking=false)` in the startup banner, which on SGLang now describes an injection that is skipped. Ask: no — a startup log line printed before backend detection; the brief's Done-when names the help string only.
