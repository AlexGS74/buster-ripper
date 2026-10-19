# Re-probe the backend after a transient startup failure

- Session: 02.00-2026-09-13-message-dump-bulk-flush--9bf7afff5573
- Findings: PR #1 review comment 4316745957 — `buster_ripper/backend.py:62-69` (P2)
- Goal: A `/v1/models` probe that fails transiently at startup cannot leave the process in `unknown` for its lifetime; once the upstream answers, the backend is classified without a restart.
- Done when: (a) an upstream that refuses `/v1/models` at `on_startup` and later answers `owned_by: "SGLang"` classifies the backend `sglang` with no restart (red before the fix: it stays `unknown` forever); (b) `--backend-type` still wins and issues no probe; (c) a definitive `sglang`/`vllm` answer caches once and stops the retries; (d) an empty `data: []` answer is retried, not cached; (e) `on_cleanup` cancels and awaits the retry task before closing the session, and a cancelled retry leaves the state uncached; (f) an upstream that never answers never blocks a request.
- Wrong witness: caching the failure and retrying only inside the startup call (bounded retries still give up permanently); or re-probing on every request (a new latency on a normal path); or treating `unknown` as `sglang` (wrong for vLLM); or a retry task that outlives `on_cleanup` (a leaked task).
- In scope: `buster_ripper/backend.py` (the cache and the retry coroutine); `buster_ripper/server.py` (`on_startup` spawn, `on_cleanup` cancel); `tests/test_backend.py`, `tests/test_server_hooks.py`.
- Out of scope: the three `get_backend_state()` consumers (`normalize.py:195-196`, `chat.py:40-43`, `messages.py:91-94`); the 5 s probe bound and the `unknown` fallback value (`T007`); `compaction.poll_kv_cache`; the `--backend-type` CLI; `RM-005`'s CLI surface.
- Agrees with: T007's done-when (`bound the probe` + `ordinary/timeout fallback unknown` — the value and the bound, not the permanence); the package's own background-poll pattern (`spawn_kv_cache_monitor`/`spawn_dump_flush` + `_cancel_background_task`, `server.py:25-45`, `:86-106`); `compaction.poll_kv_cache`'s "upstream may be starting up; ignore silently"; `G-002`'s clean shutdown; `RM-005`.
- Verify: `uv run pytest tests/test_backend.py tests/test_server_hooks.py -q`; `uv run pytest -q`.
- Capacity: `backend.py` is 95 raw lines, `server.py` 144; both stay green (a private coroutine plus one spawn and one cancel line).

## Forward
- Changed: `buster_ripper/backend.py` — a failed or empty probe returns `unknown` without caching it; `is_backend_state_cached()` and `_retry_detect_backend()` re-probe every 5 s until a definitive answer (118 raw lines).
- Changed: `buster_ripper/server.py` — `on_startup` spawns the retry task only while the state is uncached; `on_cleanup` cancels and awaits it before the session closes (155 raw lines).
- Changed: `tests/test_backend.py` — a 503 at startup then SGLang is classified by the retry, a definitive answer stops it, an empty list is retried, the override probes nothing, and a stalled retry never blocks a state read or caches.
- Changed: `tests/test_server_hooks.py` — the override spawns no retry, an unknown detection spawns one, and cleanup cancels it before closing the session.
- Ran: `uv run pytest tests/test_backend.py tests/test_server_hooks.py -q` -> 45 passed (coverage gate reports 38% because only two files are collected; no test failed).
- Ran: `uv run pytest -q` -> 538 passed, coverage 96.78%.
- Family sweep (every module cache, audited):
  - `backend._backend_state_cached` + `_backend_type` -> **fixed**: the exception path (`:62-67`) and the empty-list path (`_classify_models:78-79`) no longer cache; a definitive non-empty answer still caches once.
  - `compaction._kv_cache_usage` -> safe: last good value retained, re-polled every 10 s, `0.0` is the safe direction for `>=`.
  - `image._image_cache` -> safe: positive entries only, TTL + LRU, no failure cached.
  - `dump._sessions` -> safe: a failed flush keeps `pending`; a refusal caches nothing.
  - `session._stats`/`_db_conn` -> safe: positive updates only; a DB failure logs and keeps memory.
  - `functools.lru_cache`/`functools.cache` -> absent from the package.
- Noticed: a permanently unreachable upstream logs one probe warning every 5 s while retrying; Ask: no — log volume only, no lasting wrong result.

## Yield
- Verdict: ACCEPT
- Ran: `uv run pytest tests/test_backend.py tests/test_server_hooks.py -q` -> 45 passed (the 38% coverage gate is only the two collected files; no test failed); `uv run pytest -q` -> 538 passed, coverage 96.78%; `git diff --check` -> clean; `git status --porcelain` -> only the four listed files changed in code, plus the parent session's `.sessions/codify/...` bookkeeping and the lite brief, no other source touched.
- Wrong witness: not caching the failure — `detect_backend` returns at `backend.py:75-76` before `_backend_state_cached = True`, and the new 503/empty tests assert `not is_backend_state_cached()`; not per-request probing — `detect_backend` is called only from `server.on_startup` and `_retry_detect_backend` (`grep`), no route calls it; not unknown-as-sglang — `_classify_models` returns `sglang` only when `"sglang" in owned_by` (`backend.py:90-91`); no leaked task — `on_cleanup` cancels and awaits `backend_retry_task` first (`server.py:96`) before `session.close()` (`server.py:100`), and the cleanup test records `retry.cancelled()` at close. Branches of the gated `detect_backend` checked: override, cached, sglang, vllm, unrecognized non-empty, empty, probe exception.
- Findings: none
- Noticed: the Forward's log-volume note is accurate; a permanently unreachable upstream warns once per 5 s. Ask: no — log volume only, no lasting wrong result.
