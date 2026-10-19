# Keep an upstream cookie out of the shared proxy session

- Session: 02.00-2026-09-13-message-dump-bulk-flush--9bf7afff5573
- Findings: PR #1 review comment 4316745948 — `buster_ripper/server.py:66-68` (P1)
- Goal: An upstream `Set-Cookie` reaches the caller that earned it and never a later, unrelated caller; each request sends only the `Cookie` header its own caller forwarded.
- Done when: (a) a real upstream answers caller A with `Set-Cookie: affinity=1` and receives caller B's request with no `Cookie` header (red before the fix: it receives `affinity=1`); (b) caller A's own `Cookie: caller=own` reaches the upstream byte-identical and is not replaced by a stored cookie; (c) caller A's answer still carries `Set-Cookie: affinity=1` downstream; (d) `app["session"].cookie_jar` is a `DummyCookieJar` and the `/v1/models` probe and the `/metrics` poll share it.
- Wrong witness: only `Set-Cookie` is dropped from the downstream response (that breaks the relay contract); or a fresh `ClientSession` is built per request (that abandons `PI:T006/upstream_tls/optout_is_process_wide` and re-handshakes TLS per request); or the jar is emptied between requests (a timer/middleware, and it races concurrent requests).
- In scope: `buster_ripper/server.py` (`on_startup`'s `ClientSession`); `tests/test_server_hooks.py` (or a focused new module) for the witness.
- Out of scope: `utils.forward_headers`/`response_headers` (the request `Cookie` and the response `Set-Cookie` already travel); the `TCPConnector` and the TLS posture; `compaction.poll_kv_cache` and `backend.detect_backend`; the request-body content coding (`S16`/`RM-011`) and the response-direction sibling (`RM-008`).
- Agrees with: base parity (base built a per-request `httpx.AsyncClient` at `buster_ripper_legacy.py:668/704/847/906`, so no cookie outlived a request); `PI:T006/upstream_tls/optout_is_process_wide` (one session, one connector); `G-003`.
- Verify: `uv run pytest tests/test_server_hooks.py tests/test_backend.py -q`; `uv run pytest -q`.
- Capacity: `server.py` is 144 raw lines; one constructor operand keeps it green.

## Forward
- Changed: `buster_ripper/server.py` — `on_startup`'s `ClientSession` now passes `cookie_jar=aiohttp.DummyCookieJar()`; docstring states the one-session/no-replay rule (148 raw lines).
- Changed: `tests/test_session_cookie.py` — new witness against a real loopback upstream (`localhost`, because aiohttp's jar refuses cookies for a bare IP): caller B and the `/metrics` poll send no `Cookie` after caller A's answer sets one; caller A's own `Cookie: caller=own` arrives untouched; both `Set-Cookie` values still reach caller A; `app["session"].cookie_jar` is a `DummyCookieJar`.
- Ran: `uv run pytest tests/test_session_cookie.py -q --no-cov` -> 3 passed; each was red before the fix (upstream saw `affinity=1; caller=stored` on caller B, on the metrics poll, and in place of `caller=own`).
- Ran: `uv run pytest tests/test_server_hooks.py tests/test_backend.py -q` -> 37 passed, coverage gate fails at 37% because the subset does not cover the package (unchanged property of the command).
- Ran: `uv run pytest -q` -> 530 passed, coverage 97%.
- Table (`ClientSession` carrier at `server.py:66-68`):
  - `cookie_jar` -> `DummyCookieJar`; no store, no replay, the caller's own `Cookie` header untouched.
  - `headers`/`auth`/`proxy`/`proxy_auth`/`trust_env`/`skip_auto_headers`/`raise_for_status` -> not set; no per-caller state exists to carry.
  - `connector` -> one `TCPConnector(ssl=…)`; per-host pool, no caller identity.
  - `timeout` -> `total=600` shared; per-request overrides at three sites, not caller state.
  - `auto_decompress` -> aiohttp default; per-response, not cross-request.
- Noticed: `compaction.poll_kv_cache` swallows every exception (`except Exception: pass`), so a permanently failing `/metrics` poll is silent. Ask: no — the KV heuristic keeps its last value; nothing is leaked, lost or written wrong.
- Noticed: `utils.response_headers` still relays `Content-Encoding` over bytes aiohttp already decoded (RM-008, out of scope). Ask: no — already tracked in the roadmap.

## Yield
- Verdict: ACCEPT
- Ran: `git diff buster_ripper/server.py` -> only the two-line docstring paragraph and `cookie_jar=aiohttp.DummyCookieJar()` added; no other operand touched; `server.py` 148 raw lines
- Ran: `uv run pytest tests/test_session_cookie.py -q` -> 3 passed (coverage gate 33% is the subset, not a code failure)
- Ran: `uv run pytest tests/test_server_hooks.py tests/test_backend.py -q` -> 37 passed, coverage gate 37.29% (same subset property Forward reported)
- Ran: `uv run pytest -q` -> 530 passed, 96.72%
- Ran: aiohttp 3.13.3 jar check against the pre-fix construction -> `CookieJar()` stores `Set-Cookie` for `localhost` (`{'affinity': 1, 'caller': 'stored'}` on the next `/metrics` URL) and refuses a bare IP; `DummyCookieJar()` returns empty — so the test's `localhost` host is required and the witness is red without the fix
- Ran: read `ClientSession._request` (`client.py:682-689`) and `ClientRequest.update_cookies` (`client_reqrep.py:1100-1125`) -> with an empty jar `all_cookies` is empty, `if cookies is not None` builds an empty temp jar, `req_cookies` is empty, and `update_cookies` returns before touching the header: the caller's `Cookie` is left byte-identical
- Wrong witness: ruled out all three. Only-`Set-Cookie`-dropped -> `test_a_callers_own_cookie_is_forwarded_untouched` asserts both `Set-Cookie` values in `answer.raw` (utils untouched). Fresh `ClientSession` per request -> no construction added anywhere (`grep ClientSession(` leaves one production site, `server.py:69`), routes still read `app["session"]`, and test 3 asserts that object's jar is a `DummyCookieJar`. Jar emptied between requests -> no timer/middleware added; `DummyCookieJar.update_cookies` is a no-op and `filter_cookies` always returns an empty `SimpleCookie`
- Findings: none
- Noticed: the app's own `spawn_kv_cache_monitor` poller fires its first `/metrics` GET during `raw_proxy` startup, before the `upstream` fixture patches `config.UPSTREAM`; test 3's `seen["/metrics"] == [None]` therefore depends on that ordering (a poll landing after the patch would add a second entry and fail loudly). Ask: no — a different order fails the test visibly; no production result changes
- Noticed: test 3 asserts the jar's type on `app["session"]` but not that the poll and probe hold the same object; identity is structural (one session key, `start_kv_cache_monitoring` reads `app["session"]`). Ask: no — a second session key would fail the `isinstance` or the `seen` assertion in real use
- Noticed: `compaction.poll_kv_cache` swallows every exception (`except Exception: pass`), so a permanently failing `/metrics` poll is silent. Ask: no — the KV heuristic keeps its last value; nothing is leaked, lost or written wrong
- Noticed: `utils.response_headers` still relays `Content-Encoding` over bytes aiohttp already decoded (RM-008, out of scope). Ask: no — already tracked in the roadmap
- Noticed: `git status` also shows modified `.sessions/codify/...` handoff/roadmap files and two untracked assumption-decide iterations from other phases; they are session bookkeeping, not this change, and are left unstaged. Ask: no — no production behaviour depends on them
