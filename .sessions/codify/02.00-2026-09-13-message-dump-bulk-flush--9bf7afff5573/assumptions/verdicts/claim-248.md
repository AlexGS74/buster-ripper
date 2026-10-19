VERDICT: HOLDS
PROMOTION: INELIGIBLE — exhaustive negative over a 26-file tree; no single exact file establishes it, and the post-`prepare()` ordering detail lives in production source outside the supplied locator

## Claim
No Python test source under `tests/` exercises a failure raised after `prepare()` inside `_stream_single_request`

## What it actually does
No Python file under `tests/` names `_stream_single_request`, or calls/patches `prepare()` or any request-preparation collaborator at all — the identifiers `prepare`, `_stream_single_request`, `ClientRequest` and `client_reqrep` return zero matches across all 26 files (the only `_stream` hits are test *names*, e.g. `test_chat_stream_relays_exact_sse_bytes`). The suite intercepts the upstream hop one level above that function: route tests stub the shared session's `post`/`request` methods to return mock responses (`tests/test_routes_integration.py:71-75,102,151,169,240,343-347,406`), and compaction tests stub `session.get` (`tests/test_compaction.py:216,316,332`). Every failure the suite injects is at those stub boundaries or in proxy-local plumbing, never inside a real request lifecycle after preparation: `dump_turn` raising `RuntimeError` (`test_routes_integration.py:399-402`), `upstream.json` raising `UnicodeDecodeError` (`:239`), `flush_all`/`Path.open` raising `OSError` (`test_server_hooks.py:175`, `test_dump_flush.py:132`), `aiohttp.ClientError`/`asyncio.TimeoutError` on `session.get` (`test_compaction.py:216,316,332`), and `asyncio.CancelledError` in cleanup hooks (`test_server_hooks.py:298,341`). Under the narrow reading (a test that drives that function and fails it post-`prepare()`) and under the loose reading (any test run that trips that path), nothing in the test surface qualifies, so the verdict does not depend on choosing a definition.

## Source
Subject is the test suite, so test sources are the production surface for this claim (P10 exception). Bounded surface searched: `/Users/alexhubitski/Projects/buster-ripper/tests/**/*.py` (26 files, `__pycache__` excluded), patterns `_stream_single_request|\.prepare\(|prepare` (0 matches), `ClientRequest|client_reqrep|iter_any|iter_chunked|conn\b|writer`, `side_effect|raise |ClientError|ServerTimeoutError|TimeoutError`, `"post"|\.post\b|"request"|\.request\b|_stream`. Closest specific files: `tests/test_routes_integration.py`, `tests/test_compaction.py`, `tests/test_server_hooks.py`.

## Source Version
unavailable — the supplied locator is a working-tree directory; no revision metadata is readable from this role

## Source Fingerprint
unavailable — tree-wide negative; no single consulted file carries the verdict

## Invalidates On
- A test patching or calling `_stream_single_request` directly, or patching an aiohttp preparation collaborator (`ClientRequest.prepare`, a connection/writer stub) with a raising side effect
- A fixture that re-points the routes' upstream stub onto the real session object so the production `_stream_single_request` body executes under a post-`prepare()` fault
- Any test gaining a `prepare`-aware double for the streaming path (e.g. a fake request object whose `prepare()` succeeds and whose next call raises)
- Production source outside this locator (`buster_ripper/`) being supplied and showing `prepare()` sits inside a differently-named helper that the suite already fails — the naming evidence above would then need re-checking against that source

## Checked At
2026-10-05
