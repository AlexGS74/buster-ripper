VERDICT: HOLDS
PROMOTION: INELIGIBLE — exhaustive negative over a 26-file tree; no single exact file establishes it, the verdict rests on a bounded tree-wide search whose bytes are not recoverable as one fingerprinted locator

## Claim
No Python test source under `tests/` asserts that a hop-by-hop header is dropped at a proxy boundary

## What it actually does
Across all 26 Python files under `tests/` (test modules, `conftest.py`, and the `qa/` harness/entrypoint sources), no test asserts that any hop-by-hop header (`Connection`, `Keep-Alive`, `Proxy-Authenticate`, `Proxy-Authorization`, `TE`, `Trailer`, `Transfer-Encoding`, `Upgrade`) is dropped, stripped, or withheld at the proxy boundary. The only header assertions in the suite are on response `Content-Type` (`tests/test_routes_integration.py:154,173`); the only "header is stripped" assertion in the suite concerns a `x-anthropic-billing-header:` line embedded in a JSON *system-prompt string* (`tests/test_normalize.py:9-41`, `tests/test_routes_integration.py:110-135`), which is end-to-end payload text, not a hop-by-hop header crossing a proxy hop. Header-shaped strings elsewhere are the test peer's own HTTP plumbing (`tests/qa/tls_peer.py:90-98` sends `Content-Type`/`Content-Length`; `:305` sends `Connection: close` as a client request line for a self-probe) or prose comments about socket keep-alive (`:75`, `:231`) — none of them assert anything about a proxy dropping a header.

## Source
Subject is the test suite, so test sources are the production surface for this claim (P10 exception). Bounded surface searched: `/Users/alexhubitski/Projects/buster-ripper/tests/**/*.py` (26 files, `__pycache__` excluded), patterns `(?i)hop[-_ ]by[-_ ]hop|keep[-_]alive|proxy[-_]auth|transfer[-_]encoding|\bUpgrade\b|Trailer|\bTE\b|Connection`, plus `\bdrop|\bstrip|forward|passthrough|relay|filter`, `header`, `call_args|assert_called_with|assert_awaited_with|\.kwargs`, and `assert.*(header|Connection|auth)`. Closest specific files: `tests/test_routes_integration.py`, `tests/test_normalize.py`, `tests/qa/tls_peer.py`.

## Source Version
unavailable — the supplied locator is a working-tree directory; no revision metadata is readable from this role

## Source Fingerprint
unavailable — tree-wide negative; no single consulted file carries the verdict

## Invalidates On
- Any test under `tests/` gaining an assertion over forwarded/withheld request headers (e.g. inspecting `session.request`/`session.post` call kwargs for a header set, or a QA gate reading upstream request headers)
- Introduction of a hop-by-hop header filter helper and a test naming any of `Connection`, `Keep-Alive`, `Proxy-Authenticate`, `Proxy-Authorization`, `TE`, `Trailer`, `Transfer-Encoding`, `Upgrade`
- Renaming or relocating the `tests/` root so the searched surface is no longer the suite's full Python test source

## Checked At
2026-10-05
