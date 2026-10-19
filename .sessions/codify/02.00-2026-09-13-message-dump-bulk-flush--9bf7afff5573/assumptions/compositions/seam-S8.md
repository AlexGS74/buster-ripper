VERDICT: SAFE
PROMOTION: INELIGIBLE — member evidence lacks permanent handles, and claim 144 has no source fingerprint.

## Seam
S8 — startup backend probe duration from the shared 600 s session timeout through conditional `/v1/models` GET, its request-local 5 s timeout, and backend cache.

## Composition
`on_startup` creates a shared `ClientSession(ClientTimeout(total=600))`, then awaits `detect_backend` before aiohttp constructs the request handler or starts its site (`server.py:55-71,107-111`; `web_runner.py:278-289,389-396,116-129`). A truthy CLI override is stored and returned first; otherwise cached state is returned; neither branch issues a GET (`backend.py:41-50`). With neither branch, the producer issues `GET {backend_url}/v1/models` with `ClientTimeout(total=5)` (`backend.py:52-56`). Installed aiohttp selects that explicit object as `real_timeout`, replacing rather than inheriting the session's 600 s total for this request (`client.py:330-352,486-509,584-595`). An ordinary exception within the probe `try`, including an ordinary request timeout, sets `unknown`, marks the state cached, and returns it (`backend.py:53-71`). A `CancelledError` is outside `Exception` and escapes before the cache assignment (`backend.py:62-69`; Python `asyncio/exceptions.py:10`); the startup await then does not complete normally. Other explicit production per-request timeout sites use separate values: `/metrics` 5 s, chat 300 s, passthrough 60 s (`compaction.py:28-39`; `routes/chat.py:114-130`; `routes/passthrough.py:24-36`). They do not modify the shared session default or the probe's request-local object.

## Dimension Coverage
- Override versus cache versus uncached GET: all three mutually ordered branches traced; truthy override precedes existing cache, cache precedes GET. A falsey override does not bypass cache.
- Session/request timeout precedence: 600 s session default, explicit 5 s probe total, and distinct explicit peer values traced. The request-local object replaces the session default only for the probe; no timeout-field merge.
- Probe settlement: normal classification reaches cache assignment; ordinary probe exception or timeout returns cached `unknown`; cancellation skips both handler and assignment, propagating through the startup await rather than returning `unknown`.
- Startup ordering: uncached probe must finish or raise before request serving starts; override and cached branches return without probing. No claim of an exact five-second wall-clock deadline or guaranteed cleanup on aborted startup is made.
- Delta scoping: full trace; the supplied S8 row does not name a prior composition or its pins.

## Counterexample
None found across the enumerated dimensions. A cached `unknown` persists on later calls without a truthy override, and startup cancellation does not settle to `unknown`; both follow the specified branch mechanics rather than contradicting another listed member.

## Sources
`buster_ripper/server.py:55-71,107-111`; `buster_ripper/backend.py:41-85`; `buster_ripper/compaction.py:28-39`; `buster_ripper/routes/chat.py:114-130`; `buster_ripper/routes/passthrough.py:24-36`; `.venv/lib/python3.13/site-packages/aiohttp/client.py:330-352,486-509,584-595`; `.venv/lib/python3.13/site-packages/aiohttp/web_app.py:467-472`; `.venv/lib/python3.13/site-packages/aiohttp/web_runner.py:116-129,278-289,389-396`; `/opt/homebrew/Cellar/python@3.14/3.14.7/Frameworks/Python.framework/Versions/3.14/lib/python3.14/asyncio/exceptions.py:10`; `pyproject.toml:4`. Claim 57 also cites upstream aiohttp `v3.12.15/aiohttp/client.py` for its equivalent request-selection contract; installed aiohttp source independently establishes the exercised behavior.

## Source Versions
Repository revision unavailable; installed aiohttp 3.13.3; claim 57 upstream aiohttp v3.12.15; installed Python 3.14.7 for cancellation ancestry; project requires Python >=3.11.

## Source Fingerprint
unavailable — no permanent handles for all members and claim 144 supplies no fingerprint over its exhaustive production-source census.

## Invalidates On
Any change to member fingerprints or their anchors; startup session construction or callback registration/await order; backend override/cache/probe ordering, exception handling, classification or cache assignment; aiohttp request timeout selection, forwarding or runner startup ordering; Python cancellation inheritance or supported version range; any production `buster_ripper/` addition/removal/move of an explicit per-request `ClientTimeout` call site, including the current peer sites.

## Checked At
2026-09-28
