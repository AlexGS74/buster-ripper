VERDICT: HOLDS
PROMOTION: INELIGIBLE — the call site, session construction, and aiohttp semantics require more than one exact source locator.

## Claim
`detect_backend`'s `/v1/models` probe (`backend.py:54`) issues `session.get` with no per-request timeout, so it inherits the session default

## What it actually does
The probe calls `session.get(f"{backend_url}/v1/models")` without a timeout argument. For the shared session constructed in `on_startup`, aiohttp applies its configured `ClientTimeout(total=600)` unless overridden; the probe is skipped when the CLI override or cached result takes precedence.

## Source
`buster_ripper/backend.py:41-56`; `buster_ripper/server.py:55-65`; aiohttp Client Reference, `ClientSession.request(timeout=sentinel)` and `ClientSession.get`, https://docs.aiohttp.org/en/stable/client_reference.html#aiohttp.ClientSession.request

## Source Version
Repository source revision unavailable; aiohttp documentation: stable (installed package version unavailable).

## Source Fingerprint
sha256:145f150cbcceda7ca8f7715d8ff14d9c779aace39406affce86982b2dcbf1034 (`buster_ripper/backend.py`); sha256:fbd579a7995d0f4a2ae8a6d2994bb9e5fda344c1a900ec581caee38f599647f8 (`buster_ripper/server.py`); fetched documentation not fingerprinted.

## Invalidates On
Changes to `buster_ripper/backend.py` probe's request parameters.
Changes to `buster_ripper/server.py` shared-session timeout.
Changes to aiohttp request timeout inheritance semantics.

## Checked At
2026-09-27
