VERDICT: HOLDS
PROMOTION: ELIGIBLE

## Claim
aiohttp: a per-request `timeout=` argument to a `ClientSession` request method (`get`/`post`/`request`) replaces the session's default `ClientTimeout` for that single request

## What it actually does
`get`, `post`, and `request` forward their keyword arguments to `_request`. There, omitted `timeout` selects `self._timeout`, while an explicit `ClientTimeout` becomes `real_timeout` instead; the session's `_timeout` is not changed. A new timeout object replaces rather than merges its fields with the session default.

## Source
https://raw.githubusercontent.com/aio-libs/aiohttp/v3.12.15/aiohttp/client.py (`ClientSession._request`, `ClientSession.get`, `ClientSession.post`, `ClientSession.request`)

## Source Version
aiohttp v3.12.15 upstream source (installed version unavailable)

## Source Fingerprint
sha256:526c30a03bab9830efc53c1ae1ed55125928e2673cfd29ecbecdc203a484fa47

## Invalidates On
Change to aiohttp's request timeout selection or `get`/`post`/`request` forwarding in the used version.
Use of an aiohttp version whose request timeout behavior differs from v3.12.15.

## Checked At
2026-09-27
