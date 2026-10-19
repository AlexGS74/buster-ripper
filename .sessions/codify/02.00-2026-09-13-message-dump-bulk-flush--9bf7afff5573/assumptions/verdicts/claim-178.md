VERDICT: HOLDS
PROMOTION: ELIGIBLE

## Claim
`on_startup` passes the explicit connector to its `aiohttp.ClientSession` construction

## What it actually does
`on_startup` (buster_ripper/server.py:66–68) constructs the session as `app["session"] = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=600), connector=connector)`, where `connector` is the explicit `aiohttp.TCPConnector(ssl=not config.SKIP_TLS_VERIFY)` built on the preceding line (server.py:65) — so the session is constructed on that explicit connector, not on an aiohttp-created default connector.

## Source
buster_ripper/server.py lines 65–68, inside `async def on_startup(app)` (lines 55–81)

## Source Version
unavailable (working-tree production source; identified by the sha256 fingerprint below — git metadata not consulted per protocol)

## Source Fingerprint
sha256:a9e49823ea456af9c0076781c4713c956745b21b026dfa768faf947e0b130592

## Invalidates On
`on_startup`'s `ClientSession(...)` call dropping the `connector=connector` argument (or otherwise constructing the session without the explicit connector, e.g. via a nested/default connector)
the connector construction moving out of `on_startup` so the session no longer receives the startup-built connector

## Checked At
2026-09-28
