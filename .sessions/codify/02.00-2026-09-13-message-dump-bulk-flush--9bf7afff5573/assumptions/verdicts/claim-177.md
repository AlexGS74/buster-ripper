VERDICT: HOLDS
PROMOTION: ELIGIBLE

## Claim
`on_startup` constructs its connector as `aiohttp.TCPConnector(ssl=not config.SKIP_TLS_VERIFY)`

## What it actually does
`on_startup` (buster_ripper/server.py:65) constructs its connector with exactly that expression: `connector = aiohttp.TCPConnector(ssl=not config.SKIP_TLS_VERIFY)`. The referenced global exists as `SKIP_TLS_VERIFY: bool = False` in `buster_ripper/config.py:17`, imported into server.py via `from . import ... config` (server.py:9).

## Source
buster_ripper/server.py line 65, inside `async def on_startup(app)` (lines 55–81); corroborated attribute definition at buster_ripper/config.py line 17

## Source Version
unavailable (working-tree production source; identified by the sha256 fingerprint below — git metadata not consulted per protocol)

## Source Fingerprint
sha256:a9e49823ea456af9c0076781c4713c956745b21b026dfa768faf947e0b130592

## Invalidates On
any change to server.py's `on_startup` that alters the connector construction line (different `ssl` expression, different connector class, or removal of the explicit connector)
`config.SKIP_TLS_VERIFY` being renamed, moved, or `on_startup` starting to read the TLS posture from a different source than `config.SKIP_TLS_VERIFY`

## Checked At
2026-09-28
