VERDICT: HOLDS
PROMOTION: INELIGIBLE — establishing both the repository construction and the library inheritance requires two source locators.

## Claim
The shared `ClientSession` is constructed with `aiohttp.ClientTimeout(total=600)` inside `on_startup` (`server.py:58`) — the default every call without a per-request timeout inherits

## What it actually does
`on_startup` creates the shared session with a 600-second total timeout. In installed aiohttp, `_request` uses `self._timeout` when its per-request timeout argument remains the default sentinel; an explicit request timeout replaces it.

## Source
`buster_ripper/server.py:55-65`; `.venv/lib/python3.13/site-packages/aiohttp/client.py:330-352,486-509,584-595` (installed source).

## Source Version
Repository revision unavailable; aiohttp 3.13.3.

## Source Fingerprint
`buster_ripper/server.py` sha256:6a3e45ef69413838dc53de2939ff1bb06b268746231bf63f3589be407f3d3a5a; `aiohttp/client.py` sha256:7ce41fc1c2142f5346015455e030e3ebec22a41cde0fc299a66ce13be2cb2a9e

## Invalidates On
Change to the session construction in `buster_ripper/server.py`.
Change to aiohttp `_request` timeout selection or `ClientSession` initialization.

## Checked At
2026-09-27
