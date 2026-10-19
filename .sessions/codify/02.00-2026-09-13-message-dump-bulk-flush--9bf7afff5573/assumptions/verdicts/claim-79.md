VERDICT: HOLDS
PROMOTION: INELIGIBLE — establishing both the repository await and aiohttp's startup-before-serving ordering requires multiple source locators.

## Claim
`on_startup` awaits `detect_backend` before the app serves requests

## What it actually does
The registered startup callback awaits `backend.detect_backend`. Installed aiohttp awaits startup callbacks during runner setup, before it creates the request handler and before site start creates the listening server; the detection may short-circuit for a CLI override or cached state.

## Source
`buster_ripper/server.py:55-71,107-111`; `.venv/lib/python3.13/site-packages/aiohttp/web_app.py:467-472`; `.venv/lib/python3.13/site-packages/aiohttp/web_runner.py:116-129,278-290,389-396` (installed source).

## Source Version
Repository revision unavailable; aiohttp 3.13.3.

## Source Fingerprint
`buster_ripper/server.py` sha256:6a3e45ef69413838dc53de2939ff1bb06b268746231bf63f3589be407f3d3a5a; `aiohttp/web_app.py` sha256:94653f68030dfa1df0cbe2d34c78ba49e287f3276dd46e75057702b29803e590; `aiohttp/web_runner.py` sha256:bf51b59ca88e3908059d3491e0831ce88f5178414331a1ed30bbcefeba0333e0

## Invalidates On
Change to callback registration or `on_startup` in `buster_ripper/server.py`.
Change to aiohttp runner setup, site start, or startup signal dispatch.

## Checked At
2026-09-27
