VERDICT: HOLDS
PROMOTION: ELIGIBLE

## Claim
aiohttp `_run_app` awaits runner cleanup in its own `finally` block

## What it actually does
After `await runner.setup()`, `_run_app` enters a `try` whose `finally` performs `await runner.cleanup()`; failures before entering the `try` are not covered.

## Source
Installed `.venv/lib/python3.13/site-packages/aiohttp/web.py`, lines 325–331 and 421–425.

## Source Version
aiohttp 3.13.3

## Source Fingerprint
sha256:27348d99e8e0e46e987850279088197f2b6a6d4f3ab0dbbce38c982a6a14a6ab

## Invalidates On
Change to installed aiohttp `web.py` implementation of `_run_app`.

## Checked At
2026-09-27
