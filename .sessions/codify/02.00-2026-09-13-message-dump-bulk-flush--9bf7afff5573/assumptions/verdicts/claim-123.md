VERDICT: HOLDS
PROMOTION: ELIGIBLE

## Claim
On `GracefulExit`, aiohttp `run_app` cancels and awaits its `_run_app` task

## What it actually does
`run_app` catches `GracefulExit`, then its `finally` cancels `main_task` (created from `_run_app`) and drives the loop until that task completes, suppressing `CancelledError`.

## Source
Installed `.venv/lib/python3.13/site-packages/aiohttp/web.py`, lines 485–517.

## Source Version
aiohttp 3.13.3

## Source Fingerprint
sha256:27348d99e8e0e46e987850279088197f2b6a6d4f3ab0dbbce38c982a6a14a6ab

## Invalidates On
Change to installed aiohttp `web.py` implementation of `run_app`.

## Checked At
2026-09-27
