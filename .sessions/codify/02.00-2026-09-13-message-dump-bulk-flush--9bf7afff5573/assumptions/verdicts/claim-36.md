VERDICT: HOLDS
PROMOTION: ELIGIBLE

## Claim
An exception raised inside an app cleanup hook other than aiohttp's own `GracefulExit` propagates out of `run_app`

## What it actually does
`_run_app` awaits runner cleanup in `finally`, and `run_app` does not suppress ordinary exceptions from its main task: the exception escapes after loop teardown. Even when the first `run_until_complete` is interrupted by a signal, its second await of the cancelled main task suppresses only `CancelledError`, so a cleanup failure escapes; multiple cleanup-context failures may instead escape as an aggregate `CleanupError`.

## Source
Installed aiohttp `.venv/lib/python3.13/site-packages/aiohttp/web.py` lines 421–425, 508–521.

## Source Version
aiohttp 3.13.3

## Source Fingerprint
sha256:27348d99e8e0e46e987850279088197f2b6a6d4f3ab0dbbce38c982a6a14a6ab

## Invalidates On
Change to aiohttp `_run_app` cleanup await or `run_app` exception/suppression handling in web.py.

## Checked At
2026-09-27
