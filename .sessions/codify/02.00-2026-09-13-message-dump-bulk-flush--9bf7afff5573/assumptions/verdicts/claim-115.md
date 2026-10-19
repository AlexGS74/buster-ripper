VERDICT: HOLDS
PROMOTION: ELIGIBLE

## Claim
`_cancel_background_task` catches `asyncio.CancelledError` but not ordinary exceptions raised by awaiting the task

## What it actually does
It cancels and awaits the task, swallowing only `asyncio.CancelledError`; any ordinary exception from `await task` propagates.

## Source
`buster_ripper/server.py:82-91` (production implementation)

## Source Version
unavailable

## Source Fingerprint
sha256:fbd579a7995d0f4a2ae8a6d2994bb9e5fda344c1a900ec581caee38f599647f8

## Invalidates On
Changes to `buster_ripper/server.py` exception handling in `_cancel_background_task`.

## Checked At
2026-09-27
