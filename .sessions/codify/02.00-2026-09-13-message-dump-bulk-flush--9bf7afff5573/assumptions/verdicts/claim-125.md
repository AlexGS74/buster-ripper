VERDICT: HOLDS
PROMOTION: ELIGIBLE

## Claim
`asyncio.Task.cancelling()` retains the count of cancellation requests after `CancelledError` delivery unless explicitly reduced

## What it actually does
`cancel()` increments `_num_cancels_requested`, `cancelling()` returns it, and `uncancel()` decrements it; delivering `CancelledError` clears `_must_cancel` but does not decrement the count. The count is not itself a guarantee of another exception at the next await.

## Source
Installed Python standard library `Lib/asyncio/tasks.py`, `Task.cancel`, `Task.cancelling`, `Task.uncancel`, and `Task.__step` (lines 183–274); corroborated by Python 3.14 `asyncio` documentation, Task object / cancelling().

## Source Version
Python 3.14.7

## Source Fingerprint
sha256:a97ffdb60a42104a86f2ece781878b3371bd88e9eea8f1ac1fa8f397cc11cca5

## Invalidates On
Change to installed Python `asyncio/tasks.py` cancellation-count or exception-delivery logic.
Change to the active asyncio Task implementation or Python runtime version.

## Checked At
2026-09-27
