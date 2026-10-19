VERDICT: HOLDS
PROMOTION: ELIGIBLE

## Claim
Awaiting a child task that terminates cancelled raises `CancelledError` in the awaiting task

## What it actually does
An awaited Task is Future-like: its `__await__` returns `result()`, and `result()` raises `CancelledError` when cancelled. The exception reaches the awaiting task unless its coroutine catches it; this does not by itself mean that the awaiting task was separately sent a cancellation request.

## Source
Installed Python standard library `Lib/asyncio/futures.py`, `Future.result` and `Future.__await__` (lines 195–209, 292–300); Python 3.14 `asyncio` documentation, Task object, confirms Task inherits Future APIs.

## Source Version
Python 3.14.7

## Source Fingerprint
sha256:2f482524292a7c5d1b818aeb92434d183bd089e014cf50b785432284d2c8bc12

## Invalidates On
Change to installed Python `asyncio/futures.py` cancelled-result or await behavior.
Change to the active asyncio Task implementation or Python runtime version.

## Checked At
2026-09-27
