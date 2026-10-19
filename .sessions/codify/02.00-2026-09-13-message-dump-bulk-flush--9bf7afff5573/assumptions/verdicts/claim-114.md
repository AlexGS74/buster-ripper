VERDICT: HOLDS
PROMOTION: ELIGIBLE

## Claim
Awaiting a completed `asyncio.Task` that failed with an ordinary exception raises its stored exception

## What it actually does
The Future await implementation immediately calls `result()` when already done, and `result()` raises the stored exception for a finished future with an exception; Tasks inherit this Future behavior.

## Source
Installed CPython `asyncio/futures.py`, `Future.__await__` and `Future.result`, lines 195–209 and 292–298.

## Source Version
CPython 3.14.7 (Homebrew installed standard library)

## Source Fingerprint
sha256:2f482524292a7c5d1b818aeb92434d183bd089e014cf50b785432284d2c8bc12

## Invalidates On
Change to installed `asyncio/futures.py` Future await/result implementation or replacement of the selected Python runtime.

## Checked At
2026-09-27
