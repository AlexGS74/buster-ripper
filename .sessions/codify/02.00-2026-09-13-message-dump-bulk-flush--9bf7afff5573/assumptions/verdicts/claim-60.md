VERDICT: DOES-NOT-HOLD
PROMOTION: INELIGIBLE — the repository handler and Python's cancellation exception hierarchy require separate sources to establish the universal counterexample.

## Claim
`detect_backend` falls back to `"unknown"` on any probe failure — the `except Exception` at `backend.py:78-83` logs and returns `"unknown"`

## What it actually does
The handler catches failures that subclass `Exception`, logs a warning, sets `"unknown"`, and returns it at line 87; it does not catch all failures, notably `asyncio.CancelledError` (a `BaseException` subclass). Failures before the `try` or exceptions not subclassing `Exception` also do not enter this fallback.

## Source
`buster_ripper/backend.py:39-87` (the `try` boundary and `except Exception`); Python standard library exception hierarchy: https://docs.python.org/3/library/asyncio-exceptions.html#asyncio.CancelledError

## Source Version
Repository source revision unavailable; Python standard library documentation (current; Python 3.8+ cancellation hierarchy).

## Source Fingerprint
sha256:145f150cbcceda7ca8f7715d8ff14d9c779aace39406affce86982b2dcbf1034

## Invalidates On
Changes to `buster_ripper/backend.py` exception handler type or scope.
Changes to Python `asyncio.CancelledError` inheritance hierarchy.

## Checked At
2026-09-27
