VERDICT: HOLDS
PROMOTION: INELIGIBLE — the conclusion requires both the production control flow and Python's `CancelledError` hierarchy, not one exact source locator.

## Claim
A `CancelledError` escaping the probe's `try` block in `detect_backend` bypasses its cache assignment

## What it actually does
The probe catches `Exception`, but `asyncio.CancelledError` subclasses `BaseException` on supported Python versions (>=3.11). When it escapes the `try`, execution does not reach the `_backend_state_cached = True` assignment following the handler.

## Source
`buster_ripper/backend.py:53-69`; `pyproject.toml:4`; Python documentation, https://docs.python.org/3/library/asyncio-exceptions.html#asyncio.CancelledError (changed in Python 3.8).

## Source Version
Python documentation 3.14.7; repository Python requirement >=3.11; production source revision unavailable.

## Source Fingerprint
sha256:c2b71468af6b53efde753b0648d47eb6e06c8db4c738e08f7c01f4e4682d76ef (`buster_ripper/backend.py` only)

## Invalidates On
Modification of `buster_ripper/backend.py` exception handling or cache assignment.
Change to the supported Python version range or the `asyncio.CancelledError` inheritance hierarchy.

## Checked At
2026-09-28T00:15:27Z
