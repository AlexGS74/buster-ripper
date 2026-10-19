VERDICT: HOLDS
PROMOTION: ELIGIBLE

## Claim
`asyncio.CancelledError` is not a subclass of `Exception`

## What it actually does
The installed Python 3.14.7 standard library defines `CancelledError` directly as a subclass of `BaseException`, not `Exception`.

## Source
`/opt/homebrew/Cellar/python@3.14/3.14.7/Frameworks/Python.framework/Versions/3.14/lib/python3.14/asyncio/exceptions.py:10`

## Source Version
Python 3.14.7

## Source Fingerprint
sha256:a5971f88be14cd1417d59adf539ae48c5d818f95362a4e0eb00017e3690ab37b

## Invalidates On
Change to the installed `asyncio/exceptions.py` definition of `CancelledError` or a switch to a Python version with a different base class (notably pre-3.8).

## Checked At
2026-09-27
