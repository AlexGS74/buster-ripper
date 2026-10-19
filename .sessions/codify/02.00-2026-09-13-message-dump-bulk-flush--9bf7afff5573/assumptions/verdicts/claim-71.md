VERDICT: HOLDS
PROMOTION: INELIGIBLE — two source locators needed and exact consulted bytes have no SHA-256 fingerprint

## Claim
`UnicodeDecodeError` is not a subclass of `json.JSONDecodeError`

## What it actually does
The documented built-in hierarchy puts `UnicodeDecodeError` under `UnicodeError` and `ValueError`; the JSON decoder separately declares `JSONDecodeError` as a direct subclass of `ValueError`. They are distinct branches, not a subclass relationship.

## Source
https://docs.python.org/3/library/exceptions.html#exception-hierarchy ; https://github.com/python/cpython/blob/v3.14.0/Lib/json/decoder.py#L20

## Source Version
Python 3.14.7 documentation; CPython v3.14.0 JSON decoder

## Source Fingerprint
unavailable

## Invalidates On
The built-in exception hierarchy changing in a later Python release.
The `JSONDecodeError` inheritance declaration changing in a later Python release.

## Checked At
2026-09-27
