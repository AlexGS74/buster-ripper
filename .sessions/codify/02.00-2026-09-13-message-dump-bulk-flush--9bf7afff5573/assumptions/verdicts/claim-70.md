VERDICT: HOLDS
PROMOTION: INELIGIBLE — exact consulted source bytes have no SHA-256 fingerprint

## Claim
`json.JSONDecodeError` derives directly from `ValueError`

## What it actually does
The standard-library decoder declares `class JSONDecodeError(ValueError):`, making `ValueError` its direct base.

## Source
https://github.com/python/cpython/blob/v3.14.0/Lib/json/decoder.py#L20

## Source Version
CPython v3.14.0

## Source Fingerprint
unavailable

## Invalidates On
The `JSONDecodeError` base-class declaration changing in a later Python release.

## Checked At
2026-09-27
