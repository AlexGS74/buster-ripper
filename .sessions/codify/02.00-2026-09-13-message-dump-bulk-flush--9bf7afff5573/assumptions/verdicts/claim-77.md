VERDICT: HOLDS
PROMOTION: ELIGIBLE

## Claim
`detect_backend` returns `"unknown"` when an `Exception` is raised inside its probe `try` block

## What it actually does
The `except Exception` handler sets `_backend_type` to `"unknown"`; after caching the result, the function returns `_backend_type`.

## Source
`buster_ripper/backend.py:53-71`

## Source Version
unavailable

## Source Fingerprint
sha256:c2b71468af6b53efde753b0648d47eb6e06c8db4c738e08f7c01f4e4682d76ef

## Invalidates On
Changes to `detect_backend`'s exception handler or return path in `buster_ripper/backend.py`.

## Checked At
2026-09-28T00:13:19Z
