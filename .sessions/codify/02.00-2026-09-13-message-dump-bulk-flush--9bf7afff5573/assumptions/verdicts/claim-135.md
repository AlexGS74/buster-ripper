VERDICT: HOLDS
PROMOTION: ELIGIBLE

## Claim
`_read_body` maps client `json.JSONDecodeError` from `request.json()` to an empty dictionary

## What it actually does
The `await request.json()` call is inside a try block whose `except json.JSONDecodeError` returns `{}`. The handler does not distinguish client-originated decode errors from other `JSONDecodeError`s raised by that call.

## Source
`buster_ripper/routes/count_tokens.py:16-25`

## Source Version
unavailable

## Source Fingerprint
sha256:a69c9884cad5997e442a4eaa5c5557f13a589b227777b71fbe1c4167c38fa203

## Invalidates On
Changes to `_read_body` exception handling in `buster_ripper/routes/count_tokens.py`.

## Checked At
2026-09-27
