VERDICT: HOLDS
PROMOTION: ELIGIBLE

## Claim
`_read_body` maps client `UnicodeDecodeError` from `request.json()` to `web.HTTPBadRequest`

## What it actually does
The `await request.json()` call is inside a try block whose `except UnicodeDecodeError` raises `web.HTTPBadRequest` with a JSON error message.

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
