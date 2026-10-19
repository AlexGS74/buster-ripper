VERDICT: HOLDS
PROMOTION: ELIGIBLE

## Claim
`_read_body` sets the raised `HTTPBadRequest` content type to `application/json`

## What it actually does
Its `UnicodeDecodeError` handler raises `web.HTTPBadRequest` with `content_type="application/json"` explicitly supplied.

## Source
`buster_ripper/routes/count_tokens.py:16-25`

## Source Version
unavailable

## Source Fingerprint
sha256:a69c9884cad5997e442a4eaa5c5557f13a589b227777b71fbe1c4167c38fa203

## Invalidates On
Changes to the `web.HTTPBadRequest` constructor arguments in `buster_ripper/routes/count_tokens.py`.

## Checked At
2026-09-27
