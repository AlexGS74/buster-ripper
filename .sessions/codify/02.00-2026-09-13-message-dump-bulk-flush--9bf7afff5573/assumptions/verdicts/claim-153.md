VERDICT: HOLDS
PROMOTION: ELIGIBLE

## Claim
`_relay_chat_stream` labels upstream HTTP error responses `application/json` regardless of the upstream Content-Type

## What it actually does
For upstream status >= 400, it reads the upstream body and returns `web.Response` with `content_type="application/json"`; it never inspects the upstream Content-Type in this branch.

## Source
`buster_ripper/routes/chat.py:155-160`

## Source Version
unavailable

## Source Fingerprint
sha256:3068981c63ff3737b223b5f9bb9018a23195034afbc67df77e3c8ec37ff0a9ce

## Invalidates On
Changes to `buster_ripper/routes/chat.py` affecting `_relay_chat_stream`'s error-response branch.

## Checked At
2026-09-27
