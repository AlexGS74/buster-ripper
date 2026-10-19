VERDICT: HOLDS
PROMOTION: ELIGIBLE

## Claim
`_make_chat_request` selects `_relay_chat_stream` when the outgoing request body has a truthy `stream` value

## What it actually does
It sends `body` as the upstream JSON payload, then branches on `body.get("stream")` and returns `_relay_chat_stream(request, resp)` when truthy.

## Source
`buster_ripper/routes/chat.py:114-133`

## Source Version
unavailable

## Source Fingerprint
sha256:3068981c63ff3737b223b5f9bb9018a23195034afbc67df77e3c8ec37ff0a9ce

## Invalidates On
Change to `_make_chat_request`'s outgoing body or stream branch in `buster_ripper/routes/chat.py`.

## Checked At
2026-09-27
