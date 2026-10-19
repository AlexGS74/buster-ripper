VERDICT: HOLDS
PROMOTION: ELIGIBLE

## Claim
`_relay_chat_stream` prepares a `text/event-stream` response before forwarding successful upstream chunks

## What it actually does
For upstream status below 400, it creates a `web.StreamResponse` with `Content-Type: text/event-stream`, awaits `response.prepare(request)`, and only then begins forwarding chunks.

## Source
`buster_ripper/routes/chat.py:155-168`

## Source Version
unavailable

## Source Fingerprint
sha256:3068981c63ff3737b223b5f9bb9018a23195034afbc67df77e3c8ec37ff0a9ce

## Invalidates On
Any change to `_relay_chat_stream` in `buster_ripper/routes/chat.py`, especially response headers or the ordering of `prepare` and chunk forwarding.

## Checked At
2026-09-27
