VERDICT: HOLDS
PROMOTION: ELIGIBLE

## Claim
`_relay_chat_stream` passes no explicit response headers when an upstream HTTP error lacks Content-Type

## What it actually does
When the upstream error response has no `Content-Type` key, the conditional `headers` argument passed to `web.Response` is `None`; this does not assert that the framework emits no default headers.

## Source
`buster_ripper/routes/chat.py:155-160`

## Source Version
unavailable

## Source Fingerprint
sha256:16c7e99938b55e508b36c16092e7a89883baeff293385c3dfa3e9c161b413cae

## Invalidates On
Change to the error response header expression in `buster_ripper/routes/chat.py`.

## Checked At
2026-09-27
