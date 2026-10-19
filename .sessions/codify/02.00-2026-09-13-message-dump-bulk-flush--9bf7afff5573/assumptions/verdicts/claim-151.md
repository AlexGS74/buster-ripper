VERDICT: HOLDS
PROMOTION: ELIGIBLE

## Claim
`_relay_chat_stream` returns upstream HTTP error status and raw body without calling `resp.json()`

## What it actually does
For `resp.status >= 400`, it constructs `web.Response` with `status=resp.status` and `body=await resp.read()`; this branch never calls `resp.json()`.

## Source
`buster_ripper/routes/chat.py:155-160`

## Source Version
unavailable

## Source Fingerprint
sha256:16c7e99938b55e508b36c16092e7a89883baeff293385c3dfa3e9c161b413cae

## Invalidates On
Change to the error branch of `_relay_chat_stream` in `buster_ripper/routes/chat.py`.

## Checked At
2026-09-27
