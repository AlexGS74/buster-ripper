VERDICT: HOLDS
PROMOTION: ELIGIBLE

## Claim
`_relay_chat_stream` copies an upstream HTTP error response's Content-Type header into the downstream response when present

## What it actually does
In the HTTP error branch, when `"Content-Type" in resp.headers`, it passes `headers={"Content-Type": resp.headers["Content-Type"]}` to `web.Response`.

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
