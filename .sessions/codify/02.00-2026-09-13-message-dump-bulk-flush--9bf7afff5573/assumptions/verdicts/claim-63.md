VERDICT: DOES-NOT-HOLD
PROMOTION: ELIGIBLE

## Claim
aiohttp's `resp.json()` raises `ContentTypeError` when the response's Content-Type is not `application/json` — e.g. `text/event-stream` (SSE)

## What it actually does
With its default `content_type="application/json"`, `ClientResponse.json()` rejects `text/event-stream` with `ContentTypeError`, but also accepts types such as `application/problem+json`, despite their not being `application/json`. Callers can override or disable the expected content-type check with the `content_type` argument. Thus the categorical claim is false, although its SSE example is true.

## Source
Installed `.venv/lib/python3.13/site-packages/aiohttp/client_reqrep.py`, lines 102, 260–265, 742–773.

## Source Version
aiohttp 3.13.3

## Source Fingerprint
sha256:7844440eb67433c6454edd708c776e47e3fcfec9b8d0aeb980dcfe88c1986ac9

## Invalidates On
Change to installed aiohttp `client_reqrep.py` content-type matching or `ClientResponse.json()` defaults/validation.

## Checked At
2026-09-27
