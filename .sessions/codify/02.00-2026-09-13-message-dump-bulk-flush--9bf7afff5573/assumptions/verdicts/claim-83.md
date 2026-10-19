VERDICT: HOLDS
PROMOTION: ELIGIBLE

## Claim
With its default expected JSON content type, aiohttp `ClientResponse.json()` raises `ContentTypeError` for a `text/event-stream` response

## What it actually does
Its default expected type is `application/json`; `text/event-stream` does not match the JSON MIME regex, so after reading the response body it raises `ContentTypeError` at the content-type check (assuming the read succeeds).

## Source
`.venv/lib/python3.13/site-packages/aiohttp/client_reqrep.py:102,260-265,742-765`

## Source Version
aiohttp 3.13.3

## Source Fingerprint
sha256:7844440eb67433c6454edd708c776e47e3fcfec9b8d0aeb980dcfe88c1986ac9

## Invalidates On
Change to installed aiohttp version or `client_reqrep.py` JSON MIME matching or response JSON handling.

## Checked At
2026-09-27
