VERDICT: HOLDS
PROMOTION: ELIGIBLE

## Claim
With its default expected JSON content type, aiohttp `ClientResponse.json()` accepts `application/problem+json`

## What it actually does
The default expected type `application/json` uses a regex that matches `application/problem+json`, so this MIME type passes the content-type check; subsequent decoding still depends on a valid response body.

## Source
`.venv/lib/python3.13/site-packages/aiohttp/client_reqrep.py:102,260-265,742-772`

## Source Version
aiohttp 3.13.3

## Source Fingerprint
sha256:7844440eb67433c6454edd708c776e47e3fcfec9b8d0aeb980dcfe88c1986ac9

## Invalidates On
Change to installed aiohttp version or `client_reqrep.py` JSON MIME matching or response JSON handling.

## Checked At
2026-09-27
