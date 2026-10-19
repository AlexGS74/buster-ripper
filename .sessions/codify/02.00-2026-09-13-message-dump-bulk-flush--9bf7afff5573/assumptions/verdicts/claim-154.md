VERDICT: HOLDS
PROMOTION: ELIGIBLE

## Claim
`ClientResponse.headers` exposes the upstream response's Content-Type header value

## What it actually does
After response processing, aiohttp assigns the parsed response message's headers to `_headers`; `ClientResponse.headers` returns that case-insensitive header mapping, including its Content-Type value when present. This is the header value, not the separate `content_type` parsing/fallback property.

## Source
`.venv/lib/python3.13/site-packages/aiohttp/client_reqrep.py:427-429,528-565`

## Source Version
aiohttp 3.13.3

## Source Fingerprint
sha256:7844440eb67433c6454edd708c776e47e3fcfec9b8d0aeb980dcfe88c1986ac9

## Invalidates On
Change to aiohttp's `ClientResponse.start` header assignment or `ClientResponse.headers` property.

## Checked At
2026-09-27
