VERDICT: HOLDS
PROMOTION: ELIGIBLE

## Claim
`aiohttp.web.Response` accepts an explicit Content-Type response header alongside a byte body

## What it actually does
`Response(body=some_bytes, headers={"Content-Type": "application/example"})` retains the supplied header and stores the bytes as its body. Supplying `content_type` or `charset` constructor parameters in addition to that header is forbidden and raises `ValueError`.

## Source
/Users/alexhubitski/Projects/buster-ripper/.venv/lib/python3.13/site-packages/aiohttp/web_response.py:628-688,697-703

## Source Version
aiohttp 3.13.3 (installed)

## Source Fingerprint
sha256:3ca73388d5382e67ed5ea295be844cac06ce7950a5a5237e8b39c7b2259ece65

## Invalidates On
Changes to installed aiohttp/web_response.py constructor header validation or bytes body setter.

## Checked At
2026-09-27
