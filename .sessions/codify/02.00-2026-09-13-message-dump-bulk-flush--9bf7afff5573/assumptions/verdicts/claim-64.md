VERDICT: HOLDS
PROMOTION: ELIGIBLE

## Claim
`aiohttp.ContentTypeError` is a subclass of `aiohttp.ClientError`

## What it actually does
`ContentTypeError` inherits `ClientResponseError`, which inherits `ClientError`.

## Source
`.venv/lib/python3.13/site-packages/aiohttp/client_exceptions.py:61-65,144`

## Source Version
aiohttp 3.13.3

## Source Fingerprint
sha256:bb229bc48da979984a97b9442c1331dd47aeb0d91fa5e98f5a9185ab4afa25e3

## Invalidates On
Change to installed aiohttp version or `client_exceptions.py` inheritance declarations.

## Checked At
2026-09-27
