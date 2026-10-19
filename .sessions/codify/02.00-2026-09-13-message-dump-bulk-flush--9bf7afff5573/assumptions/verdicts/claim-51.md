VERDICT: HOLDS
PROMOTION: INELIGIBLE — the full claim requires two source locators (Request.json/text and HeadersMixin.charset), rather than one exact locator establishing all of it.

## Claim
aiohttp's `Request.json()` decodes the request body using the charset from the Content-Type header (default utf-8) before parsing, so bytes undecodable under that charset raise `UnicodeDecodeError`, not `json.JSONDecodeError`

## What it actually does
`json()` awaits `text()` before calling the JSON loader; `text()` decodes body bytes using `self.charset or "utf-8"`, and `charset` is read from the Content-Type header. A decode failure propagates as `UnicodeDecodeError` before JSON parsing occurs (with the default decoder).

## Source
Installed `.venv/lib/python3.13/site-packages/aiohttp/web_request.py:686-695`; `.venv/lib/python3.13/site-packages/aiohttp/helpers.py:769-776`.

## Source Version
aiohttp 3.13.3 (installed distribution directory).

## Source Fingerprint
web_request.py sha256:2aaacea7a01e581e5eead2ab1b9e4ba3b65bc2ae3d0f176b2a78ae5b6b76bb4e
helpers.py sha256:435df4ecf0849d6cf844ff1cad4c3c764e7c734605d848b7272c2429f471cf91

## Invalidates On
Change to installed aiohttp `Request.json()` or `Request.text()` implementation.
Change to installed aiohttp `HeadersMixin.charset` header parsing.

## Checked At
2026-09-27
