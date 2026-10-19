VERDICT: HOLDS
PROMOTION: INELIGIBLE — the inheritance/status and request-handler exception handling require two source files; no single exact source locator establishes the whole claim.

## Claim
An uncaught aiohttp `web.HTTPBadRequest` from a request handler is sent as an HTTP 400 response

## What it actually does
The request protocol catches `HTTPException` raised by the handler and passes that exception as the response to `finish_response`. `HTTPBadRequest` inherits `HTTPException` and constructs its response with status code 400.

## Source
Installed aiohttp `.venv/lib/python3.13/site-packages/aiohttp/web_protocol.py:500-515`; `.venv/lib/python3.13/site-packages/aiohttp/web_exceptions.py:80-113,234-239`.

## Source Version
aiohttp 3.13.3

## Source Fingerprint
web_protocol.py sha256:eacf5d33399aa96efb6f33354f5d75b863522c5a3a80d99f0e0ed7cd89c0f319
web_exceptions.py sha256:ee722e8b0859dfdbc927d2ab5aa02b03941c59b51daa4cf608bc04a496a978df

## Invalidates On
aiohttp request protocol changes its `HTTPException` handling.
aiohttp changes `HTTPBadRequest` inheritance or status code.

## Checked At
2026-09-27
