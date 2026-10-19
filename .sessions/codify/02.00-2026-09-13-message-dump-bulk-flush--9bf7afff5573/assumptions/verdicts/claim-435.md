VERDICT: HOLDS
PROMOTION: INELIGIBLE — the verdict rests on a two-file chain in the CPython source (raise site in Lib/urllib/request.py plus the `self.code = code` assignment in Lib/urllib/error.py); no single exact locator states both halves

## Claim
`urllib.request.urlopen` raises `urllib.error.HTTPError` carrying the response status on `.code` for an HTTP 4xx or 5xx answer

## What it actually does
The default opener's `HTTPErrorProcessor.http_response` routes every response whose status is outside 200–299 to `OpenerDirector.error('http', ..., code, ...)`, which dispatches `http_error_<code>`. No default handler claims 4xx/5xx codes, so `HTTPDefaultErrorHandler.http_error_default` raises `HTTPError(req.full_url, code, msg, hdrs, fp)`, and `HTTPError.__init__` stores that same status in `self.code`. (3xx responses are first offered to `HTTPRedirectHandler`, and `HTTPError` is a subclass of `URLError`, so it is still the "protocol errors" exception family the docs describe.)

## Source
- https://raw.githubusercontent.com/python/cpython/3.11/Lib/urllib/request.py — `HTTPErrorProcessor.http_response` (lines 628–637), `OpenerDirector.error` (lines 544–561), `HTTPDefaultErrorHandler.http_error_default` (lines 641–643), `build_opener` default handler list (lines 579–582)
- https://raw.githubusercontent.com/python/cpython/3.11/Lib/urllib/error.py — `HTTPError.__init__` sets `self.code = code`
- https://docs.python.org/3/library/urllib.error.html — `HTTPError.code`: "An HTTP status code as defined in RFC 2616"

## Source Version
CPython 3.11 branch source (project declares `requires-python = ">=3.11"` in pyproject.toml); docs.python.org/3 is Python 3.15.0 documentation

## Source Fingerprint
sha256:f084fd652ccd61bb4550924492940fe45b084438790b2058e24f9cd05197d023 (Lib/urllib/request.py)
sha256:d12b3cc66af3f42a8ebe63e1c91d24f92c6237b6a93a3702938dffabd812d77b (Lib/urllib/error.py)

## Invalidates On
- `HTTPErrorProcessor.http_response` stops treating non-2xx statuses as errors or stops forwarding the status code
- `HTTPDefaultErrorHandler.http_error_default` is changed to raise something other than `HTTPError`
- `HTTPError.__init__` stops assigning its status argument to `.code`
- The project moves to a Python version where this dispatch or attribute changed

## Checked At
2026-10-17T23:47:04Z
