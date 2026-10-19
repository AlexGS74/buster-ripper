VERDICT: HOLDS
PROMOTION: ELIGIBLE

## Claim
`aiohttp.web.HTTPRequestEntityTooLarge` carries the HTTP status code 413

## What it actually does
`HTTPRequestEntityTooLarge` is defined as `class HTTPRequestEntityTooLarge(HTTPClientError)` with the class attribute `status_code = 413`, overriding the inherited `HTTPClientError.status_code = 400`. Because it subclasses `HTTPException`, which is itself a `Response` subclass, that value is used to build the response (`status=self.status_code` in `HTTPException.__init__`), so an instance carries 413 both as `exc.status_code` and as the status of the response aiohttp sends. The name is re-exported from `aiohttp.web` (explicit `as` re-export plus an entry in `web.__all__`), so `aiohttp.web.HTTPRequestEntityTooLarge` resolves to this class.

## Source
`/Users/alexhubitski/Projects/buster-ripper/.venv/lib/python3.13/site-packages/aiohttp/web_exceptions.py` — lines 313-314 (`class HTTPRequestEntityTooLarge(HTTPClientError):` / `status_code = 413`); supporting: line 80/85/106 (`HTTPException(Response, Exception)`, `status=self.status_code`), line 234/239 (`HTTPClientError` defaults to 400). Export: `aiohttp/web.py` line 69 (`HTTPRequestEntityTooLarge as HTTPRequestEntityTooLarge`) and line 205 (`"HTTPRequestEntityTooLarge"` in `__all__`). No `__init__.py` or other module overrides `status_code` for this class.

## Source Version
aiohttp 3.13.3 (installed source, `.venv/lib/python3.13/site-packages/aiohttp/__init__.py`)

## Source Fingerprint
sha256:ee722e8b0859dfdbc927d2ab5aa02b03941c59b51daa4cf608bc04a496a978df (web_exceptions.py)
sha256:27348d99e8e0e46e987850279088197f2b6a6d4f3ab0dbbce38c982a6a14a6ab (web.py)

## Invalidates On
- `status_code` on `HTTPRequestEntityTooLarge` changes value (e.g. to 422) or the attribute is removed so the inherited `HTTPClientError.status_code = 400` applies.
- `HTTPRequestEntityTooLarge` is re-parented to a class whose `status_code` differs, or `HTTPException.__init__` stops deriving the response status from `status_code`.
- The symbol is dropped from `aiohttp/web.py`'s re-exports/`__all__`, so `aiohttp.web.HTTPRequestEntityTooLarge` no longer resolves to this class.
- The installed aiohttp version's `web_exceptions.py` no longer hashes to the fingerprint above.

## Checked At
2026-10-06T22:20:43Z
