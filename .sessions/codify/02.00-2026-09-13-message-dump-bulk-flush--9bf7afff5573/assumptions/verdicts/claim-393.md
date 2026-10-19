VERDICT: HOLDS
PROMOTION: ELIGIBLE

## Claim
aiohttp skips the request-body size check whenever the configured `client_max_size` is falsy, so `0` imposes no limit

## What it actually does
The size guard in `Request.read()` is wrapped in a truthiness test — `if self._client_max_size:` — so with `client_max_size=0` the accumulated body length is never compared and `HTTPRequestEntityTooLarge` is never raised; the body is read to completion without bound. Both body-accumulation paths in `Request.post()` are guarded the same way (`if 0 < max_size < size:`), so `0` disables the check there too. `Application.__init__` stores `client_max_size` with no validation and passes it straight to each `Request`, so `0` reaches the guard unchanged, and `Application.client_max_size` / `Request.client_max_size` report `0`. No other component caps total request-body size: the `limit` handed to the payload `StreamReader` becomes `_low_water`/`_high_water` backpressure marks only (`self._limit` exists solely on `DataQueue`), not a hard overflow check.

Note (not part of the claim, but bounds its scope): `0` is the only falsy `int`, so this is exactly the `0` case. A *negative* value is truthy, so the guard runs and `body_size >= self._client_max_size` is satisfied on the first chunk — a negative limit rejects every request that has a body rather than removing the limit.

## Source
`/Users/alexhubitski/Projects/buster-ripper/.venv/lib/python3.13/site-packages/aiohttp/web_request.py` — `Request.read()` lines 670-684 (`if self._client_max_size:` / `if body_size >= self._client_max_size:`), `Request.post()` lines 722 and 746-750 and 774-777 (`if 0 < max_size < size:`); corroborating plumbing in `aiohttp/web_app.py` lines 128/175/510 (no validation, direct propagation) and `aiohttp/streams.py` lines 150-155 (parser `limit` → water marks only). `HTTPRequestEntityTooLarge` is raised from only these three sites in the package.

## Source Version
aiohttp 3.13.3 (installed source, `.venv/lib/python3.13/site-packages/aiohttp/__init__.py`)

## Source Fingerprint
sha256:2aaacea7a01e581e5eead2ab1b9e4ba3b65bc2ae3d0f176b2a78ae5b6b76bb4e (web_request.py)
sha256:94653f68030dfa1df0cbe2d34c78ba49e287f3276dd46e75057702b29803e590 (web_app.py)

## Invalidates On
- The `if self._client_max_size:` guard in `Request.read()` becomes an unconditional or `is not None` comparison, or any new `client_max_size` enforcement site appears outside `web_request.py`.
- `Application.__init__` or `Request.__init__` starts rejecting/normalizing `0` (e.g. `if client_max_size <= 0: raise ValueError`) or substitutes a default when given `0`.
- The payload `StreamReader` starts enforcing its `limit` as a total-body cap (raising on overflow) rather than using it only for `_low_water`/`_high_water` backpressure.
- The installed aiohttp version's `web_request.py` no longer hashes to the fingerprint above.

## Checked At
2026-10-06T22:20:43Z
