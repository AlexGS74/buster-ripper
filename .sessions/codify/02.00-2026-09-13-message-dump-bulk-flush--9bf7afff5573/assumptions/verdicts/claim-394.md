VERDICT: HOLDS
PROMOTION: ELIGIBLE

## Claim
aiohttp raises `web.HTTPRequestEntityTooLarge` while reading a request body that exceeds the application's configured `client_max_size`

## What it actually does
While accumulating the request body, `Request.read()` compares the running length against the request's `_client_max_size` and raises `HTTPRequestEntityTooLarge(max_size=..., actual_size=...)` once the body reaches it. That value is the application's configured one: `Application.__init__` stores `client_max_size` and `Application._make_request` passes `client_max_size=self._client_max_size` into every `Request`. The same exception is raised from both multipart branches of `Request.post()` (`0 < max_size < size`), and `text()`/`json()` inherit the check because they go through `read()`. The exception then escapes the handler and `web_protocol.py` catches `HTTPException` and uses it directly as the response.

Two precisions that do not overturn the claim: the comparison is `>=`, so a body whose size is exactly equal to `client_max_size` also raises (not only one that strictly exceeds it); and the check lives in the `Request` body API only — reading `request.content` / `request.payload` directly bypasses `client_max_size` entirely.

## Source
`/Users/alexhubitski/Projects/buster-ripper/.venv/lib/python3.13/site-packages/aiohttp/web_request.py` — `Request.read()` lines 670-684 (`raise HTTPRequestEntityTooLarge(max_size=self._client_max_size, actual_size=body_size)` at 678-680), `Request.post()` lines 720-722, 746-750, 774-777. Propagation: `aiohttp/web_app.py` line 128/175 (stored, unvalidated) and line 510 (`_make_request` → `client_max_size=self._client_max_size`). Response conversion: `aiohttp/web_protocol.py` lines 513-515. `grep` for `HTTPRequestEntityTooLarge` across the package shows `web_request.py` as the only raiser.

## Source Version
aiohttp 3.13.3 (installed source, `.venv/lib/python3.13/site-packages/aiohttp/__init__.py`)

## Source Fingerprint
sha256:2aaacea7a01e581e5eead2ab1b9e4ba3b65bc2ae3d0f176b2a78ae5b6b76bb4e (web_request.py)
sha256:94653f68030dfa1df0cbe2d34c78ba49e287f3276dd46e75057702b29803e590 (web_app.py)

## Invalidates On
- The `>=` comparison or the `raise` in `Request.read()` is replaced (e.g. a different exception type, a `413` response returned instead of raised, or truncation instead of an error).
- `Application._make_request` stops forwarding `client_max_size`, or `Request.__init__` starts overriding it with a per-request/default value.
- Enforcement moves to the parser or protocol layer (`http_parser.py` / `web_protocol.py`) so the error is raised before the handler reads the body.
- `Request.content`/`payload` starts applying `client_max_size`, changing what "reading a request body" covers.
- The installed aiohttp version's `web_request.py` no longer hashes to the fingerprint above.

## Checked At
2026-10-06T22:20:43Z
