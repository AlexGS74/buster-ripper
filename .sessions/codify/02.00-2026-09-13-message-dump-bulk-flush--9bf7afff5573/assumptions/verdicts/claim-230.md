VERDICT: HOLDS
PROMOTION: ELIGIBLE

## Claim
When an aiohttp request handler raises, the aiohttp server sends its own generated error response for that request

## What it actually does
The exception is caught in `RequestHandler._handle_request()`, which substitutes a server-built response and then runs it through the normal send path — `finish_response()` does `await prepare_meth(request)` followed by `await resp.write_eof()` — so the connection gets an aiohttp-generated error response for that request rather than an aborted turn.

Dispatch at aiohttp 3.13.3 (`aiohttp/web_protocol.py`):
- `:513 except HTTPException as exc: resp = exc` → the raised exception *is* the response (its own status/text/headers) and is sent.
- `:518 except asyncio.TimeoutError` → logs "Request handler timed out." then `resp = self.handle_error(request, 504)` and sends it.
- `:522 except Exception as exc: resp = self.handle_error(request, 500, exc)` and sends it.
- A handler that returns `None` or a non-response is also answered with a generated 500 (`finish_response()`: "Missing return statement on request handler" / "Web-handler should return a response instance" → `HTTPInternalServerError`).

What `handle_error()` (`:722`) generates for the plain-exception case: it logs `"Error handling request from %s"` with `exc_info`, then builds `Response(status=status, text=message, content_type=ct)` itself — body `"500 Internal Server Error\n\n<HTTPStatus description>"` as `text/plain`, or an `<html><h1>500 Internal Server Error</h1>…` document as `text/html` when the request's `Accept` contains `text/html`; with `debug=True` the traceback is substituted into the body (escaped for the HTML variant). `resp.force_close()` is called, so the docstring's "It always closes current connection" holds — the error response is sent and then the connection is closed (also relevant: `_keep_alive` becomes false because `resp.keep_alive` is False).

Carve-outs, stated so the verdict is not read as unconditional:
- `asyncio.CancelledError` is re-raised before any response (`:516`), producing no error response.
- A `BaseException` that is not an `Exception` (e.g. `KeyboardInterrupt`, `SystemExit`) escapes `_handle_request` to `start()`'s `except BaseException: self.force_close(); raise` (`:655`) — connection closed, no response.
- If bytes have already gone out for that request, `handle_error()` raises `ConnectionError("Response is sent already, cannot send another response with the error message")` (`:749`, guard is `request.writer.output_size > 0`), caught in `start()` at `:603` as "Ignored premature client disconnection" → `break`. So the server does *not* emit a second header block on an already-committed response; the error turn is dropped and the connection closed.
- Errors raised by aiohttp's own parsing (bad request line/headers) take a separate route: `_ErrInfo` messages and `self._make_error_handler(...)` produce a 400 through the same `handle_error()`.

## Source
`https://raw.githubusercontent.com/aio-libs/aiohttp/v3.13.3/aiohttp/web_protocol.py` — `_handle_request` (`:513-523`), `finish_response`, `handle_error` (`:722-780`), `start` exception handling (`:603`, `:652-656`). Source basis: upstream source at the version pinned by the caller's `uv.lock` (aiohttp 3.13.3); no installed-source root was supplied, so the exact tag was fetched instead. Documentation was not needed — the dispatch is explicit in the protocol loop. No test file was used (P10); nothing was executed (P1).

## Source Version
aiohttp 3.13.3 (upstream tag `v3.13.3`; pinned by `uv.lock` line 15ff)

## Source Fingerprint
web_protocol.py: sha256:eacf5d33399aa96efb6f33354f5d75b863522c5a3a80d99f0e0ed7cd89c0f319

## Invalidates On
- aiohttp upgraded past 3.13.3 with any change to `RequestHandler._handle_request`'s except-clause ordering or to `handle_error()`'s response construction
- any change to the `request.writer.output_size > 0` guard or the `ConnectionError` route in `start()` that makes an already-committed request receive (or not receive) an error turn
- removal/renaming of `finish_response()`'s prepare + `write_eof()` path, or `handler_cancellation` changes altering `CancelledError` handling

## Checked At
2026-10-05T18:20:15Z
