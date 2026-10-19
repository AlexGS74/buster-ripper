VERDICT: HOLDS
PROMOTION: ELIGIBLE

## Claim
aiohttp's cumulative `ClientTimeout.total` deadline covers consumption of the response payload, not only the wait for response headers

## What it actually does
The single `TimerContext` built from `total` is handed all the way down to the response
`StreamReader`, and the payload reader re-enters that same context on every wait for data.
So the total deadline is live across body consumption, not just header arrival:

1. `client.py:591-595` constructs the handle from `real_timeout.total`, immediately above
   the library's own comment: *"timeout is cumulative for all request operations (request,
   redirects, responses, data consuming)"*.
2. `client.py:622` `timer = tm.timer()`; the header-wait block `with timer:` at `client.py:624`
   is only one consumer. At `client.py:743-744` the *same* object is passed as
   `conn.protocol.set_response_params(timer=timer, ...)` — note `read_timeout=real_timeout.sock_read`
   is passed separately, so the payload timer and `sock_read` are distinct mechanisms.
3. `client_proto.py:237-249` forwards that `timer` into `HttpResponseParser`;
   `http_parser.py:272` stores it and `http_parser.py:397-402` / `419-424` / `435-440`
   passes it into the payload `StreamReader`; `streams.py:166` keeps it as `self._timer`,
   which becomes `ClientResponse.content`.
4. `streams.py:354-372` — `StreamReader._wait()` does `with self._timer:` around
   `await waiter`. `TimerContext.__enter__` (`helpers.py:675-694`) registers the *current task*,
   so a task blocked awaiting body bytes is inside the cumulative deadline's task set and is
   cancelled by `TimerContext.timeout()` (`helpers.py:716-721`), which `__exit__`
   (`helpers.py:696-714`) converts back into `asyncio.TimeoutError`.
5. `streams.py:567-569` — `_read_nowait()` calls `self._timer.assert_timeout()` on every
   buffered read, so reads after the deadline fired raise `asyncio.TimeoutError` too.
6. The deadline itself stays armed through body consumption: `client.py:908-913` does not
   cancel `handle` when `_request` returns; it registers `handle.cancel` as a connection
   release callback (`connector.py:202-223`, fired by `Connection.release()`/`close()`), which
   happens only once the payload is consumed or the response is released.

## Source
`/Users/alexhubitski/Projects/buster-ripper/.venv/lib/python3.13/site-packages/aiohttp/streams.py`
lines 166, 354-372 (`_wait`), 567-569 (`_read_nowait`); with
`/Users/alexhubitski/Projects/buster-ripper/.venv/lib/python3.13/site-packages/aiohttp/client.py`
lines 591-595, 622-624, 743-748, 908-913 and
`/Users/alexhubitski/Projects/buster-ripper/.venv/lib/python3.13/site-packages/aiohttp/helpers.py`
lines 675-721.

Corroborating library documentation (version-pinned to the installed release):
`https://docs.aiohttp.org/en/v3.13.3/_sources/client_reference.rst.txt`, `ClientTimeout.total` —
"Total number of seconds for the whole request." No documentation is installed inside the
package, so the installed source is the establishing locator and the fetched reference page
is corroboration only.

## Source Version
aiohttp 3.13.3 (`__version__ = "3.13.3"` in
`.venv/lib/python3.13/site-packages/aiohttp/__init__.py`), installed at
`/Users/alexhubitski/Projects/buster-ripper/.venv/lib/python3.13/site-packages/aiohttp`

## Source Fingerprint
sha256:ae5c0bede93a0a430c62297f7bf128916bf6eae1e6b738b794aaa59cb36b5c27  (aiohttp/streams.py)
sha256:7ce41fc1c2142f5346015455e030e3ebec22a41cde0fc299a66ce13be2cb2a9e  (aiohttp/client.py)
sha256:435df4ecf0849d6cf844ff1cad4c3c764e7c734605d848b7272c2429f471cf91  (aiohttp/helpers.py)
sha256:e7afd6b4b4ad6460453d82b3804818eafdb8264855d72e891269aec5e253d92a  (aiohttp/client_proto.py)
sha256:cfa763643394b3b85d3f3a654c4b00572cf4a1ffab400c13ef1cfc3a95e19ee6  (aiohttp/http_parser.py)

## Invalidates On
- Removing or narrowing `with self._timer:` in `streams.py::_wait`, or dropping the
  `self._timer.assert_timeout()` call in `_read_nowait`.
- `client.py::set_response_params(...)` no longer receiving the request-level `timer`, or
  `HttpResponseParser`/`StreamReader` no longer threading `timer` into the payload.
- `client.py` cancelling `handle` at `_request` return instead of deferring it to
  `resp.connection.add_callback(handle.cancel)`, which would stop the deadline at headers.
- A refactor that introduces a separate response-body timer in place of `total` (the
  commented-out `response_body_timeout` field in `client.py:202-216`), or an aiohttp version
  bump whose listed files hash differently from the digests above.

## Checked At
2026-10-06T22:21:05Z
