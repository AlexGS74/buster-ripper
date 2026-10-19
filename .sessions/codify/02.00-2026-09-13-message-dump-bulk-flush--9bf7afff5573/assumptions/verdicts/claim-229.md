VERDICT: HOLDS
PROMOTION: ELIGIBLE

## Claim
Preparing a second, distinct response object for a request whose first response is already committed writes a second status line and header block onto that connection

## What it actually does
`StreamResponse.prepare()` guards only against re-preparing *itself*, never against the request already having a committed response, so a fresh response object binds the request's original `StreamWriter` and re-serializes a new status line plus header block into it; `StreamWriter.write_headers()` has no "headers already written" guard — it resets `_headers_written = False` and replaces the buffered header bytes, which `send_headers()` then pushes to the transport.

Mechanism, in order, at aiohttp 3.13.3:

1. `web_response.py` `prepare()`: `if self._eof_sent: return None` / `if self._payload_writer is not None: return self._payload_writer`. Both are per-response-object state. A brand-new, distinct response fails neither test and falls through to `_start(request)`.
2. `web_response.py` `_start()`: `writer = self._payload_writer = request._payload_writer`. The request hands out the *same* writer the first response used: `web_request.py:169` assigns `_payload_writer` once in `BaseRequest.__init__` and nothing ever clears it (`_finish()` at `web_request.py:829` only closes multipart temp files); `web_protocol.py` `start()` builds exactly one `StreamWriter(self, loop)` per request message and `finish_response()` never nulls it.
3. `_start()` → `_write_headers()` builds `status_line = f"HTTP/{version[0]}.{version[1]} {self._status} {self._reason}"` and calls `await writer.write_headers(status_line, self._headers)`.
4. `http_writer.py:214-219` `write_headers()`: `buf = _serialize_headers(status_line, headers)` (status line + all headers + terminating `\r\n\r\n`), then `self._headers_written = False; self._headers_buf = buf`. No check for a prior header block on this writer.
5. `send_headers()` (`http_writer.py:221-232`) bails only on `if not self._headers_buf or self._headers_written` — step 4 just made both false — then `self._write(headers_buf)` → `transport.write(chunk)`.

On the wire this means two committed responses on one connection, with no error raised anywhere in `prepare()`.

Delivery timing differs by class, and this is the only qualification on the claim:
- `StreamResponse` (`_send_headers_immediately = True`): the second status line + header block reaches the transport *inside* `await resp2.prepare(request)`. `send_headers()` has no `_eof` check, so this is true even if the first response already sent `write_eof()`.
- `Response` (`_send_headers_immediately = False`): the block is buffered in `_headers_buf` and goes out with the next `write()`/`write_eof()` on that writer — which is exactly what aiohttp's own `finish_response()` does (`await prepare_meth(request); await resp.write_eof()`). Swallow corner: if the first response already drove the writer to `_eof`, `set_eof`/`write_eof` return early at `if self._eof` (`http_writer.py:238`, `:264`), so a *buffered* `Response` header block never reaches the wire. The buffer is still overwritten, and any `StreamResponse` second prepare still writes through.

Contrast worth recording: aiohttp's *own* error path does guard against this — `web_protocol.py` `handle_error()` raises `ConnectionError("Response is sent already, cannot send another response with the error message")` when `request.writer.output_size > 0`. The absence of such a guard is specific to application code calling `prepare()` on a second response itself.

## Source
`https://raw.githubusercontent.com/aio-libs/aiohttp/v3.13.3/aiohttp/web_response.py` (`prepare`, `_start`, `_write_headers`, `_send_headers_immediately` on both classes), with the no-guard step in `aiohttp/http_writer.py:214-232` and the surviving request writer in `aiohttp/web_request.py:169` / `:829` — same tag. Source basis: upstream source at the version pinned by the caller's `uv.lock` (aiohttp 3.13.3); installed-source directory was not supplied, so the exact tag was fetched instead. No test file was used (P10); nothing was executed (P1).

## Source Version
aiohttp 3.13.3 (upstream tag `v3.13.3`; pinned by `uv.lock` line 15ff)

## Source Fingerprint
web_response.py: sha256:3ca73388d5382e67ed5ea295be844cac06ce7950a5a5237e8b39c7b2259ece65
http_writer.py: sha256:7db46d28f612a916ed01daff82aa63176fb8b08d4448bf1d3c317ec58fe600c6
web_request.py: sha256:2aaacea7a01e581e5eead2ab1b9e4ba3b65bc2ae3d0f176b2a78ae5b6b76bb4e

## Invalidates On
- aiohttp upgraded past 3.13.3 with any change to `StreamResponse.prepare`/`_start` request-writer binding
- any change to `StreamWriter.write_headers`/`send_headers` buffering introduced in 3.13.x (`_headers_buf`, `_headers_written`, `_eof` early returns)
- any aiohttp version that adds a "response already committed" guard to `prepare()` or clears `BaseRequest._payload_writer` after `write_eof()`

## Checked At
2026-10-05
