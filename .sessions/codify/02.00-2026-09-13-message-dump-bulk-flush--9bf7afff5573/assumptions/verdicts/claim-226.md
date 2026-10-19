VERDICT: HOLDS
PROMOTION: ELIGIBLE — holds verdict, one exact tagged source locator, sha256 of consulted bytes, concrete invalidation conditions below

## Claim
aiohttp's `web.StreamResponse.prepare()` sends the response head for that object on its first call

## What it actually does
For a `StreamResponse` whose head has not been written yet (`_eof_sent` false and `_payload_writer` is None), `prepare(request)` runs `_start(request)`, which binds `self._payload_writer = request._payload_writer`, computes default headers, fires the `on_response_prepare` hook, and calls `_write_headers()` — which serializes the status line plus headers into the stream writer and, because `StreamResponse._send_headers_immediately` is `True`, flushes them to the transport. So the first call does put the response head on the wire for that object.

## Evidence
`aiohttp/web_response.py` @ v3.13.3:

- L454-460 `prepare()` — first guard-by-guard path: `if self._eof_sent: return None`; `if self._payload_writer is not None: return self._payload_writer`; otherwise `return await self._start(request)`.
- L462-470 `_start()` — `self._req = request`; `writer = self._payload_writer = request._payload_writer`; `await self._prepare_headers()`; `await request._prepare_hook(self)`; `await self._write_headers()`.
- L537-548 `_write_headers()` — builds `status_line = f"HTTP/{v[0]}.{v[1]} {self._status} {self._reason}"`, `await writer.write_headers(status_line, self._headers)`, then `if self._send_headers_immediately: writer.send_headers()`.
- L92 class default `_send_headers_immediately = True` on `StreamResponse`; the only override in the package is L626 `Response._send_headers_immediately = False` (grep across `web_response.py`/`web_protocol.py`/`abc.py`/`http_writer.py` found no other assignment and no public setter).
- `aiohttp/http_writer.py` @ v3.13.3 L210-234 — `write_headers()` serializes with `_serialize_headers()` into `self._headers_buf` (buffered, not yet transported); `send_headers()` does `self._write(headers_buf)` — the actual transport push. Both happen inside the first `prepare()` for a plain `StreamResponse`, so the claim holds under either reading of "sends".

Corroborating documentation (same version): `web_reference.html` @ v3.13.3, `StreamResponse.prepare()` — "Send HTTP header. You should not change any header data after calling this method. The coroutine calls on_response_prepare signal handlers after default headers have been computed and directly before headers are sent." Also `StreamResponse.prepared` — "True if prepare() has been called".

Scope caveat, stated because it is the same method on a subclass: a `web.Response` (e.g. from `json_response`) inherits this `prepare()`, but with `_send_headers_immediately = False` the head is *buffered* at `prepare()` time and only reaches the transport on the first body `write()`/`write_eof()` (`StreamWriter.send_headers()` / `set_eof()`). That is a different subject object than the one the claim names, not a competing reading of it.

## Source
https://raw.githubusercontent.com/aio-libs/aiohttp/v3.13.3/aiohttp/web_response.py (L92, L454-470, L537-548); corroborated by https://raw.githubusercontent.com/aio-libs/aiohttp/v3.13.3/aiohttp/http_writer.py (L210-234) and https://docs.aiohttp.org/en/v3.13.3/web_reference.html

## Source Version
aiohttp 3.13.3 (upstream tag `v3.13.3`; matches the version pinned in this repo's `uv.lock`)

## Source Fingerprint
sha256:3ca73388d5382e67ed5ea295be844cac06ce7950a5a5237e8b39c7b2259ece65 (web_response.py)
sha256:7db46d28f612a916ed01daff82aa63176fb8b08d4448bf1d3c317ec58fe600c6 (http_writer.py)
sha256:638fae090c8b220a96fae8fef2e44bac2aebf7cf3ac15a952ad8381b1938e48f (web_reference.html)

## Invalidates On
- Any change to `StreamResponse._start` / `_write_headers` so the head is no longer written during the first `prepare()` (e.g. `StreamResponse._send_headers_immediately` default flipped to `False`, or L547-548 guarded differently).
- aiohttp version bump whose fingerprint differs from `3ca73388…` above — the header-buffering split between `write_headers()` and `send_headers()` is recent and behaviour-visible.
- Introduction of a public API that sets `_send_headers_immediately` on a plain `StreamResponse`, which would make "first call sends the head" default-dependent rather than unconditional.

## Checked At
2026-10-05T18:18:15Z
