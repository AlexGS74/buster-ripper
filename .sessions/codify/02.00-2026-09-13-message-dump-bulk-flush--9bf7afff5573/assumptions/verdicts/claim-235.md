VERDICT: HOLDS
PROMOTION: ELIGIBLE

## Claim
aiohttp's client session computes `Content-Length` for a request body it sends

## What it actually does
`ClientRequest.update_body_from_data()` derives the length from the payload it
just built and writes it into the request headers:
`if not self.chunked and hdrs.CONTENT_LENGTH not in self.headers:` → take
`payload.size`, and when it is not `None`, set `Content-Length: str(size)`. The
same call also sets `Content-Length: 0` for a body-less request whose method is
not in `GET_METHODS` (and is not chunked). So a caller passing `data=` gets a
computed `Content-Length` without supplying one, and that computed value is what
the writer later enforces: `write()` calls `self._get_content_length()` and passes
it to `write_bytes(...)` → `_body.write_with_length(writer, content_length)`.
Boundary of the "computes" scope, stated so the verdict is not read as broader
than the source: the computation happens **when the payload size is knowable** —
bytes/bytearray/memoryview (`len`), `str` (encoded length), `BytesIO`
(buffer minus tell), and file-like objects (`os.fstat(...).st_size - start`).
When `payload.size` is `None` (async iterables / `StreamReader` / unseekable
streams) aiohttp sets `self.chunked = True` and sends
`Transfer-Encoding: chunked` with **no** `Content-Length`; likewise `compress=`
forces `chunked = True` in `update_content_encoding()`. A caller-supplied
`Content-Length` is respected rather than recomputed (and combining it with
`chunked=True` raises `ValueError` in `update_transfer_encoding()`);
replacing the body via `_update_body()` deletes and recomputes it.

## Source
`aiohttp/client_reqrep.py` @ tag `v3.13.3` — `ClientRequest.update_body_from_data`
(lines 1173-1212, specifically 1178-1187 and 1199-1203), `_get_content_length`
(lines 911-926), `update_transfer_encoding` (1142-1159), `_update_body`
(1214-1230), `write`/`write_bytes` (1330-1377, 1473-1474).
Supporting: `aiohttp/payload.py` @ `v3.13.3` — `Payload.size` (190-197,
`_size = None` default at 154), `BytesPayload` size (364-366),
`IOBasePayload.size` (543-566), `BytesIOPayload.size` (853-862),
`AsyncIterablePayload` (971-991, no size override → `None`), registry (1110-1120);
`update_content_encoding` chunked-on-compress in `client_reqrep.py` 1124-1140.

Source basis: upstream source fetched at the exact version tag pinned in this
project's `uv.lock` (`aiohttp 3.13.3`). No local `Source root` /
`Dependency source roots` was supplied with the dispatch, so the installed copy
was not read; the fetched files are the consulted bytes.

## Source Version
aiohttp 3.13.3 (`v3.13.3` tag; matches the pin in `uv.lock`)

## Source Fingerprint
sha256:7844440eb67433c6454edd708c776e47e3fcfec9b8d0aeb980dcfe88c1986ac9 (client_reqrep.py)
sha256:3ba9ec60d50b2fb01e33673227a4d85fbde77159cbe71270b7901e831c0c2aac (payload.py)

## Invalidates On
- Any aiohttp version where `update_body_from_data` stops assigning
  `hdrs.CONTENT_LENGTH` from `payload.size`, or reverses the
  "size known → Content-Length, size unknown → chunked" branch.
- Any change to `Payload.size` semantics for the concrete payload types above
  (e.g. `BytesPayload` reporting `None`), since the header is derived solely
  from that property.
- Any version where `compress=` no longer forces chunked encoding (that would
  restore `Content-Length` for compressed bodies).
- Reading these claims against a pinned version other than 3.13.3 without
  re-consulting these files.

## Checked At
2026-10-05T14:26:00+10:00
