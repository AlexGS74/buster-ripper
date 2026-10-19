VERDICT: HOLDS
PROMOTION: ELIGIBLE

## Claim
With aiohttp's `auto_decompress` enabled, `ClientResponse.read()` returns the decompressed payload rather than the wire bytes

## What it actually does
When `auto_decompress` is on and the response carries a recognized `Content-Encoding` (`gzip`, `deflate`, `br`, `zstd`), the payload parser wraps the `StreamReader` in a `DeflateBuffer`: raw wire chunks are fed to the wrapper, decompressed, and only the decompressed bytes are pushed into the inner `StreamReader` that `ClientResponse.content` exposes. `ClientResponse.read()` awaits `self.content.read()` on that inner stream, so it returns (and caches in `self._body`) decompressed bytes. Wire-size accounting is kept separately on the stream as `total_compressed_bytes`, which is the source's own acknowledgement that the buffered content is not the compressed input. If `auto_decompress` is disabled, or the encoding is absent/unrecognized, `read()` returns the raw wire bytes instead.

## Source
`.venv/lib/python3.13/site-packages/aiohttp/http_parser.py` — `HttpPayloadParser.__init__` lines 798-804 (`if response_with_body and compression and self._auto_decompress: real_payload = DeflateBuffer(payload, compression)`), `DeflateBuffer.feed_data` lines 1007-1048 (`chunk = self.decompressor.decompress_sync(...)` then `self.out.feed_data(chunk, len(chunk))`), and `HttpParser.feed_data` line 458 (`messages.append((msg, payload))` — the message tuple carries the **inner** stream, not the wrapper). Consumed at `client_reqrep.py` line 539 (`message, payload = await protocol.read()`), line 570 (`self.content = payload`), lines 689-707 (`async def read(...): self._body = await self.content.read()`). The compiled parser actually used by this install (`_http_parser.cpython-313-darwin.so` present) shows the same wiring in `_http_parser.pyx` lines 460-473 (`payload = StreamReader(...)`; `self._payload = payload`; `if encoding is not None and self._auto_decompress: self._payload = DeflateBuffer(payload, encoding)`; `self._messages.append((msg, payload))`). No installed docs ship with the package, so the verdict is grounded in the installed source (rank 2), both parser implementations.

## Source Version
aiohttp 3.13.3 (`aiohttp-3.13.3.dist-info/METADATA` — installed package source)

## Source Fingerprint
sha256:cfa763643394b3b85d3f3a654c4b00572cf4a1ffab400c13ef1cfc3a95e19ee6 http_parser.py
sha256:f9823c608638b8a77fec1c2bd28dc5c043f08c775ec59e7e262dd74b4ebb7384 _http_parser.pyx
sha256:7844440eb67433c6454edd708c776e47e3fcfec9b8d0aeb980dcfe88c1986ac9 client_reqrep.py

## Invalidates On
- aiohttp version change that alters `DeflateBuffer` wrapping or moves decompression out of the payload parser (e.g. to a `content`-property hook or response middleware)
- a release where `messages.append((msg, payload))` hands the wrapper rather than the inner `StreamReader` to `ClientResponse.content`
- `ClientResponse.read()` reimplemented to bypass `self.content` / to read `raw_headers`-paired wire bytes
- a change to the recognized-encoding set at `http_parser.py:557` / `_http_parser.pyx:438` that stops matching the encoding actually used on the wire
- running with `AIOHTTP_NO_EXTENSIONS=1` in a build whose pure-Python parser diverges from the Cython one

## Checked At
2026-10-06
