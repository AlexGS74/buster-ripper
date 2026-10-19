VERDICT: HOLDS
PROMOTION: ELIGIBLE

## Claim
aiohttp's `ClientResponse.read()` hands over payload bytes without applying the response's charset

## What it actually does
`read()` is byte-level only. It fills the body cache from the stream reader — `self._body = await self.content.read()` — and returns `self._body`, annotated `-> bytes`. The function body never touches `self.headers`, never parses `Content-Type`, never calls `get_encoding()`, `codecs.lookup`, or `.decode()`; `StreamReader.read()` (`streams.py:414`) likewise returns `bytes`. Charset handling lives entirely in the separate `text()`/`json()` layer, which calls `await self.read()` first and then decodes.

Distinction worth recording for a later re-check: `read()` does return *decompressed* bytes when `auto_decompress` is on, but that is `Content-Encoding` (gzip/deflate) applied at the transport layer during streaming, not the `Content-Type` charset. Decompression is not charset application, so it does not qualify this claim.

## Source
`/Users/alexhubitski/Projects/buster-ripper/.venv/lib/python3.13/site-packages/aiohttp/client_reqrep.py` lines 689–707 (`ClientResponse.read`), contrasted with lines 732–740 (`ClientResponse.text`); `/Users/alexhubitski/Projects/buster-ripper/.venv/lib/python3.13/site-packages/aiohttp/streams.py` line 414 (`StreamReader.read` returns `bytes`). Corroborated by upstream `client_reference.rst`: `read()` — "Read the whole response's body as `bytes`."

## Source Version
aiohttp 3.13.3 (installed, `.venv/lib/python3.13/site-packages/aiohttp-3.13.3.dist-info/METADATA`)

## Source Fingerprint
sha256:7844440eb67433c6454edd708c776e47e3fcfec9b8d0aeb980dcfe88c1986ac9 (client_reqrep.py, full file bytes)

## Invalidates On
- The installed `client_reqrep.py` digest changes and `ClientResponse.read` begins referencing `get_encoding()`, the `Content-Type` header, or any decode/codec call.
- `read()`'s return annotation or body changes from raw `self._body` bytes to a decoded or charset-normalised value.
- `response_class=`/`_body` is overridden so the cached value is not the raw stream bytes.

## Checked At
2026-10-06
