VERDICT: HOLDS
PROMOTION: ELIGIBLE

## Claim
An aiohttp `web.Response` constructed with a `str` passed to `body=` sends that string's encoded text over the wire, not a serialization of an internal payload object

## What it actually does
`Response.__init__` with `text=None` assigns through the `body` setter, which — for a `str` (not `bytes`/`bytearray`) — calls `payload.PAYLOAD_REGISTRY.get(body)`; `str` is registered to `StringPayload`, so `self._body` becomes a `StringPayload` whose constructor *immediately* replaces the string with `value.encode(real_encoding)` (utf-8, since the setter passes no `encoding`/`content_type`) and stores it as `_value`. At send time `Response.write_eof` takes the `isinstance(self._body, Payload)` branch and calls `self._body.write(self._payload_writer)`, which is `BytesPayload.write` → `await writer.write(self._value)` — i.e. the encoded bytes of the original string are written verbatim. The payload object itself is never serialized; it only supplies bytes plus `Content-Type: text/plain; charset=utf-8` (copied in the setter when no Content-Type is set) and `Content-Length` from `payload.size` in `Response._start`.

Both halves of the claim are settled by the same path (encoded text yes, payload-object serialization no), so the row is answered whole rather than partially.

Caveats read out of the same source, none of which reverse the verdict:
- Encoding for the plain `Response(body=s)` path is always utf-8 regardless of any `content_type=`/`Content-Type` the caller supplies; a caller-declared non-utf-8 charset therefore mislabels utf-8 bytes but still does not serialize the payload object.
- If compression is enabled, `_do_start_compression` replaces the transmitted bytes with the deflate/gzip form of `self._body` — a transport encoding of the same text, not a payload serialization.

## Source
/Users/alexhubitski/Projects/buster-ripper/.venv/lib/python3.13/site-packages/aiohttp/web_response.py — `Response.__init__` L628-691, `body` setter L697-721, `write_eof` L769-786, `_start` L788-804, `_do_start_compression` L806-830
/Users/alexhubitski/Projects/buster-ripper/.venv/lib/python3.13/site-packages/aiohttp/payload.py — `PAYLOAD_REGISTRY.register(StringPayload, str)` L1111, `PayloadRegistry.get` L110-131, `StringPayload.__init__` L430-458, `BytesPayload.write` L392-407

## Source Version
aiohttp 3.13.3 (`__version__` in .venv/lib/python3.13/site-packages/aiohttp/__init__.py); installed source, no bundled documentation

## Source Fingerprint
sha256:3ca73388d5382e67ed5ea295be844cac06ce7950a5a5237e8b39c7b2259ece65 (web_response.py)
sha256:3ba9ec60d50b2fb01e33673227a4d85fbde77159cbe71270b7901e831c0c2aac (payload.py)

## Invalidates On
- web_response.py bytes change such that the `body` setter re-gains a `str` rejection (`TypeError: body argument must be bytes`) or stops routing through `PAYLOAD_REGISTRY`
- payload.py bytes change such that `str` is no longer registered to `StringPayload`, or `StringPayload` stops encoding eagerly / `BytesPayload.write` stops writing `self._value`
- aiohttp version moves to a release where `Response.write_eof` sends payloads via `write_with_length` or another writer path that frames the body differently
- re-checking against a different installed copy of aiohttp than the fingerprinted site-packages root above

## Checked At
2026-10-06
