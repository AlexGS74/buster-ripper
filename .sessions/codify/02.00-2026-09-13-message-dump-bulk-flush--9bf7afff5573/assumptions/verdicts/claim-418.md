VERDICT: DOES-NOT-HOLD
PROMOTION: ELIGIBLE

## Claim
On an aiohttp `web.Response` constructed with a `str` passed to `body=`, reading the `.body` attribute returns `bytes` rather than a payload object

## What it actually does
It returns a **payload object**, not bytes. `Response.body` getter is a bare `return self._body`, and the setter's `str` branch stores the registry result directly: `self._body = body = payload.PAYLOAD_REGISTRY.get(body)`. For a `str` that result is a `StringPayload` instance (`PAYLOAD_REGISTRY.register(StringPayload, str)`), which subclasses `BytesPayload` → `Payload`, so `isinstance(resp.body, bytes)` is `False`. The getter's own annotation says as much: `-> Optional[Union[bytes, Payload]]`, and downstream code in `Response` branches on `isinstance(self._body, Payload)` rather than treating it as bytes. Only values that were already `bytes`/`bytearray` are stored raw (setter line 701-702). The encoded bytes are reachable as `resp.body.decode()` (returns `str`), `await resp.body.as_bytes()` (returns `bytes`), or the private `resp.body._value`; `resp.body.size` gives the byte count.

## Source
`/Users/alexhubitski/Projects/buster-ripper/.venv/lib/python3.13/site-packages/aiohttp/web_response.py` — `Response.body` getter lines 693-695 and setter lines 697-721 (`self._body = body = payload.PAYLOAD_REGISTRY.get(body)` at line 705). Corroborated in the same package by `payload.py` line 1111 (registry binding `str` → `StringPayload`), `class StringPayload(BytesPayload)` line 430, `class BytesPayload(Payload)` line 350 with `decode` line 380-381 and `as_bytes` line 383-390.

## Source Version
aiohttp 3.13.3 (`aiohttp/__init__.py` line 1: `__version__ = "3.13.3"`), installed source. No documentation ships in the wheel.

## Source Fingerprint
sha256:3ca73388d5382e67ed5ea295be844cac06ce7950a5a5237e8b39c7b2259ece65 (web_response.py)
sha256:3ba9ec60d50b2fb01e33673227a4d85fbde77159cbe71270b7901e831c0c2aac (payload.py)

## Invalidates On
- `Response.body` setter starts normalising `str` to `text.encode(...)` bytes before storing (i.e. the `isinstance(body, (bytes, bytearray))` branch is widened or a `str` branch is added).
- `PAYLOAD_REGISTRY` stops mapping `str` to a `Payload` subclass, or `get` is changed to return encoded bytes.
- `Response.body` getter is changed to unwrap/serialise `_body` instead of returning it verbatim.
- Any aiohttp upgrade whose `web_response.py`/`payload.py` bytes differ from the fingerprints above — re-check before reuse.

## Checked At
2026-10-06
