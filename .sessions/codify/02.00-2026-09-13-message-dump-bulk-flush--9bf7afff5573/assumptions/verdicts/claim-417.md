VERDICT: HOLDS
PROMOTION: ELIGIBLE

## Claim
aiohttp's `web.Response(body=...)` accepts a `str` without raising in `__init__` or in the `body` property setter

## What it actually does
`Response.__init__` performs no type validation on `body` (its only guard is `body is not None and text is not None`) and delegates to the `body` setter. The setter routes a `str` — which is neither `None` nor `bytes`/`bytearray` — through `payload.PAYLOAD_REGISTRY.get(body)`, and `str` is a registered type (`PAYLOAD_REGISTRY.register(StringPayload, str)`), so no `LookupError` is raised and therefore the `ValueError("Unsupported body type ...")` branch is never reached. Construction succeeds; the only exception path still available is a value-specific encoding failure, not a type rejection.

## Source
`/Users/alexhubitski/Projects/buster-ripper/.venv/lib/python3.13/site-packages/aiohttp/web_response.py` — `Response.__init__` lines 628-691 (`self.body = body` at line 688) and `Response.body` setter lines 697-721. Corroborated in the same package by `payload.py` line 1111 (`PAYLOAD_REGISTRY.register(StringPayload, str)`), `PayloadRegistry.get` lines 110-131 (fast `_normal_lookup` by exact type, `raise LookupError()` only when nothing matches), and `StringPayload.__init__` lines 430-458.

## Source Version
aiohttp 3.13.3 (`aiohttp/__init__.py` line 1: `__version__ = "3.13.3"`), installed source. No documentation ships in the wheel, so the installed source is the ground here; docs on aiohttp readthedocs were not fetched.

## Source Fingerprint
sha256:3ca73388d5382e67ed5ea295be844cac06ce7950a5a5237e8b39c7b2259ece65 (web_response.py)
sha256:3ba9ec60d50b2fb01e33673227a4d85fbde77159cbe71270b7901e831c0c2aac (payload.py)

## Invalidates On
- `Response.body` setter in `web_response.py` changes the `str` branch (e.g. adds an `isinstance(body, str)` rejection or routes `str` off the registry).
- `PAYLOAD_REGISTRY.register(StringPayload, str)` is removed or `str` is re-registered to a factory that raises, in `payload.py`.
- `PayloadRegistry.get` gains a path that raises `LookupError` for an exactly-registered type.
- Any aiohttp upgrade whose `web_response.py`/`payload.py` bytes differ from the fingerprints above — re-check before reuse.

## Caveat
Not raising is a statement about the type, not about every value of it: `StringPayload.__init__` calls `value.encode(real_encoding)` (default `utf-8`), so a `str` containing an unpaired surrogate (e.g. `"\ud800"`) raises `UnicodeEncodeError` out of `__init__`. A `str` above `TOO_LARGE_BYTES_BODY` triggers a `ResourceWarning`, not an exception.

## Checked At
2026-10-06
