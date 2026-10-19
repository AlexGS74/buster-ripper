VERDICT: DOES-NOT-HOLD
PROMOTION: ELIGIBLE

## Claim
`str()` applied to an aiohttp `StringPayload` yields the text it was constructed with, not an object representation of the payload

## What it actually does
`str()` on a `StringPayload` yields Python's default object representation — e.g.
`<aiohttp.payload.StringPayload object at 0x7f...>` — not the construction text.
Neither `StringPayload` nor any class in its MRO (`StringPayload` → `BytesPayload` →
`Payload(ABC)` → `object`) defines `__str__` or `__repr__`, so `str()` falls through to
`object.__str__`, which delegates to `object.__repr__`. The constructed text is not kept
as a string at all: `StringPayload.__init__` encodes it (`value.encode(real_encoding)`) and
passes the bytes to `BytesPayload`, which stores them in `self._value`. The text is
recoverable only via the synchronous `BytesPayload.decode(encoding="utf-8", errors="strict")`
(`return self._value.decode(encoding, errors)`), the async `await payload.as_bytes()`
(bytes, not text), or direct `_value` access (bytes) — none of which `str()` touches.

## Source
`.venv/lib/python3.13/site-packages/aiohttp/payload.py` — `class Payload(ABC)` (line 151),
`class BytesPayload(Payload)` (line 350) with `decode` (line 380) and `as_bytes` (line 383),
`class StringPayload(BytesPayload)` (lines 430-458); the file contains no `def __str__` and
no `def __repr__` (grep over `payload.py`, `formdata.py`, `multipart.py` returns none).
Exported as `aiohttp.StringPayload` via `aiohttp/__init__.py:88` and `payload.py:47`.

## Source Version
aiohttp 3.13.3 (`aiohttp/__init__.py:1` — `__version__ = "3.13.3"`)

## Source Fingerprint
sha256:3ba9ec60d50b2fb01e33673227a4d85fbde77159cbe71270b7901e831c0c2aac (aiohttp/payload.py, full file bytes)

## Invalidates On
- Any aiohttp release that adds `__str__` or `__repr__` to `Payload`, `BytesPayload`, or `StringPayload` in `payload.py`
- A change to `StringPayload.__init__` that stores the original `str` value instead of only the encoded bytes (changes what `decode()`/`_value` return, not `str()`)
- Replacing this installed package with a different aiohttp version whose `payload.py` bytes differ from the fingerprint above
- A repo-local subclass of `StringPayload` overriding `__str__` — that would make the behaviour true for the subclass only, never for `aiohttp.StringPayload` itself

## Checked At
2026-10-07T02:15:21Z
