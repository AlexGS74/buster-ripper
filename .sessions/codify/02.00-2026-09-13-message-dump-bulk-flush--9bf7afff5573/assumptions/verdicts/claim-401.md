VERDICT: HOLDS
PROMOTION: ELIGIBLE

## Claim
aiohttp's `web.Request.json()` raises `LookupError`, not `UnicodeDecodeError`, when the request's Content-Type names a charset whose codec Python does not know

## What it actually does
`BaseRequest.json()` is a two-line delegation — `body = await self.text()` then `loads(body)` — and `BaseRequest.text()` ends in `bytes_body.decode(self.charset or "utf-8")`, where `self.charset` is the raw `charset` parameter lifted out of the Content-Type header with no check against Python's codec registry. So an unknown codec name reaches CPython's codec lookup and surfaces as `LookupError: unknown encoding: <charset>`, which propagates out of `json()` untouched (`json()` and `text()` contain no `try`/`except`). `UnicodeDecodeError` belongs to the *other* branch only: a codec Python does know but whose body bytes are malformed for it — and by hierarchy it is `UnicodeDecodeError → UnicodeError → ValueError`, disjoint from `LookupError`, so the two are never alternatives for the same failure.

Precision on the wording "json() raises": the `raise` executes one frame below, inside `text()`. That is not a reading choice that flips the verdict — under either "propagates" or "is the frame containing the raise", the exception a caller of `json()` receives is `LookupError`, and the claim's "not `UnicodeDecodeError`" half is true either way. Recorded only so a later re-check knows which lines to look at.

## Source
Establishing (installed aiohttp source, the supplied bounded root):
- `.venv/lib/python3.13/site-packages/aiohttp/web_request.py` lines 686-695 — `text()` (line 689-690: `encoding = self.charset or "utf-8"` / `return bytes_body.decode(encoding)`) and `json()` (692-695). `Request` subclasses `BaseRequest` (line 842 / 118) and neither method is overridden — `grep "async def text\|async def json"` matches only 686 and 692.
- `.venv/lib/python3.13/site-packages/aiohttp/helpers.py` lines 769-776 — `HeadersMixin.charset` returns `self._content_dict.get("charset")`, i.e. a plain parameter lookup; and lines 388-399 — `parse_content_type()` only splits the header via the email `HeaderParser`, never validating or normalizing the value against a codec list.

Corroborating (CPython, for the unknown-codec exception type):
- `https://docs.python.org/3/library/codecs.html#codecs.lookup` — "If no `CodecInfo` object is found, a `LookupError` is raised", and `codecs.getdecoder()` "Raises a `LookupError` in case the encoding cannot be found"; `codecs.decode()` assigns `UnicodeDecodeError` to *decoding errors* of a registered codec.
- `https://docs.python.org/3/library/exceptions.html#LookupError` — `LookupError` "can be raised directly by `codecs.lookup()`"; `#UnicodeError` — `UnicodeError` "is a subclass of `ValueError`".
- CPython `Python/codecs.c` @ tag `v3.13.0`, `lookup_codec()`: `PyErr_Format(PyExc_LookupError, "unknown encoding: %s", encoding)`; reached from `bytes.decode` via `bytes_decode_impl → PyUnicode_FromEncodedObject → PyUnicode_Decode` (`Objects/bytesobject.c`, `Objects/unicodeobject.c` @ same tag).

Negative check on the same path: every `LookupError` reference in the package is `payload.LookupError` (a distinct error class), netrc/proxy helpers (`helpers.py:249-309`), multipart (`multipart.py:906`), or client-response code (`client.py`, `client_reqrep.py`) — none wraps `BaseRequest.text()`/`json()`. No `UnicodeDecodeError` handler exists in `web_request.py` at all.

## Source Version
aiohttp 3.13.3 (`aiohttp/__init__.py:1`, inside the supplied source root). CPython docs consulted live at the `3.14` docs build plus upstream sources pinned at tag `v3.13.0`; the project's own interpreter binary was not inspected (out of scope for this role).

## Source Fingerprint
sha256:2aaacea7a01e581e5eead2ab1b9e4ba3b65bc2ae3d0f176b2a78ae5b6b76bb4e — aiohttp/web_request.py (whole file; consulted lines 640-749, 842)
sha256:435df4ecf0849d6cf844ff1cad4c3c764e7c734605d848b7272c2429f471cf91 — aiohttp/helpers.py (whole file; consulted lines 380-424, 738-782)

## Invalidates On
- An aiohttp release that wraps `BaseRequest.text()`'s `.decode()` in `try`/`except LookupError`, or that validates/normalizes `charset` against the codec registry before decoding — re-check `web_request.py` lines 686-695 and `helpers.py` `HeadersMixin.charset`.
- An aiohttp release where `json()` stops delegating to `text()` (e.g. decodes through a bytes-capable `loads`), changing which frame raises and with what error.
- A CPython release that reports an unregistered encoding name as `ValueError`/`UnicodeDecodeError` instead of `LookupError` — re-check `lookup_codec()` in `Python/codecs.c` and `Doc/library/codecs.rst`.

## Checked At
2026-10-06
