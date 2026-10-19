VERDICT: DOES-NOT-HOLD
PROMOTION: ELIGIBLE

## Claim
For an aiohttp `web.Response` built with a `str` `body=`, the payload its `body` getter returns decodes back to that same string under the response's own charset

## What it actually does
`body=` with a `str` always encodes the string as **utf-8**, independently of the charset the response declares; `Response.body` returns a `StringPayload` whose bytes are utf-8, while `Response.charset` reports the Content-Type parameter the caller asked for. So the round trip holds only when the response's charset resolves to utf-8 — which is the case for the plain construction `Response(body=s)` (nothing declared → the payload supplies `text/plain; charset=utf-8`) — and breaks for any non-utf-8 charset reaching Content-Type: `StringPayload.decode` is inherited `BytesPayload.decode` → `self._value.decode(encoding, "strict")`, which yields mojibake for a superset charset (`Response(body="café", content_type="text/plain", charset="iso-8859-1").body.decode(...charset)` → `"cafÃ©"`) and raises `UnicodeDecodeError` where the utf-8 bytes are not representable (`charset="ascii"`), with non-ASCII text. ASCII-only text masks the break under any single-byte charset. Two further gaps in the claim as written: `Response(body=s, content_type="text/html")` leaves `response.charset` as `None`, so there is no "response's own charset" to decode with — `Response.text` hides that with a `self.charset or "utf-8"` fallback — and `Response(body=s, charset="latin-1")` (charset without content_type) silently **drops** the charset argument entirely (the `elif content_type is not None:` branch is not entered), so the header ends up utf-8.

Corroborating deviation: the stable docs state the opposite intent — under `Response.body`: "Assigning `str` to `body` will make the `body` type of `aiohttp.payload.StringPayload`, which tries to encode the given data based on *Content-Type* HTTP header, while defaulting to `UTF-8`." The installed source never passes encoding or Content-Type to the payload factory, so the documented header-driven encoding does not happen. `text=` is the parameter that honors a declared charset (`body = text.encode(charset)`), and there the getter returns plain `bytes`, not a payload.

Reading discipline (P9 check): "the response's own charset" has exactly one referent in the library — `Response.charset`, parsed from Content-Type — so the falsifier does not require redefining a term; it requires only the universal reading of "an `web.Response` built with a `str` `body=`". If the claim was intended to cover only the default construction with nothing declared, that narrow sub-case does hold; the row as written is unconditional and is refuted by the declared-non-utf-8 constructions above.

## Source
`.venv/lib/python3.13/site-packages/aiohttp/web_response.py` — `Response.body` setter line 705 (`self._body = body = payload.PAYLOAD_REGISTRY.get(body)`, no encoding/content_type argument), lines 712-713 (payload content-type applied only when none set), `Response.__init__` lines 641-691 (charset honored only on the `text=` and `content_type=` paths), `text` getter line 730. Decisive corroboration: `.venv/lib/python3.13/site-packages/aiohttp/payload.py` — `PayloadRegistry.get` lines 110-131, `StringPayload.__init__` lines 430-458 (`real_encoding = "utf-8"`, `value.encode(real_encoding)`), `BytesPayload.decode` lines 380-381. `charset` getter: `aiohttp/helpers.py` `HeadersMixin.charset` lines 769-776. Documentation locator (contradicted): https://docs.aiohttp.org/en/stable/web_reference.html#aiohttp.web.Response.body

## Source Version
aiohttp 3.13.3 (installed, `.venv/lib/python3.13/site-packages/aiohttp`, `__init__.py` `__version__ = "3.13.3"`); docs from `en/stable`

## Source Fingerprint
sha256:3ca73388d5382e67ed5ea295be844cac06ce7950a5a5237e8b39c7b2259ece65  (aiohttp/web_response.py)
sha256:3ba9ec60d50b2fb01e33673227a4d85fbde77159cbe71270b7901e831c0c2aac  (aiohttp/payload.py)
sha256:435df4ecf0849d6cf844ff1cad4c3c764e7c734605d848b7272c2429f471cf91  (aiohttp/helpers.py)

## Invalidates On
- Any change to `Response.body` setter in `aiohttp/web_response.py` (the `PAYLOAD_REGISTRY.get(body)` call at line 705) that starts forwarding `encoding=` / `content_type=` from the response's declared charset
- Any change to `StringPayload.__init__` in `aiohttp/payload.py` (lines 430-458) that makes `real_encoding` default to something other than `"utf-8"`, or that reads a Content-Type-derived charset
- A release where the installed `web_response.py` / `payload.py` bytes no longer hash to the fingerprints above (re-check before reuse)
- Upstream implementing the documented "encode based on Content-Type header" behaviour for `str` bodies, or removing/deprecating `str` acceptance in `Response(body=)`

## Checked At
2026-10-07T03:39:26Z
