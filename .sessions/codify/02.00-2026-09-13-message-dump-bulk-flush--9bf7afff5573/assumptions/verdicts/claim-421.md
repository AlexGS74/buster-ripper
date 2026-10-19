VERDICT: HOLDS
PROMOTION: ELIGIBLE

## Claim
On an aiohttp `web.Response` constructed with a `str` passed to `body=`, the `.text` property returns that string

## What it actually does
`Response.text` (web_response.py L723-730) is `self._body.decode(self.charset or "utf-8")` when `_body` is not `None`. For a `str` given to `body=`, `_body` is a `StringPayload` (a `BytesPayload`), and `BytesPayload.decode` is `self._value.decode(encoding, errors)` where `_value` is the original string encoded to utf-8 by `StringPayload.__init__`. `charset` comes from `HeadersMixin.charset`, which parses the `Content-Type` header; the `body` setter copies `StringPayload.content_type` into it as `text/plain; charset=utf-8` whenever the caller did not set one, so the decode uses utf-8 and yields a string equal to the one passed in.

Read: `.text` returns a *newly decoded, value-equal* str, not the same object the caller passed — the `str` itself is not retained anywhere (`_value` holds bytes). The claim is about the string's content, so this satisfies it; it is stated here because it is the one reading under which the sentence would be false.

Second caveat, same formula: if the caller supplies a `Content-Type`/`content_type` with a non-utf-8 charset, `body=` still encoded utf-8 while `.text` decodes with the declared charset, so `.text` can raise `UnicodeDecodeError` or return mojibake. That is a header-vs-payload mismatch the setter does not reconcile, not a different `.text` contract.

## Source
/Users/alexhubitski/Projects/buster-ripper/.venv/lib/python3.13/site-packages/aiohttp/web_response.py — `text` getter L723-730, `body` setter L697-721, `charset` property L333-336
/Users/alexhubitski/Projects/buster-ripper/.venv/lib/python3.13/site-packages/aiohttp/payload.py — `StringPayload.__init__` L430-458, `BytesPayload.decode` L380-381
/Users/alexhubitski/Projects/buster-ripper/.venv/lib/python3.13/site-packages/aiohttp/helpers.py — `HeadersMixin.charset` L769-776

## Source Version
aiohttp 3.13.3 (`__version__` in .venv/lib/python3.13/site-packages/aiohttp/__init__.py); installed source, no bundled documentation

## Source Fingerprint
sha256:3ca73388d5382e67ed5ea295be844cac06ce7950a5a5237e8b39c7b2259ece65 (web_response.py)
sha256:3ba9ec60d50b2fb01e33673227a4d85fbde77159cbe71270b7901e831c0c2aac (payload.py)
sha256:435df4ecf0849d6cf844ff1cad4c3c764e7c734605d848b7272c2429f471cf91 (helpers.py)

## Invalidates On
- web_response.py bytes change such that `text` reverts to returning a stored `_text` attribute (set only by the `text=` kwarg) instead of decoding `_body`
- payload.py bytes change such that `StringPayload` stores the `str` rather than its encoded bytes, or `BytesPayload.decode` stops delegating to `self._value.decode`
- helpers.py `HeadersMixin.charset` changes its default/None handling so `.text` decodes with a codec other than the one `StringPayload` encoded with
- any release where the `body` setter stops copying `StringPayload.content_type` into `Content-Type` (charset inference for `.text` would change)

## Checked At
2026-10-06
