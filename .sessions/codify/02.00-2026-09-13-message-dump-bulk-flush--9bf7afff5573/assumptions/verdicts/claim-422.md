VERDICT: DOES-NOT-HOLD
PROMOTION: ELIGIBLE

## Claim
No accessor on aiohttp's `web.Response` returns the `str` value that was passed to `body=`

## What it actually does
`Response.text` is such an accessor. It is a property defined on `Response` itself (web_response.py L723-730) whose getter is `return self._body.decode(self.charset or "utf-8")` when `_body is not None`. A `str` passed to `body=` is stored by the `body` setter (L697-721) as `StringPayload` — `str` is registered to that factory in `payload.py` L1111 — whose constructor encodes the string to utf-8 (`StringPayload.__init__` L430-458) and whose `decode` (`BytesPayload.decode` L380-381) decodes those bytes with the charset `HeadersMixin.charset` parses out of the `Content-Type` header (`helpers.py` L769-776); the setter set that header to `text/plain; charset=utf-8`, so `.text` hands back exactly the string that was passed.

Checked independently of the other rows, and the full accessor surface of `Response` was enumerated (own: `body`, `text`, `content_length`, `content_type`, `charset`, `last_modified`, `etag`; inherited from `StreamResponse`: `prepared`, `task`, `status`, `chunked`, `compression`, `reason`, `keep_alive`, `body_length`, `output_length`, `headers`, `cookies`; plus `MutableMapping.__getitem__`, which only returns values the handler itself stored in `_state`). `text` is the only one returning string content; `body` returns the `StringPayload` object, not the `str` (`Response.body.decode()` would give the string, but that is an accessor on `Payload`, not on `Response`). One counterexample is enough to falsify the universal negative.

Stated so the verdict is auditable rather than quietly definitional: the string is not *retained* as an object — `_value` holds bytes, so `.text` returns a freshly decoded, value-equal `str`. Only under a strict object-identity reading of "returns the `str` value" would this row survive, and that is not the plain meaning of the sentence; the recoverable-content reading is contradicted outright by the getter shown above.

## Source
/Users/alexhubitski/Projects/buster-ripper/.venv/lib/python3.13/site-packages/aiohttp/web_response.py — `text` getter L723-730, `body` setter L697-721, `body` getter L693-695, `charset` property L333-336
/Users/alexhubitski/Projects/buster-ripper/.venv/lib/python3.13/site-packages/aiohttp/payload.py — `PAYLOAD_REGISTRY.register(StringPayload, str)` L1111, `StringPayload.__init__` L430-458, `BytesPayload.decode` L380-381
/Users/alexhubitski/Projects/buster-ripper/.venv/lib/python3.13/site-packages/aiohttp/helpers.py — `HeadersMixin.charset` L769-776

## Source Version
aiohttp 3.13.3 (`__version__` in .venv/lib/python3.13/site-packages/aiohttp/__init__.py); installed source, no bundled documentation

## Source Fingerprint
sha256:3ca73388d5382e67ed5ea295be844cac06ce7950a5a5237e8b39c7b2259ece65 (web_response.py)
sha256:3ba9ec60d50b2fb01e33673227a4d85fbde77159cbe71270b7901e831c0c2aac (payload.py)
sha256:435df4ecf0849d6cf844ff1cad4c3c764e7c734605d848b7272c2429f471cf91 (helpers.py)

## Invalidates On
- web_response.py bytes change such that `Response.text` stops decoding `_body` (e.g. reverts to a `_text` attribute set only via the `text=` kwarg) — the counterexample would disappear and the claim could become true
- payload.py bytes change such that `str` is unregistered from `StringPayload`, `StringPayload` stops encoding eagerly, or `BytesPayload.decode` is removed/changed
- helpers.py `HeadersMixin.charset` changes so the decode codec no longer matches the encode codec for the default `text/plain; charset=utf-8`
- a release adding/removing `Response` properties, which would require re-running the accessor enumeration above

## Checked At
2026-10-06
