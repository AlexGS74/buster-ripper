VERDICT: HOLDS
PROMOTION: ELIGIBLE — holds verdict, one exact tagged source locator, sha256 of consulted bytes, concrete invalidation conditions below

## Claim
A freshly constructed `aiohttp.web.json_response` is unprepared and holds no payload writer

## What it actually does
`json_response(...)` is a pure factory: it serializes `data` to `text` and returns `Response(text=…, body=…, status=…, headers=…, content_type=…)`. Construction touches only `_state`, `_headers`, `_status`/`_reason`, and the body payload — it never assigns `_payload_writer` or `_eof_sent`, which keep their class defaults `None` and `False`. Since `prepared` is `self._eof_sent or self._payload_writer is not None`, a freshly built response reports `prepared is False` and `resp._payload_writer is None`. Both halves of the claim hold.

## Evidence
`aiohttp/web_response.py` @ v3.13.3:

- L833-856 `json_response()` — end to end: `if data is not sentinel: … text = dumps(data)`, then `return Response(text=text, body=body, status=status, reason=reason, headers=headers, content_type=content_type)`. No `await`, no `prepare`, no request argument exists at this level.
- L87-88 class defaults on `StreamResponse`: `_payload_writer: Optional[AbstractStreamWriter] = None` and `_eof_sent: bool = False`.
- L94-117 `StreamResponse.__init__` — body of the constructor is `self._state = {}`, the `_real_headers`/`headers` branch assigning `self._headers`, and `self._set_status(status, reason)`. Nothing else.
- L157-163 `_set_status` — assigns `self._status` / `self._reason` only.
- L121-122 `prepared` — `return self._eof_sent or self._payload_writer is not None` → `False or None is not None` → `False` on a fresh object.
- L628-690 `Response.__init__` — validates `body`/`text`/`content_type`/`charset`, builds `real_headers`, calls `super().__init__(status=…, reason=…, _real_headers=real_headers)`, then `self.text = text` / `self.body = body`, and sets `self._zlib_executor_size` / `self._zlib_executor`. The `text=` path for `json_response(data=…)` folds into `body = text.encode(charset)` (L662-668) and then the `body` setter (L697-721), which assigns `self._body`, optionally copies payload headers, and sets `self._compressed_body = None` — again no writer.
- L626 `Response._send_headers_immediately = False` — related but separate from the claim: it governs *when* the head reaches the transport once `prepare()` runs, not whether the object is prepared before that.

The claim is about the library's factory, so the library source is the production surface; no repo test or fixture was used to establish it (P10).

## Source
https://raw.githubusercontent.com/aio-libs/aiohttp/v3.13.3/aiohttp/web_response.py (L87-88, L94-122, L157-163, L626-690, L697-721, L833-856)

## Source Version
aiohttp 3.13.3 (upstream tag `v3.13.3`; matches the version pinned in this repo's `uv.lock`)

## Source Fingerprint
sha256:3ca73388d5382e67ed5ea295be844cac06ce7950a5a5237e8b39c7b2259ece65 (web_response.py)

## Invalidates On
- `json_response` gaining a `request` parameter or any preparation step in the factory body (L833-856).
- `StreamResponse.__init__` or `Response.__init__` assigning `_payload_writer` / `_eof_sent` eagerly (e.g. a pre-bound writer or pooled-response optimization).
- A redefinition of `prepared` (L121-122) that stops being the `_eof_sent or _payload_writer is not None` disjunction — the claim's "unprepared" is measured by exactly that property.
- aiohttp version bump whose `web_response.py` fingerprint differs from `3ca73388…` above.

## Checked At
2026-10-05T18:18:15Z
