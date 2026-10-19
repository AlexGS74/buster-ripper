VERDICT: HOLDS
PROMOTION: ELIGIBLE

## Claim
aiohttp's `web.Request.raw_headers` yields request header names exactly as the client sent them

## What it actually does
`web.Request.raw_headers` (defined on `BaseRequest`) returns the `raw_headers` tuple of the
`RawRequestMessage` the HTTP parser produced, and that tuple is built from the literal name bytes
off the wire — `self._raw_name = at[:length]` / `+= at[:length]` in the C parser's
`_on_header_field`, appended verbatim as `(self._raw_name, self._raw_value)` in `_process_header`.
Case folding / canonicalisation happens only on the *other* branch of the parser
(`find_header(self._raw_name)` feeds the decoded `headers` CIMultiDict), never on `raw_headers`.
So names arrive as `(bytes, bytes)` pairs in exact wire order, duplicates and mixed case intact.

The pure-Python fallback parser (`HttpParser` is unavailable ⇒ `aiohttp/http_parser.py`) does the
same thing: `bname, bvalue = line.split(b":", 1)` … `raw_headers.append((bname, bvalue))`.

Boundary conditions that do not contradict the claim, but that a reader of this verdict should know:
- Entries are `bytes`, not `str`; there is no decoding step to latin-1/utf-8 on the raw view.
- A name carrying leading/trailing space or tab is rejected outright (`InvalidHeader`), so no
  OWS is ever present in a raw name. The parser's own comment warns that "raw" does **not**
  mean the *value* keeps its surrounding OWS — values are `lstrip`/`strip`ped. The claim is about
  names, so it is unaffected.
- `BaseRequest.clone(headers=…)` rebuilds `raw_headers` as
  `tuple((k.encode("utf-8"), v.encode("utf-8")) for k, v in CIMultiDictProxy(CIMultiDict(headers)).items())`
  (web_request.py:236-240). For a *cloned* request the raw names are whatever casing the
  CIMultiDict preserved, not what the client put on the wire. Only for un-cloned requests is the
  property literally "as the client sent them".
- aiohttp's web server speaks HTTP/1.x only, so there is no HPACK/HTTP/2 path on the request side
  where names would be lowercase by protocol.

## Source
Primary locator: upstream `aiohttp/_http_parser.pyx` at tag `v3.13.3`, `_on_header_field`
(l.395-402), `_process_header` (l.380-393), `RawRequestMessage.raw_headers` (l.115, 122-128).
Chain confirmed by `aiohttp/web_request.py` `BaseRequest.raw_headers` (l.506-509) +
`BaseRequest.__init__` (`self._message = message`, l.169), `aiohttp/web_protocol.py`
`request = self._request_factory(message, payload, …)` (l.588-594), and the pure-Python
`aiohttp/http_parser.py` `_HeadersParser.parse_headers` (l.146-238).
Documentation corroborates but does not settle case: `web_reference.html` →
`BaseRequest.raw_headers` "HTTP headers of response as unconverted bytes, a sequence of
(key, value) pairs" (note: that docstring says "response" — a copy-paste defect in the docs; the
attribute is the request's).

## Source Version
aiohttp 3.13.3 — files fetched from `https://raw.githubusercontent.com/aio-libs/aiohttp/v3.13.3/aiohttp/…`.
(The project's `uv.lock` pins aiohttp 3.13.3, matching this tag; no dependency source root was
supplied, so the verdict rests on the upstream tag, not the installed copy.)

## Source Fingerprint
sha256 of the exact consulted bytes (digest is of the whole fetched file; line ranges above are
contiguous subsets read from those same bytes):
- `_http_parser.pyx` — sha256:f9823c608638b8a77fec1c2bd28dc5c043f08c775ec59e7e262dd74b4ebb7384
- `web_request.py` — sha256:2aaacea7a01e581e5eead2ab1b9e4ba3b65bc2ae3d0f176b2a78ae5b6b76bb4e
- `web_protocol.py` — sha256:eacf5d33399aa96efb6f33354f5d75b863522c5a3a80d99f0e0ed7cd89c0f319
- `http_parser.py` — sha256:cfa763643394b3b85d3f3a654c4b00572cf4a1ffab400c13ef1cfc3a95e19ee6

## Invalidates On
- aiohttp bumps past v3.13.3 and `_http_parser.pyx::_process_header`/`_on_header_field` changes how
  `_raw_name` is accumulated, or `find_header`'s canonicalisation is applied to the raw tuple.
- `web_request.py` stops delegating — i.e. `BaseRequest.raw_headers` returns something other than
  `self._message.raw_headers`, or `BaseRequest.__init__`/`_request_factory` starts rebuilding the
  message's raw headers.
- aiohttp gains server-side HTTP/2 (name casing would then be lowercase-by-protocol, not
  as-sent-by-a-client-in-mixed-case).
- The installed aiohttp version diverges from the 3.13.3 tag above (the digest set is the check).

## Checked At
2026-10-05T21:10:03Z
