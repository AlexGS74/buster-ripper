VERDICT: HOLDS
PROMOTION: ELIGIBLE

## Claim
aiohttp's `ClientResponse.raw_headers` yields response header names exactly as the upstream sent them

## What it actually does
`ClientResponse.raw_headers` returns `self._raw_headers` (client_reqrep.py:432-433), and the only
assignment on the response path is `self._raw_headers = message.raw_headers` inside
`ClientResponse.start()` (client_reqrep.py:566) — the `raw_headers` tuple of the
`RawResponseMessage` the HTTP parser produced. Nothing between the socket and that property touches
name casing: `_on_header_field` accumulates the wire bytes (`self._raw_name = at[:length]`,
`+= at[:length]`), `_process_header` appends `(self._raw_name, self._raw_value)` as-is, and
`_new_response_message(...)` stores that tuple unchanged on the message. Canonicalisation
(`find_header`) affects only the decoded `headers` CIMultiDict, never `raw_headers`. A full grep for
`.upper()` / `.title()` / name-side `.lower()` across `client_reqrep.py`, `http_parser.py` and
`_http_parser.pyx` finds only *value* comparisons (`transfer-encoding`, `connection`,
`content-encoding`) and `method.upper()` — no header-name folding anywhere on the response path.

So the property is `(bytes, bytes)` pairs in exact upstream wire order, duplicates and original
casing intact.

Boundary conditions (do not contradict the claim):
- Entries are undecoded `bytes`, not `str`.
- "raw" covers names fully but values only partially: leading/trailing OWS is stripped from the
  value (`http_parser.py:149-150` says so explicitly: *"note: 'raw' does not mean inclusion of OWS
  before/after the field value"*). Names never carry OWS — a name starting/ending with space or tab
  raises `InvalidHeader`.
- Every hop of an automatic redirect chain gets its own `ClientResponse` built the same way, so
  `resp.raw_headers` is that hop's upstream bytes (earlier hops are in `resp.history`).
- `ClientResponse._raw_headers` is never rebuilt, appended to, or normalised after `start()` in this
  version — the grep above returns exactly three sites (class default, property, assignment).
- ⚠ Stale docstring worth not trusting: `http_parser.py::HttpResponseParser.parse_headers` says
  *"Header name is in upper case."* That is not what the function does (leftover from an older
  aiohttp); the code and the C parser both preserve as-sent casing. Documented so a later reader
  does not "correct" this verdict toward the docstring.

## Source
Primary locator: upstream `aiohttp/client_reqrep.py` at tag `v3.13.3`, `raw_headers` property
(l.431-433) and `ClientResponse.start()` (l.560-574).
Name bytes established by `aiohttp/_http_parser.pyx` `_process_header` (l.380-393),
`_on_header_field` (l.395-402), `_on_headers_complete` response branch + `_new_response_message`
(l.448-451, l.248-268); pure-Python fallback equivalent in `aiohttp/http_parser.py`
`_HeadersParser.parse_headers` (l.146-238) and `HttpResponseParser.parse_headers` (l.509-571).
Documentation corroborates: `client_reference.html` → `ClientResponse.raw_headers` — "**Unmodified**
HTTP headers of response as unconverted bytes, a sequence of (key, value) pairs."

## Source Version
aiohttp 3.13.3 — files fetched from `https://raw.githubusercontent.com/aio-libs/aiohttp/v3.13.3/aiohttp/…`.
The project's `uv.lock` pins aiohttp 3.13.3, matching this tag; no dependency source root was
supplied, so the verdict rests on the upstream tag rather than the installed copy.
Scope note: at this version there is no HTTP/2 client code path (`grep -i http2` returns zero hits
in `client.py`, `client_proto.py`, `client_reqrep.py`, and `ClientSession`'s documented constructor
has no `http2=` parameter), so every `ClientResponse` comes from the HTTP/1.x parser. Under HTTP/2
the equivalent statement would be trivially true only because HPACK forces lowercase names, not
because aiohttp preserves casing.

## Source Fingerprint
sha256 of the exact consulted bytes (digest covers the whole fetched file; the line ranges read are
contiguous subsets of those bytes):
- `client_reqrep.py` — sha256:7844440eb67433c6454edd708c776e47e3fcfec9b8d0aeb980dcfe88c1986ac9
- `_http_parser.pyx` — sha256:f9823c608638b8a77fec1c2bd28dc5c043f08c775ec59e7e262dd74b4ebb7384
- `http_parser.py` — sha256:cfa763643394b3b85d3f3a654c4b00572cf4a1ffab400c13ef1cfc3a95e19ee6
- `client_proto.py` — sha256:e7afd6b4b4ad6460453d82b3804818eafdb8264855d72e891269aec5e253d92a
- `client.py` — sha256:7ce41fc1c2142f5346015455e030e3ebec22a41cde0fc299a66ce13be2cb2a9e

## Invalidates On
- aiohttp bumps past v3.13.3 and `ClientResponse.start()` stops assigning `message.raw_headers`
  verbatim (e.g. reintroduces a normalised copy), or `_raw_headers` gains another assignment site.
- `_http_parser.pyx::_process_header`/`_on_header_field` changes how `_raw_name` is accumulated, or
  `find_header`'s canonicalised name starts being written into the raw tuple.
- HTTP/2 client support returns (`h2` path builds header tuples in Python; a `.lower()` there would
  make names lowercase-by-aiohttp rather than as the upstream put them on the wire).
- The installed aiohttp version diverges from the 3.13.3 tag above.

## Checked At
2026-10-05T21:12:41Z
