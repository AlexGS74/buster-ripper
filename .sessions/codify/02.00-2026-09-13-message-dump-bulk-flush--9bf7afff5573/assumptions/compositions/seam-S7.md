VERDICT: SAFE
PROMOTION: INELIGIBLE — member evidence supplies session claim numbers, not permanent handles for every member; the consulted Python hierarchy documentation has no exact-byte fingerprint.

## Seam
S7 — client raw bytes → aiohttp charset decoding → JSON parsing → messages/chat handler catch or count_tokens helper catch → count_tokens HTTP exception transport → typed response; separately upstream response bytes → `resp.json()` → upstream-attributed catch.

## Composition
`Request.json()` awaits `text()`, which decodes bytes using the Content-Type charset or UTF-8, before calling the JSON loader. A decoding failure is `UnicodeDecodeError` (`UnicodeError` → `ValueError`); a syntactically invalid decoded string raises `json.JSONDecodeError` (a different direct child of `ValueError`). Neither is a subclass of the other. At each of the three production client parse sites, explicit exception types, rather than a general `ValueError` catch, distinguish these paths. Messages and chat return `{"error":"Invalid JSON"}` as JSON HTTP 400 for either exception. Count-tokens delegates to `_read_body`: syntax failure becomes `{}` and continues through session selection/token estimation toward a count response, whereas undecodable bytes raise `HTTPBadRequest` with JSON error text and `application/json`; aiohttp transports that exception as HTTP 400. Chat's separate upstream `resp.json()` catch returns JSON HTTP 502 with `{"error":"Invalid JSON from upstream"}` for either exception; it cannot misattribute a client parse error because client parsing occurs outside that try block.

## Dimension Coverage
- Client decoding versus JSON syntax: distinct exception branches beneath `ValueError`; explicit catches cover both without catching unrelated `ValueError`s. An unrelated `ValueError` is not mapped to these invalid-JSON responses by the cited parse catches.
- Production client parse sites: messages, chat, and count_tokens; the other production route, passthrough, uses `request.read()` rather than `request.json()`.
- Messages and chat: both decode failure and JSON syntax failure return direct JSON 400.
- Count-tokens: syntax failure substitutes an empty body, whereas decode failure raises a JSON-content-type HTTPBadRequest which aiohttp sends as HTTP 400. These are materially different paths, not interchangeable error handling.
- Upstream chat parse: either parse exception returns a 502 attributed to upstream, not a client 400; the upstream `aiohttp.ClientError` branch is separate.
- This is not a guarantee that every possible invalid charset name or arbitrary `ValueError` is classified as invalid JSON; those errors are outside the two enumerated exception types.

## Counterexample
None found within the enumerated decode/syntax exception paths and route boundaries. Whether the count-tokens syntax fallback should instead be a 400 is an unstated product preference, not an established incompatibility.

## Sources
- `buster_ripper/routes/messages.py:24-29`; `buster_ripper/routes/chat.py:23-43,114-149`; `buster_ripper/routes/count_tokens.py:16-25,28-50,85`; `buster_ripper/routes/passthrough.py:7-12`; `buster_ripper/routes/__init__.py` (empty).
- `.venv/lib/python3.13/site-packages/aiohttp/web_request.py:686-695`; `.venv/lib/python3.13/site-packages/aiohttp/helpers.py:769-776`; `.venv/lib/python3.13/site-packages/aiohttp/web_protocol.py:500-515`; `.venv/lib/python3.13/site-packages/aiohttp/web_exceptions.py:80-113,234-239`.
- Python exception hierarchy and JSON decoder cited in member verdicts: `https://docs.python.org/3/library/exceptions.html#exception-hierarchy`; `/opt/homebrew/Cellar/python@3.14/3.14.7/Frameworks/Python.framework/Versions/3.14/lib/python3.14/json/decoder.py:20`.

## Source Versions
Repository route revisions unavailable; installed aiohttp 3.13.3; exception hierarchy per Python 3.14.7 documentation; installed Python 3.14.7 JSON decoder. Member evidence for the decoder also cites CPython v3.14.0 source.

## Source Fingerprint
unavailable — permanent member handles and exact consulted documentation bytes are unavailable; member verdicts contain individual available file SHA-256 values.

## Invalidates On
Any change to a member claim, its source fingerprint or exception ancestry; any production route file added, removed, or changed; changes to client or upstream parsing, catch scope/order, fallback or status/content type; changes to aiohttp charset decoding or HTTPException transport; or changes to the Python exception hierarchy or JSON decoder inheritance.

## Checked At
2026-09-27
