VERDICT: SAFE
PROMOTION: INELIGIBLE — member 156 has no single source fingerprint and the supplied verdicts do not establish permanent handles for every member.

## Seam
S9 — outgoing chat request `stream` value through dispatch to successful SSE byte relay or raw upstream error response; false/unset through JSON parsing and response transformation. The messages route is the production peer.

## Composition
`_make_chat_request` sends the body as JSON to the upstream and branches on `body.get("stream")`. For a truthy value, `_relay_chat_stream` tests upstream status before preparing the downstream response: status >= 400 reads raw upstream bytes and returns the upstream status, copying the upstream `Content-Type` header only if present; status < 400 prepares a `text/event-stream` response and writes each upstream `iter_any()` chunk unchanged before EOF. The response MIME type does not select the branch. For false or absent `stream`, `_make_chat_request` instead calls `resp.json()`, handles `ClientError` (including `ContentTypeError`) as a JSON 502, or passes parsed JSON to overflow checking and `_apply_response_transforms`, which selects eval or non-eval transformations and serializes JSON. `handle_messages` independently chooses its streaming handler using `body.get("stream", False)`; its successful streaming peer writes original chunks unchanged.

## Dimension Coverage
- **Request-flag authority and true versus false/unset:** traced both truthy chat stream branch and nonstream fallthrough, including messages' analogous dispatch. Neither chat branch sniffs response `Content-Type` to decide streaming.
- **Successful stream and upstream HTTP error:** status < 400 is prepared with SSE headers and raw chunks; status >= 400 returns raw bytes with upstream status, without JSON parsing or a prepared SSE response.
- **Upstream error `Content-Type` present versus absent:** `ClientResponse.headers` exposes the received header; when present its value is copied to `web.Response` alongside bytes, and when absent the response constructor receives `headers=None`. This establishes no *explicit* copied header in the absent case, not an assertion about any header later supplied by aiohttp.
- **Nonstream MIME variants and error handling:** `text/event-stream` fails default `resp.json()` MIME validation after body read (assuming read succeeds); `application/problem+json` passes the MIME validation, subject to successful decoding. `ContentTypeError` is a `ClientError` and maps to JSON 502 in the chat request handler.
- **Nonstream transformations:** parsed JSON goes through overflow detection and, when that does not raise, eval-mode or non-eval choice transformation and JSON serialization; caught transformation exceptions can leave only partial changes. This is distinct from the raw stream relay.

## Counterexample
None found for the listed dimensions. The actual emitted default `Content-Type` for an upstream error lacking that header is not established by these members; the supported contract is only that no explicit header is passed.

## Sources
- `buster_ripper/routes/chat.py:114-142,155-200,245-279`
- `buster_ripper/routes/messages.py:65-70,73-85,121-176`
- `.venv/lib/python3.13/site-packages/aiohttp/client_reqrep.py:102,260-265,427-429,528-565,742-772`
- `.venv/lib/python3.13/site-packages/aiohttp/http_parser.py:146-149,235-238`
- `.venv/lib/python3.13/site-packages/aiohttp/client_exceptions.py:61-65,144`
- `.venv/lib/python3.13/site-packages/aiohttp/web_response.py:628-703`
- `.venv/lib/python3.13/site-packages/multidict/_multidict_py.py:1175-1184,1220-1224`

## Source Versions
Repository production sources: revision unavailable. Installed aiohttp: 3.13.3; installed multidict: 6.7.1.

## Source Fingerprint
unavailable (member 156 lacks a complete atomic source fingerprint; this session composition is not promotion-eligible).

## Invalidates On
Any change to a listed member verdict/fingerprint or the production boundaries: messages stream dispatch/raw relay, chat outgoing body or branch, stream success/error status handling and header copy, nonstream JSON/overflow/transform behavior, aiohttp response-header exposure and JSON MIME matching/exception hierarchy/web response header/body semantics, or multidict absent-key lookup. Recompose the intersecting full dimension; stale member verdicts require full retrace.

## Checked At
2026-09-27
