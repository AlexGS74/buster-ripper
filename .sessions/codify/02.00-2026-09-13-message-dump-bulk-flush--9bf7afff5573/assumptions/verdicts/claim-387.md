VERDICT: HOLDS
PROMOTION: ELIGIBLE

## Claim
With aiohttp's `auto_decompress` enabled, `ClientResponse.headers` still contains the upstream `Content-Encoding`

## What it actually does
Decompression is decided *from* the header but never removes it. In both parser implementations, every parsed header — `Content-Encoding` included — is appended to the headers collection that becomes the response message; the `Content-Encoding` value is additionally copied out to a separate field used only to choose the decompressor. `ClientResponse._headers` is then assigned that message headers object verbatim and exposed read-only via the `headers` property, so `resp.headers["Content-Encoding"]` still yields the upstream value (e.g. `"gzip"`) even though `resp.content`/`read()` are serving decompressed bytes. `raw_headers` likewise keeps the original wire header bytes. Nothing in the package deletes it: the only `del self.headers[...]` sites in `client_reqrep.py` are lines 1109/1218/1222, all inside `class ClientRequest` (starts line 793) for `COOKIE`, `CONTENT_LENGTH`, and `TRANSFER_ENCODING` on the outgoing request; `class ClientResponse` (lines 287-792) contains no header removal at all. The server/response-side writes (`web_response.py:434,829`, `web_fileresponse.py:391`) and `ClientRequest.update_content_encoding` (`client_reqrep.py:1124-1139`, request compression) touch different header objects and do not affect a client response's headers.

## Source
`.venv/lib/python3.13/site-packages/aiohttp/_http_parser.pyx` — `_process_header` lines 385-388 (`self._headers.append((name, value))` unconditionally, then `if name is CONTENT_ENCODING: self._content_encoding = value`, i.e. capture-only) and `_on_headers_complete` lines 419/434-439 (`headers = CIMultiDictProxy(CIMultiDict(self._headers))`; the encoding is read into a local `encoding`, the `headers` object is never pruned) → `_new_response_message(..., headers, raw_headers, ...)`. Pure-Python equivalent: `http_parser.py` lines 556-558 (`enc = headers.get(hdrs.CONTENT_ENCODING, "")`; sets `encoding` only) and line 571 (`return (headers, raw_headers, ...)`). Consume side: `client_reqrep.py` line 565 (`self._headers = message.headers`), lines 428-429 (`def headers(self): return self._headers`). No installed docs ship with the package, so this is grounded in the installed source (rank 2), both parser paths.

## Source Version
aiohttp 3.13.3 (`aiohttp-3.13.3.dist-info/METADATA`); corroborating read-only-proxy semantics from multidict 6.7.1

## Source Fingerprint
sha256:f9823c608638b8a77fec1c2bd28dc5c043f08c775ec59e7e262dd74b4ebb7384 _http_parser.pyx
sha256:cfa763643394b3b85d3f3a654c4b00572cf4a1ffab400c13ef1cfc3a95e19ee6 http_parser.py
sha256:7844440eb67433c6454edd708c776e47e3fcfec9b8d0aeb980dcfe88c1986ac9 client_reqrep.py
sha256:546439f0fed539dea545fdd65408a76fad9a0f5e833dd01646debcaa68893171 multidict/_multidict_py.py

## Invalidates On
- aiohttp adding header-mutation on the response path (e.g. deleting/rewriting `Content-Encoding` in `_process_header`, `_on_headers_complete`, `HeadersParser.parse_headers`, or `ClientResponse.start`) now that the payload is decompressed
- a release that moves decompression from the parser into a `ClientResponse.content`/`headers` wrapper that reconciles the header with the decoded body
- a response-header normalizer that drops content-coding headers for hop-by-hop or decoded-body handling
- `ClientResponse._headers` becoming a mutable `CIMultiDict` that user code or middleware conventionally clears
- a change to the recognized-encoding detection at `_http_parser.pyx:438` / `http_parser.py:557` that leaves the header value present but no longer matches it

## Checked At
2026-10-06
