VERDICT: HOLDS
PROMOTION: ELIGIBLE

## Claim
With aiohttp's default C parser, a response header name recognized case-insensitively is stored under aiohttp's own canonical spelling rather than the spelling the upstream sent

## What it actually does
The client-side C parser is `aiohttp._http_parser.HttpResponseParser`, which is a subclass of the same `HttpParser` base as the request parser and does not override header handling: `__init__` calls `self._init(cparser.HTTP_RESPONSE, ...)`, and llhttp's `on_header_field` / `on_headers_complete` callbacks run the shared `cdef _process_header()`. That function resolves every field name through `find_header(self._raw_name)`; a hit in the generated table returns the canonical `hdrs` `istr` instance, so an upstream sending `CONTENT-LENGTH: 12` or `content-length: 12` is stored under `Content-Length`, and `etag` is stored under `Etag` (aiohttp's spelling, not RFC title case). Non-table names (e.g. `X-RateLimit-Remaining`) fall through to `raw_header.decode('utf-8', 'surrogateescape')` and keep the upstream's spelling. Matching is genuinely case-insensitive at every position because `tools/gen.py` writes both upper- and lower-case `case` arms per character.

Caveat worth carrying: the upstream's exact bytes survive in `RawResponseMessage.raw_headers`, which `_process_header` appends in parallel as `(self._raw_name, self._raw_value)`. The claim is about the parsed `headers` store — the `CIMultiDictProxy` built from `self._headers` — which is the canonicalized one.

## Source
`aiohttp/_http_parser.pyx`: `cdef class HttpResponseParser(HttpParser)` `__init__` (mode `cparser.HTTP_RESPONSE`, no header-path override) and `cdef _process_header()` lines 379-393 with `cdef inline object find_header(bytes raw_header)` lines 98-107; `aiohttp/_find_header.c` case arms; `aiohttp/hdrs.py` lines 50-107 for the canonical spellings; `aiohttp/_headers.pxi` for the `headers` tuple; `aiohttp/http_parser.py` lines 1073-1085 (`HttpResponseParser` rebinds to `._http_parser` unless `AIOHTTP_NO_EXTENSIONS`) — aiohttp 3.13.3 sdist, `aiohttp-3.13.3.tar.gz` verified against the sha256 declared in this repo's `uv.lock`.

## Source Version
aiohttp 3.13.3 (sdist sha256 a949eee43d3782f2daae4f4a2819b2cb9b0c5d3b7f7a927067cc84dafdbb9f88; `_http_parser.pyx` byte-identical to GitHub tag v3.13.3); multidict 6.7.1

## Source Fingerprint
aiohttp/_http_parser.pyx sha256:f9823c608638b8a77fec1c2bd28dc5c043f08c775ec59e7e262dd74b4ebb7384; aiohttp/_find_header.c sha256:b3c52b9f1bfc08377636e2821330ac2d6cb835b5455df2cc36583eb366c28370; aiohttp/hdrs.py sha256:dab8f933203eeb245d60f856e542a45b888d5a110094620e4811f90f816628d1; aiohttp/_headers.pxi sha256:9fbd35936f1d54f8f0467c798fa2e92612d37e3eddaaedaf26dedfd0eeb43b28; tools/gen.py sha256:b3a79dac016fe1a618fd2e1d5a708ca9c632df6f2427e79c46666488304311dd; aiohttp/http_parser.py sha256:cfa763643394b3b85d3f3a654c4b00572cf4a1ffab400c13ef1cfc3a95e19ee6

## Invalidates On
- aiohttp release where `_http_parser.pyx::_process_header` stops substituting `headers[idx]` for a table hit, or `HttpResponseParser` starts overriding the header callbacks
- a change to a response-header constant in `aiohttp/hdrs.py`, or a name entering/leaving the `istr` set that `tools/gen.py` compiles into `_find_header.c` / `_headers.pxi`
- `tools/gen.py` case-arm generation becoming spelling-sensitive
- the client running with `AIOHTTP_NO_EXTENSIONS` set, or the C extension absent, which routes responses through `http_parser.HttpResponseParserPy` (preserves upstream spelling)

## Checked At
2026-10-05
