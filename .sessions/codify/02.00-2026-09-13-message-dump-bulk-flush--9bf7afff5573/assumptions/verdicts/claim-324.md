VERDICT: HOLDS
PROMOTION: ELIGIBLE

## Claim
With aiohttp's default C parser, a request header name recognized case-insensitively is stored under aiohttp's own canonical spelling rather than the spelling the client sent

## What it actually does
`aiohttp._http_parser.HttpParser._process_header()` converts the raw field-name bytes from llhttp with `find_header(self._raw_name)`. `find_header()` looks the name up (case-insensitively) in a generated trie of `aiohttp/hdrs.py`'s known header names and, on a hit, returns the canonical `istr` object from the generated `headers` tuple instead of the decoded raw name — so a client sending `content-TYPE: text/html` ends up with the key `Content-Type` in `message.headers`, and `ETag`/`etag`/`ETAG` all become `Etag` (aiohttp's own spelling, which is not title case). Only names *not* in the table fall back to `raw_header.decode('utf-8', 'surrogateescape')`, i.e. the client's spelling. The recognition really is case-insensitive at every position: `tools/gen.py` emits both the upper and lower `case` for each character of each known name (`_find_header.c` ships 1197 case arms, e.g. 49 `'t'` and 49 `'T'`).

Caveat worth carrying: the sender's exact bytes are not lost — `RawRequestMessage.raw_headers` is appended separately as `(self._raw_name, self._raw_value)` in the same function, so original spelling survives there while `headers` (the `CIMultiDictProxy`) holds the canonicalized key. "Stored" in the claim is read as the parsed `headers` store, which is the only `str`-keyed one.

## Source
`aiohttp/_http_parser.pyx` lines 98-107 (`cdef inline object find_header(bytes raw_header)`) and 379-393 (`cdef _process_header(self)`); decisive supporting locators: `aiohttp/_find_header.c` (case arms), `aiohttp/hdrs.py` lines 50-107 (`CONTENT_TYPE = istr("Content-Type")`, `ETAG = istr("Etag")`, `TE = istr("TE")`), `aiohttp/_headers.pxi` (the `headers` tuple returned on a hit), `aiohttp/http_parser.py` lines 1073-1085 (`HttpRequestParser` rebinds to `._http_parser` unless `AIOHTTP_NO_EXTENSIONS`) — aiohttp 3.13.3 sdist, extracted from `aiohttp-3.13.3.tar.gz` verified against the sha256 declared in this repo's `uv.lock`.

## Source Version
aiohttp 3.13.3 (sdist sha256 a949eee43d3782f2daae4f4a2819b2cb9b0c5d3b7f7a927067cc84dafdbb9f88, byte-identical `_http_parser.pyx` to tag v3.13.3 on GitHub); multidict 6.7.1

## Source Fingerprint
aiohttp/_http_parser.pyx sha256:f9823c608638b8a77fec1c2bd28dc5c043f08c775ec59e7e262dd74b4ebb7384; aiohttp/_find_header.c sha256:b3c52b9f1bfc08377636e2821330ac2d6cb835b5455df2cc36583eb366c28370; aiohttp/hdrs.py sha256:dab8f933203eeb245d60f856e542a45b888d5a110094620e4811f90f816628d1; aiohttp/_headers.pxi sha256:9fbd35936f1d54f8f0467c798fa2e92612d37e3eddaaedaf26dedfd0eeb43b28; tools/gen.py sha256:b3a79dac016fe1a618fd2e1d5a708ca9c632df6f2427e79c46666488304311dd; aiohttp/http_parser.py sha256:cfa763643394b3b85d3f3a654c4b00572cf4a1ffab400c13ef1cfc3a95e19ee6

## Invalidates On
- `_http_parser.pyx::_process_header` / `find_header` stops mapping a table hit to `headers[idx]` (or `_headers.append` starts taking the decoded raw name) in a new aiohttp release
- any header constant in `aiohttp/hdrs.py` changes spelling, or the name moves in/out of the `istr` set picked up by `tools/gen.py` (`_headers.pxi` / `_find_header.c` regeneration via `./tools/gen.py`)
- `tools/gen.py` stops emitting both case variants per character, making the trie spelling-sensitive
- the server runs with `AIOHTTP_NO_EXTENSIONS` set or the `_http_parser` C extension fails to build, which routes requests through `http_parser.HttpRequestParserPy` (claim-326 behaviour instead)

## Checked At
2026-10-05
