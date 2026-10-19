VERDICT: HOLDS
PROMOTION: ELIGIBLE

## Claim
aiohttp's pure-Python HTTP parser stores received header names with the spelling they arrived in

## What it actually does
`aiohttp.http_parser.HeadersParser.parse_headers()` splits each raw line at the first `:`, takes `bname` exactly as received, and does `name = bname.decode("utf-8", "surrogateescape")` followed by `headers.add(name, value)` — there is no canonicalization step, no case-folding, and no lookup table anywhere in the path. An upstream or client sending `CONTENT-TYPE` or `x-ratelimit-remaining` gets that exact spelling as the stored key (only leading/trailing SP/HT are rejected, not case). The class is shared: `HttpParser.__init__` creates one `HeadersParser` used by both `HttpRequestParserPy` and `HttpResponseParserPy`, and trailers reuse it. This is the same code the C parser would fall back to when `_http_parser` is unavailable or `AIOHTTP_NO_EXTENSIONS` is set — the C module's `find_header()` table substitution has no counterpart here. The storage layer does not undo this: multidict's `istr` is a `str` subclass holding the value as passed, with the lowercase form kept separately (`canonical` / entry `identity`) purely for case-insensitive comparison.

## Source
`aiohttp/http_parser.py` lines 158-180 (`bname, bvalue = line.split(b":", 1)` … `name = bname.decode("utf-8", "surrogateescape")`) and 234-235 (`headers.add(name, value)` / `raw_headers.append((bname, bvalue)))`), instantiated once at line 286 in `class HttpParser(abc.ABC, Generic[_MsgT])`; corroborating storage locators: `multidict/_multilib/istr.h` (`istr_new` keeps the value, computes `canonical` via `str.lower()`) and `multidict/_multidict_py.py` `class istr(str)` / `class _Entry` (`key: str` alongside `identity`) — from the aiohttp 3.13.3 and multidict 6.7.1 sdists, each verified against the sha256 declared in this repo's `uv.lock`. No test file was consulted.

## Source Version
aiohttp 3.13.3 (sdist sha256 a949eee43d3782f2daae4f4a2819b2cb9b0c5d3b7f7a927067cc84dafdbb9f88; `http_parser.py` byte-identical to GitHub tag v3.13.3); multidict 6.7.1 (sdist sha256 ec6652a1bee61c53a3e5776b6049172c53b6aaba34f18c9ad04f82712bac623d)

## Source Fingerprint
aiohttp/http_parser.py sha256:cfa763643394b3b85d3f3a654c4b00572cf4a1ffab400c13ef1cfc3a95e19ee6; multidict/_multilib/istr.h sha256:34a63c28a8a427d5a0c50aa15bd2f0ececc0d67181e74fe65666daaec48cacd6; multidict/_multidict_py.py sha256:546439f0fed539dea545fdd65408a76fad9a0f5e833dd01646debcaa68893171; multidict/docs/index.rst sha256:0bd9654ca72c8a152954d5f454a44dd0fe6c16da516c31e44d25a790b5d35cbc

## Invalidates On
- an aiohttp release where `HeadersParser.parse_headers` wraps the decoded name (e.g. `istr(name)` from a canonical table, or mirrors `_http_parser.find_header`) before `headers.add`
- a split making `HttpRequestParserPy`/`HttpResponseParserPy` stop sharing this `HeadersParser` instance, so one side canonicalizes
- a multidict release where `CIMultiDict` stores the case-folded form as the visible key instead of the passed spelling
- a change to this repo's pinned aiohttp/multidict versions in `uv.lock`, which invalidates the fingerprints above

## Checked At
2026-10-05
