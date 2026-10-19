VERDICT: HOLDS
PROMOTION: ELIGIBLE

## Claim
Iterating `aiohttp.ClientResponse.headers.items()` yields one pair per occurrence of a repeated response header name

## What it actually does
`ClientResponse.headers` is a `CIMultiDictProxy[str]` built by the HTTP parser with one `headers.add(name, value)` call per response header line, so each occurrence of a repeated name is stored as a separate entry. `items()` returns a view whose iterator walks the raw entry list and yields `(key, value)` per stored entry — not per distinct key — including every occurrence of a duplicated header name, in insertion order. `len()` of the dict/view counts entries, not distinct keys, consistent with the per-occurrence iteration.

## Source
Primary: `/Users/alexhubitski/Projects/buster-ripper/.venv/lib/python3.13/site-packages/multidict/_multidict_py.py` — `MultiDict.items()` (line 722-724, returns `_ItemsView`), `_ItemsView._iter` (line 98-102, `for e in self._md._keys.iter_entries(): yield self._md._key(e.key), e.value`), `_HtKeys.iter_entries` (line 595-596, `filter(None, self.entries)` — every entry, duplicates included), `MultiDict.add` (line 775-778, appends a new `_Entry` on every call), `MultiDictProxy.items` (line 1196-1198, delegates to underlying `MultiDict`), `__len__` returning `self._used` entry count (line 715-716).
Chain in aiohttp: `/Users/alexhubitski/Projects/buster-ripper/.venv/lib/python3.13/site-packages/aiohttp/client_reqrep.py` line 428-429 (`headers` property returns `CIMultiDictProxy[str]`) and line 565; `/Users/alexhubitski/Projects/buster-ripper/.venv/lib/python3.13/site-packages/aiohttp/http_parser.py` line 149 + 235 + 238 (one `headers.add(name, value)` per parsed header line, wrapped in `CIMultiDictProxy`).
Documentation: multidict README in `/Users/alexhubitski/Projects/buster-ripper/.venv/lib/python3.13/site-packages/multidict-6.7.1.dist-info/METADATA` lines 76-78: multidict "behaves mostly like a regular `dict` but it may have several *values* for the same *key* and *preserves insertion ordering*"; line 108: the C extension is "for speed" (reference semantics as in the pure-Python implementation).

Note on installed implementation: CPython loads the compiled `_multidict` extension (`_multidict.so`, not readable as source); the verdict is grounded in the pure-Python reference implementation plus the package's own documentation, which states the C extension exists only for speed.

## Source Version
multidict 6.7.1, aiohttp 3.13.3 (from dist-info directory names in site-packages)

## Source Fingerprint
sha256:546439f0fed539dea545fdd65408a76fad9a0f5e833dd01646debcaa68893171 (multidict/_multidict_py.py)
sha256:cfa763643394b3b85d3f3a654c4b00572cf4a1ffab400c13ef1cfc3a95e19ee6 (aiohttp/http_parser.py)
sha256:7844440eb67433c6454edd708c776e47e3fcfec9b8d0aeb980dcfe88c1986ac9 (aiohttp/client_reqrep.py)
sha256:4913fdaae936a309c29b89b98c2dbebd237203fb27ab9e68ae5e4ec17ba3024f (multidict-6.7.1.dist-info/METADATA)

## Invalidates On
- Any change to multidict `_multidict_py.py` view/entry iteration (`_ItemsView._iter`, `iter_entries`, `add`) — bump invalidates the fingerprints above
- Any change to aiohttp `http_parser.py` header assembly (e.g., switching from `add()` per line to a deduplicating/last-wins merge) or `client_reqrep.py` `headers` property type
- Upgrade of multidict or aiohttp to a version where the pure-Python reference implementation's duplicate-occurrence semantics differ, or where `headers` is no longer a `CIMultiDictProxy`

## Checked At
2026-10-06T19:01:14Z
