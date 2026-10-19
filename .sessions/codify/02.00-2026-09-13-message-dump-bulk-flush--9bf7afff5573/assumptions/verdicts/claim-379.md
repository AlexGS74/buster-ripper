VERDICT: HOLDS
PROMOTION: INELIGIBLE — the collapse depends on `multidict.CIMultiDict.__setitem__` semantics; the `multidict` package was not a supplied source root, so the fingerprinted bytes below (aiohttp only) do not cover the deciding behaviour.

## Claim
aiohttp's client merges two request header names that differ only in case into one entry before the wire

## What it actually does
On the `ClientSession` path — every `session.request()/get()/post()`, and `ws_connect` which delegates to `_request` — the caller's `headers=` argument passes through `ClientSession._prepare_headers`. That function tracks first-seen names in a case-**sensitive** `added_names: Set[str]` and applies each "first-seen" name with `result[key] = value` on a `CIMultiDict`. Since `CIMultiDict` keys compare case-insensitively, the differently-cased second name *replaces* the first instead of standing beside it: `headers=[("X-Foo", "1"), ("x-foo", "2")]` becomes one stored entry carrying the later value, and the writer emits one header line.

Two qualifications, both visible in the same source:

- "Merges" means **overwrite / last-wins**, not value-joining. Nothing comma-joins or otherwise combines the two values, and nothing case-normalizes the surviving name — `_serialize_headers` writes each key verbatim.
- The collapse is **asymmetric with respect to casing**: duplicates with *identical* casing are not merged. The first occurrence goes through `result[key] = value`, subsequent ones through `result.add(key, value)`, so `[("X-Foo", "1"), ("X-Foo", "2")]` keeps two entries and two wire lines, while `[("X-Foo", "1"), ("x-foo", "2")]` keeps one.
- The merge is a session-level step, not a serialization-level one. `ClientRequest.update_headers` itself never merges: every non-Host pair is `add()`ed onto a fresh `CIMultiDict`, and `_serialize_headers` writes one line per item. Code that constructs a `ClientRequest` directly (bypassing `ClientSession._request`) therefore keeps both case-variant entries.

## Source
`/Users/alexhubitski/Projects/buster-ripper/.venv/lib/python3.13/site-packages/aiohttp/client.py` — `_prepare_headers` at lines 1207-1221 (case-sensitive `added_names` set + `result[key] = value`), called for request headers at line 549.

Corroborating locators (same package): `client_reqrep.py:1056-1079` (`update_headers`: `add()` for every non-Host pair, `self.headers[key] = value` only for `hdrs.HOST_ALL`); `hdrs.py:119-121` (`HOST_ALL` = every case permutation of "Host" — aiohttp's own intent-exposing use of assignment, where a user-supplied `host`/`HOST` must *replace* the auto-generated `Host` entry rather than emit a second Host line; this is what establishes inside aiohttp that `CIMultiDict` assignment replaces case-insensitively-equal entries); `_http_writer.pyx:141-153` (`_serialize_headers` — one `Key: Value\r\n` line per stored item, key written verbatim via `PyObject_Str` for `istr`).

Not read: `multidict` (the `CIMultiDict` implementation) sits outside the supplied source roots, per the bounded-source rule. The exact root that would close that gap directly is `<site-packages>/multidict`.

## Source Version
aiohttp 3.13.3 (`aiohttp/__init__.py:1`), installed at the supplied dependency source root.

## Source Fingerprint
sha256:7ce41fc1c2142f5346015455e030e3ebec22a41cde0fc299a66ce13be2cb2a9e  (client.py)
sha256:7844440eb67433c6454edd708c776e47e3fcfec9b8d0aeb980dcfe88c1986ac9  (client_reqrep.py)
sha256:56514404ce87a15bfc6b400026d73a270165b2fdbe70313cfa007de29ddd7e14  (_http_writer.pyx)

## Invalidates On
- Any change to `ClientSession._prepare_headers` (client.py:1207-1221): making `added_names` case-insensitive, or routing repeated names through `add()` instead of `result[key] = value`, stops the collapse and both case variants reach the wire.
- Any change in `multidict` to `CIMultiDict.__setitem__` such that assignment no longer replaces all case-insensitively-equal items (the replacement semantics the verdict rests on; multidict source was not in the supplied roots and is not fingerprinted here).
- Any change to `ClientRequest.update_headers` (client_reqrep.py:1056-1079) or `_serialize_headers` (_http_writer.pyx:125-160) that groups, dedupes, or case-normalizes items at write time.
- aiohttp version change: this collapse path is version 3.13.3 as installed; re-check against the installed bytes before reuse.

## Checked At
2026-10-06
