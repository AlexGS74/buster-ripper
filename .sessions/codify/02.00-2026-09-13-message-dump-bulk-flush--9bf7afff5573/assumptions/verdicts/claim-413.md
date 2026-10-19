VERDICT: HOLDS
PROMOTION: ELIGIBLE

## Claim
aiohttp's `ClientResponse.text()` raises no `LookupError` for a Content-Type charset that Python cannot resolve, because `get_encoding()` suppresses it and decoding proceeds with UTF-8 or the session's `fallback_charset_resolver`

## What it actually does
`ClientResponse.get_encoding()` wraps the only call that can resolve the declared charset — `codecs.lookup(encoding).name` — in `contextlib.suppress(LookupError, ValueError)`, so an unresolvable Content-Type `charset=` never propagates; execution falls through to the JSON/RDAP branch which returns `"utf-8"`, or to `self._resolve_charset(self, self._body)`, which `ClientSession.__init__` binds to the `fallback_charset_resolver` argument whose default is `lambda r, b: "utf-8"`. `text(encoding=None)` therefore decodes `self._body` with a name that is either a canonical codec from a successful `codecs.lookup`, the literal `"utf-8"`, or whatever the session resolver returns — no `LookupError` on the Content-Type path.

Scope notes (not contradictions of the claim, but the boundaries of it):
- Passing `encoding=` explicitly to `text()` bypasses `get_encoding()` entirely; `self._body.decode(encoding, errors=errors)` then raises `LookupError` for a bad name. The claim's subject is the Content-Type-derived charset, which is the `encoding=None` path.
- A caller-supplied `fallback_charset_resolver` that returns a name `codecs` cannot resolve puts the `LookupError` back into `bytes.decode()` at line 740. The suppression covers the Content-Type charset only, not resolver output.
- `get_encoding()` on a not-yet-read body raises `RuntimeError`, not `LookupError`; `text()` calls `await self.read()` first, so this is unreachable via `text()`.

## Source
`.venv/lib/python3.13/site-packages/aiohttp/client_reqrep.py` lines 709-740 (`get_encoding`, `text`) — the suppression is line 715, the utf-8 branch lines 718-723, the resolver call line 730, the decode line 740. Corroborating, consulted separately: `aiohttp/client.py` line 306 (`fallback_charset_resolver: _CharsetResolver = lambda r, b: "utf-8"`), line 425 (`self._resolve_charset = fallback_charset_resolver`), line 230 (`_CharsetResolver = Callable[[ClientResponse, bytes], str]`); `aiohttp/helpers.py` lines 323-359 (`parse_mimetype` strips quotes/whitespace, so a bogus charset value does reach `codecs.lookup`).

## Source Version
aiohttp 3.13.3 (`aiohttp/__init__.py` line 1: `__version__ = "3.13.3"`; installed as `aiohttp-3.13.3.dist-info`), CPython 3.13 platform darwin. No documentation ships in the wheel; the verdict rests on installed production source, which is rank 2 and the settled ground here.

## Source Fingerprint
sha256:7844440eb67433c6454edd708c776e47e3fcfec9b8d0aeb980dcfe88c1986ac9  (client_reqrep.py — establishes the verdict)
sha256:7ce41fc1c2142f5346015455e030e3ebec22a41cde0fc299a66ce13be2cb2a9e  (client.py — resolver binding and default)
sha256:435df4ecf0849d6cf844ff1cad4c3c764e7c734605d848b7272c2429f471cf91  (helpers.py — charset parameter extraction)

## Invalidates On
- `client_reqrep.py` no longer matching sha256:7844440eb67433c6454edd708c776e47e3fcfec9b8d0aeb980dcfe88c1986ac9, specifically if `contextlib.suppress(LookupError, ValueError)` at line 715 is narrowed, removed, or moved to a `try/except` that re-raises.
- An aiohttp release where `get_encoding()` returns the raw declared charset instead of `codecs.lookup(...).name` on success, or where the fallthrough after a failed lookup stops routing to `_resolve_charset`.
- `client.py` changing the `fallback_charset_resolver` default away from `lambda r, b: "utf-8"` to a resolver that returns unvalidated or Content-Type-derived names.
- `text()` changed to call `codecs.lookup(encoding)` outside a suppress before decoding, or to resolve an explicit `encoding=` through `get_encoding()`.

## Checked At
2026-10-06
