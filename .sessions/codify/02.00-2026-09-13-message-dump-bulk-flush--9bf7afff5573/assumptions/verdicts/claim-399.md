VERDICT: DOES-NOT-HOLD
PROMOTION: ELIGIBLE

## Claim
aiohttp's `ClientResponse.text()` raises `LookupError` when the response's Content-Type names a charset whose codec Python does not know

## What it actually does
An unrecognised charset in `Content-Type` never reaches the caller as `LookupError`. `get_encoding()` performs the codec validity check itself and swallows the failure:

```python
encoding = mimetype.parameters.get("charset")
if encoding:
    with contextlib.suppress(LookupError, ValueError):
        return codecs.lookup(encoding).name
```

When that `codecs.lookup()` raises, control falls through instead of propagating, and `text()` decodes with whatever the fallback chain yields: `utf-8` for `application/json` / `application/rdap` (RFC 7159 / RFC 7483), otherwise `self._resolve_charset(self, self._body)` — the session's `fallback_charset_resolver`, which defaults to `lambda r, b: "utf-8"` (`client.py:306,425`; class-level default `_resolve_charset = lambda *_: "utf-8"` at `client_reqrep.py:315`). The observable failure mode on an unknown charset is therefore a silently wrong encoding, and if the body is not decodable as the fallback it surfaces as `UnicodeDecodeError` from `self._body.decode(...)`, not `LookupError`.

`LookupError` can still escape `text()` on a different path than the claim names: an explicit `encoding="…"` argument is passed straight to `bytes.decode()` with no validity check. That is the caller's argument, not the response's Content-Type, so it does not rescue the claim.

## Source
`/Users/alexhubitski/Projects/buster-ripper/.venv/lib/python3.13/site-packages/aiohttp/client_reqrep.py` lines 315, 689–740 (`ClientResponse.read`, `get_encoding`, `text`); defaults at `/Users/alexhubitski/Projects/buster-ripper/.venv/lib/python3.13/site-packages/aiohttp/client.py` lines 306, 425.

## Source Version
aiohttp 3.13.3 (installed, `.venv/lib/python3.13/site-packages/aiohttp-3.13.3.dist-info/METADATA`)

## Source Fingerprint
sha256:7844440eb67433c6454edd708c776e47e3fcfec9b8d0aeb980dcfe88c1986ac9 (client_reqrep.py, full file bytes)

## Invalidates On
- The installed `client_reqrep.py` digest changes and `get_encoding()` no longer wraps `codecs.lookup(encoding)` in `contextlib.suppress(LookupError, ValueError)`.
- `text()` is changed to validate the header charset and re-raise, or `ClientResponse` gains a strict-charset mode enabled by default.
- The claim is re-scoped to the explicit `encoding=` argument rather than the Content-Type charset — a different claim with a different answer.

## Checked At
2026-10-06
