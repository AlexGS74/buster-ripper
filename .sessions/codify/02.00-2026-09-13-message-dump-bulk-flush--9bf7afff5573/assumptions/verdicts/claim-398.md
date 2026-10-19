VERDICT: HOLDS
PROMOTION: ELIGIBLE

## Claim
aiohttp's `ClientResponse.text()` decodes the payload with the charset named in the response's Content-Type header

## What it actually does
With the default `encoding=None`, `text()` calls `self.get_encoding()`, which parses the `Content-Type` header, takes the `charset` parameter, and returns `codecs.lookup(encoding).name`; `text()` then runs `self._body.decode(encoding, errors=errors)`. So the Content-Type charset is the encoding actually used for the decode — it is the first precedence, outranking the JSON/`rdap` UTF-8 default and the session's `fallback_charset_resolver`.

Two documented qualifications, neither of which contradicts the claim:
- an explicit `encoding=` argument to `text()` bypasses header detection entirely (`if encoding is None: encoding = self.get_encoding()`),
- the header charset is used only when Python knows the codec; `codecs.lookup()` failing is suppressed at `client_reqrep.py:715`, and detection then falls through to the JSON/`rdap` UTF-8 default or the session's `fallback_charset_resolver`.

## Source
`/Users/alexhubitski/Projects/buster-ripper/.venv/lib/python3.13/site-packages/aiohttp/client_reqrep.py` lines 709–740 (`ClientResponse.get_encoding`, `ClientResponse.text`); corroborated by upstream `docs/aiohttp/client_reference.rst` published as `ClientResponse.text` ("If *encoding* is `None` content encoding is determined from the Content-Type header, or using the `fallback_charset_resolver` function") and `ClientResponse.get_encoding`.

## Source Version
aiohttp 3.13.3 (installed, `.venv/lib/python3.13/site-packages/aiohttp-3.13.3.dist-info/METADATA`); docs consulted from `docs.aiohttp.org/en/stable` as corroboration only.

## Source Fingerprint
sha256:7844440eb67433c6454edd708c776e47e3fcfec9b8d0aeb980dcfe88c1986ac9 (client_reqrep.py, full file bytes)

## Invalidates On
- The installed aiohttp `client_reqrep.py` digest changes and `ClientResponse.text`/`get_encoding` no longer resolve encoding from the `Content-Type` `charset` parameter.
- `get_encoding()` is changed to reorder precedence (e.g. body sniffing or a JSON default ahead of the header charset).
- The response class in use is a `response_class=` override whose `text()` does not delegate to `ClientResponse.get_encoding`.

## Checked At
2026-10-06
