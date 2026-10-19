VERDICT: HOLDS
PROMOTION: ELIGIBLE

## Claim
aiohttp's client sends a default `Accept-Encoding` header when neither the session nor the request headers name one

## What it actually does
`ClientRequest.DEFAULT_HEADERS` includes `hdrs.ACCEPT_ENCODING` mapped to `_gen_default_accept_encoding()` (at minimum `"gzip, deflate"`, extended with `"br"`/`"zstd"` when brotli/zstd are available). `update_auto_headers()` — called unconditionally from `ClientRequest.__init__` — adds each default only if that header name is absent from `self.headers`. Because `ClientSession._request()` merges session `headers` defaults with request-level headers via `_prepare_headers()` BEFORE constructing the `ClientRequest`, an `Accept-Encoding` named at either level appears in `used_headers` and suppresses the default; when neither names it, the default is added to the outgoing request headers and sent. Header-name matching is case-insensitive (`istr`/`CIMultiDict`). (A header can also be suppressed via `skip_auto_headers`, which the claim does not mention and which does not contradict it.)

## Source
/Users/alexhubitski/Projects/buster-ripper/.venv/lib/python3.13/site-packages/aiohttp/client_reqrep.py — lines 105–114 (`_gen_default_accept_encoding`), 803–806 (`DEFAULT_HEADERS`), 895–896 (`update_headers` then `update_auto_headers` in `__init__`), 1081–1095 (`update_auto_headers` adds a default only `if hdr not in used_headers`); with /Users/alexhubitski/Projects/buster-ripper/.venv/lib/python3.13/site-packages/aiohttp/client.py line 549 (`headers = self._prepare_headers(headers)` merges session defaults + request headers before `ClientRequest` is built) and lines 1207–1221 (`_prepare_headers`).

## Source Version
aiohttp 3.13.3 (installed package: `aiohttp/__init__.py` `__version__ = "3.13.3"`; `aiohttp-3.13.3.dist-info`). Installed package ships no docs directory; verdict grounded in the installed source.

## Source Fingerprint
sha256:7844440eb67433c6454edd708c776e47e3fcfec9b8d0aeb980dcfe88c1986ac9 (client_reqrep.py)
sha256:7ce41fc1c2142f5346015455e030e3ebec22a41cde0fc299a66ce13be2cb2a9e (client.py)

## Invalidates On
- `DEFAULT_HEADERS` in client_reqrep.py drops `hdrs.ACCEPT_ENCODING`, or `_gen_default_accept_encoding` returns no header/empty value
- `update_auto_headers` (or the `used_headers` membership check) changes so defaults are added regardless of existing headers, or is no longer called from `ClientRequest.__init__`
- `ClientSession._request`/`_prepare_headers` stops merging session defaults into the header set passed to `ClientRequest`
- Upstream aiohttp version whose client_reqrep.py bytes differ from the fingerprint above

## Checked At
2026-10-06
