VERDICT: HOLDS
PROMOTION: ELIGIBLE

## Claim
aiohttp's `ClientSession.__init__` accepts `connector=` and has no `ssl` parameter — `ssl` is a `TCPConnector.__init__` parameter and a per-request parameter

## What it actually does
`ClientSession.__init__` (client.py:274-309) is a keyword-only signature accepting `connector: Optional[BaseConnector] = None` (client.py:278) with no `ssl` parameter and no `**kwargs` — the only ssl-named parameter is the distinct, deprecated `ssl_shutdown_timeout` (client.py:308), so `ClientSession(ssl=...)` is a `TypeError`. `ssl` lives on `TCPConnector.__init__` as `ssl: Union[bool, Fingerprint, SSLContext] = True` (connector.py:978) and on requests as `ssl: Union[SSLContext, bool, Fingerprint] = True` in `ClientSession._request` (client.py:513, also `ws_connect` at client.py:1007); a per-request value takes precedence over the connector's in `TCPConnector._get_ssl_context` (connector.py:1262-1268).

## Source
aiohttp/client.py:274-309 (`ClientSession.__init__` signature); corroborated by aiohttp/connector.py:978 and aiohttp/client.py:513

## Source Version
aiohttp 3.13.3 (installed at .venv/lib/python3.13/site-packages/aiohttp, Python 3.13)

## Source Fingerprint
sha256:7ce41fc1c2142f5346015455e030e3ebec22a41cde0fc299a66ce13be2cb2a9e (client.py); sha256:bd3db604db820ed6da744d54abb1c2ef3a4e5821ccc48e27dcfb42cfbcd9664c (connector.py)

## Invalidates On
- aiohttp adds an `ssl` (or `**kwargs`) parameter to `ClientSession.__init__`
- aiohttp removes the `ssl` parameter from `TCPConnector.__init__` or from per-request methods
- installed client.py/connector.py bytes no longer match the fingerprints above (version drift)

## Checked At
2026-09-28
