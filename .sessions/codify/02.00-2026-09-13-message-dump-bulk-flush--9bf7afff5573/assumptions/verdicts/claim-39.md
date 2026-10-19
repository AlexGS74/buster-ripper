VERDICT: HOLDS
PROMOTION: ELIGIBLE

## Claim
aiohttp's `TCPConnector` verifies the upstream certificate with its default `ssl=True` and disables verification with `ssl=False`

## What it actually does
`TCPConnector.__init__` defaults `ssl: Union[bool, Fingerprint, SSLContext] = True` (connector.py:978) and stores it via `_merge_ssl_params` (connector.py:1003). `TCPConnector._get_ssl_context` (connector.py:1243-1274) maps the connector-level `True` to the module-level `_SSL_CONTEXT_VERIFIED` (connector.py:1274) and a not-True value (`False`) to `_SSL_CONTEXT_UNVERIFIED` (connector.py:1271-1273). `_make_ssl_context(True)` builds `ssl.create_default_context()` — the stdlib verified context (check_hostname on, `CERT_REQUIRED`) — while `_make_ssl_context(False)` sets `check_hostname = False` and `verify_mode = ssl.CERT_NONE` (connector.py:912-913), i.e. verification disabled. Note: a per-request `ssl=` overrides the connector setting (connector.py:1262-1268); the claim is about connector-level behaviour, which is exactly what the source does.

## Source
aiohttp/connector.py:1243-1274 (`TCPConnector._get_ssl_context`), with connector.py:902-925 (`_make_ssl_context`) and connector.py:978 (default)

## Source Version
aiohttp 3.13.3 (installed at .venv/lib/python3.13/site-packages/aiohttp, Python 3.13)

## Source Fingerprint
sha256:bd3db604db820ed6da744d54abb1c2ef3a4e5821ccc48e27dcfb42cfbcd9664c (connector.py)

## Invalidates On
- aiohttp changes `_get_ssl_context` so connector-level `ssl=True` no longer selects a verified context or `ssl=False` no longer selects `CERT_NONE`
- aiohttp changes the `TCPConnector.__init__` `ssl` default away from `True`
- installed connector.py bytes no longer match the fingerprint above (version drift)

## Checked At
2026-09-28
