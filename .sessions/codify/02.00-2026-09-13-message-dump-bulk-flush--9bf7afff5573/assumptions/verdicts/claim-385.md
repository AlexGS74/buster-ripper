VERDICT: HOLDS
PROMOTION: ELIGIBLE

## Claim
a request passing no `auto_decompress` uses the session's value

## What it actually does
`ClientSession._request` declares `auto_decompress: Optional[bool] = None`, and immediately
before dispatch it substitutes the session's stored value whenever the argument was omitted:
`if auto_decompress is None: auto_decompress = self._auto_decompress`. That resolved value is
then handed to the connection protocol for the response (`set_response_params(auto_decompress=...)`).
Every public HTTP verb reaches this code through `_request` and forwards only `**kwargs`, so a
caller that does not name `auto_decompress` cannot leave it unset — `None` is the "not passed"
sentinel and the session value is what gets used. Note `None` and "omitted" are the same case;
explicitly passing `auto_decompress=False` is honored per-request and does not fall back.

## Source
`/Users/alexhubitski/Projects/buster-ripper/.venv/lib/python3.13/site-packages/aiohttp/client.py`
— `class ClientSession`, `async def _request` (line 486):
- line 518: `auto_decompress: Optional[bool] = None,` (per-request default is "unset", not `True`)
- lines 601-602: `if auto_decompress is None:` / `auto_decompress = self._auto_decompress` (session fallback)
- line 747: `auto_decompress=auto_decompress,` passed to `conn.protocol.set_response_params(...)` (line 743)
- funnel: line 478 `return _RequestContextManager(self._request(method, url, **kwargs))` and lines
  1287, 1297, 1317, 1325, 1333, 1339 — `get`/`post`/`put`/`patch`/`delete` all call `self._request(..., **kwargs)`,
  so none of them can inject its own `auto_decompress` default.
- corroborating type surface: line 195, `_RequestOptions` TypedDict declares `auto_decompress: Union[bool, None]`.

Downstream consumer (same package): `client_proto.py:227` `auto_decompress: bool = True` in
`set_response_params`, i.e. the protocol parameter is itself `True` by default but the client always
passes an explicit resolved value, so the protocol default is never what governs a session request.

Ranked-source note: installed package ships no docs and no `.pyi` stubs; the installed source is the
settling authority. `connector.py:1652` is a second `set_response_params` caller on the
connect/tunnel path, not the per-request path this claim is about.

## Source Version
aiohttp 3.13.3 (`__version__ = "3.13.3"` in `aiohttp/__init__.py`)

## Source Fingerprint
sha256:7ce41fc1c2142f5346015455e030e3ebec22a41cde0fc299a66ce13be2cb2a9e  (client.py)
sha256:416b2c15a0fe0da1457303f7ea52d44334669526792b18af24153eca0e42d708  (aiohttp/__init__.py, version provenance)

## Invalidates On
- Any aiohttp release whose `client.py` bytes differ from the fingerprint above — re-read lines 518 and 601-602.
- The per-request default changing from `None` to a concrete `True`/`False`, or to a distinct sentinel type, which would remove or redirect the session fallback.
- Removal or relocation of the `if auto_decompress is None: auto_decompress = self._auto_decompress` block, or the resolved value no longer being passed at line 747.
- A public verb (`get`/`post`/`request`/…) ceasing to forward `**kwargs` to `_request` and instead supplying its own `auto_decompress` argument.

## Checked At
2026-10-06
