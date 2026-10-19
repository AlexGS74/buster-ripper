VERDICT: HOLDS
PROMOTION: ELIGIBLE

## Claim
Every exception class aiohttp defines for a client-side timeout is a subclass of `aiohttp.ClientError`

## What it actually does
aiohttp defines exactly three exception classes whose purpose is signalling a
client-side timeout — `ServerTimeoutError`, `ConnectionTimeoutError` and
`SocketTimeoutError` — and all three sit inside the `ClientError` subtree:
`SocketTimeoutError`/`ConnectionTimeoutError` → `ServerTimeoutError` →
`ServerConnectionError` → `ClientConnectionError` → `ClientError` → `Exception`
(`ServerTimeoutError` additionally inherits `asyncio.TimeoutError`, so it is
matched by *both* an `aiohttp.ClientError` handler and an
`asyncio.TimeoutError` handler). The claim is exhaustive in the direction that
matters: `ClientTimeout` and `ClientWSTimeout` are dataclasses, `TimeoutHandle`
is a scheduler, none of them exceptions.

Exhaustiveness was established by scanning every top-level `class` statement in
the client-side module surface at the pinned version (`client.py`,
`client_proto.py`, `client_reqrep.py`, `client_ws.py`, `connector.py`,
`helpers.py`, `http_exceptions.py`, `streams.py`, `resolver.py`, `multipart.py`,
`payload.py`, `http_websocket.py`, `_websocket/models.py`, `abc.py`,
`base_protocol.py`, `tcp_helpers.py`, `cookiejar.py`,
`client_middleware_digest_auth.py`, `client_exceptions.py`): the only
timeout-classified exceptions are the three above, all in
`aiohttp/client_exceptions.py`.

Boundary cases deliberately excluded (not client-side timeouts, so they do not
falsify the claim — recorded so the scope can be re-checked):
`aiohttp.web_exceptions.HTTPClientError` (HTTP 4xx *status* base — "client
error" names the offending peer, not the side that timed out; derives from
`HTTPError`/`Response`/`Exception`, **not** from `aiohttp.ClientError`), and the
timeout-*named* status responses `HTTPRequestTimeout` (408) and
`HTTPGatewayTimeout` (504), which are server-generated responses. A reader who
counted those would get a different answer, so if "client-side timeout" was
meant as "any class whose name mentions a timeout", re-extract the claim.

## Source
`aiohttp/client_exceptions.py` @ tag `v3.13.3`, class definitions verbatim:
`class ClientError(Exception)`,
`class ClientConnectionError(ClientError)`,
`class ServerConnectionError(ClientConnectionError)`,
`class ServerTimeoutError(ServerConnectionError, asyncio.TimeoutError)`,
`class ConnectionTimeoutError(ServerTimeoutError)`,
`class SocketTimeoutError(ServerTimeoutError)`.
Corroborated (not load-bearing) by the published hierarchy on the pinned docs
page, `client_reference.html` → "Hierarchy of exceptions" and the per-class
"Derived from …" lines, which place all three timeout classes under
`ClientError` and state for `ServerTimeoutError`: "To catch all timeouts,
including the total timeout, use `asyncio.TimeoutError`."

## Source Version
aiohttp 3.13.3 — the version resolved in this workspace's committed
`uv.lock` (`pyproject.toml` requires `aiohttp`, `requires-python >= 3.11`). No
`Source root:` / `Dependency source roots:` locator was supplied with the
dispatch, so the version-pinned upstream tag `v3.13.3` and its matching
documentation build were used instead of the installed package.

## Source Fingerprint
- `sha256:bb229bc48da979984a97b9442c1331dd47aeb0d91fa5e98f5a9185ab4afa25e3`
  — `https://raw.githubusercontent.com/aio-libs/aiohttp/v3.13.3/aiohttp/client_exceptions.py`
  (primary; the bytes the verdict is grounded in)
- `sha256:a56614286d9c86b6ca347057247299a7740d1ab6b417178ca46d6467a1d77fc6`
  — `https://docs.aiohttp.org/en/v3.13.3/client_reference.html` (corroboration)

## Invalidates On
- Any aiohttp version whose `aiohttp/client_exceptions.py` bases the timeout
  classes differently (e.g. a release that drops `ServerConnectionError` from
  `ServerTimeoutError`'s MRO, or adds a new timeout exception outside the
  `ClientError` subtree).
- aiohttp ≥ 4.0 restructuring of the client exception tree — the docs note that
  the hierarchy "has been significantly modified" once already (2.0), so a
  re-cut of `ClientTimeout`/`ClientWSTimeout` handling could introduce a
  non-`ClientError` timeout class.
- A target version other than 3.13.3: if `uv.lock`'s aiohttp pin moves, re-check
  against that tag before reusing this verdict.
- A re-extraction of "client-side timeout" that includes
  `web_exceptions.HTTPRequestTimeout` / `HTTPGatewayTimeout` /
  `HTTPClientError`, none of which is a `ClientError` subclass.

## Checked At
2026-10-05T21:17:58Z
