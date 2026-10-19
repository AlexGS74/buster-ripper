VERDICT: HOLDS
PROMOTION: ELIGIBLE

## Claim
`aiohttp.ClientTimeout(total=None)` leaves the cumulative deadline unset, so no total-time timeout fires for a request

## What it actually does
`ClientTimeout.total` is passed verbatim into `TimeoutHandle`, and both of the handle's
arms are conditional on a non-`None`, strictly positive timeout: `TimeoutHandle.start()`
returns `None` (no `loop.call_at` deadline is scheduled at all) and `TimeoutHandle.timer()`
returns a `TimerNoop` instead of a `TimerContext`. With `total=None` there is therefore no
cumulative deadline object and no task-cancelling timer for the request; the other
`ClientTimeout` fields (`connect`, `sock_connect`, `sock_read`) are armed independently and
are unaffected.

Two source facts sharpen this:
- `total: Optional[float] = None` is the *dataclass default*, so a bare `ClientTimeout()`
  also disables the total deadline; only `ClientSession`'s own default
  (`DEFAULT_TIMEOUT = ClientTimeout(total=5 * 60, sock_connect=30)`) supplies one.
- `timeout=0` behaves the same as `None` (`timeout is not None and timeout > 0` is false),
  so `ClientTimeout(total=0)` does not produce an immediate timeout either.

A side use of the value exists at `connector.py:1458` and `connector.py:1467`, where
`timeout.total` is forwarded as `ssl_handshake_timeout=` to `loop.start_tls()`. With
`total=None` that becomes `None`, i.e. asyncio's own "no handshake timeout" — it does not
create a total-time deadline, so it does not contradict the claim.

## Source
`/Users/alexhubitski/Projects/buster-ripper/.venv/lib/python3.13/site-packages/aiohttp/helpers.py`
lines 609-625 (`TimeoutHandle.start` / `TimeoutHandle.timer`), plus lines 643-656
(`TimerNoop`) and 659-673 (`TimerContext.assert_timeout`); constructed from
`/Users/alexhubitski/Projects/buster-ripper/.venv/lib/python3.13/site-packages/aiohttp/client.py`
lines 202-207 (`total: Optional[float] = None`), 224 (`DEFAULT_TIMEOUT`), and 592-595
(`tm = TimeoutHandle(self._loop, real_timeout.total, ...)` / `handle = tm.start()`).

Corroborating library documentation (version-pinned to the installed release):
`https://docs.aiohttp.org/en/v3.13.3/_sources/client_reference.rst.txt`, `ClientTimeout`
entry — "`total` … Total number of seconds for the whole request. `float`, ``None`` by default."
No documentation is installed inside the package, so the installed source is the
establishing locator and the fetched reference page is corroboration only.

## Source Version
aiohttp 3.13.3 (`__version__ = "3.13.3"` in
`.venv/lib/python3.13/site-packages/aiohttp/__init__.py`), installed at
`/Users/alexhubitski/Projects/buster-ripper/.venv/lib/python3.13/site-packages/aiohttp`

## Source Fingerprint
sha256:7ce41fc1c2142f5346015455e030e3ebec22a41cde0fc299a66ce13be2cb2a9e  (aiohttp/client.py)
sha256:435df4ecf0849d6cf844ff1cad4c3c764e7c734605d848b7272c2429f471cf91  (aiohttp/helpers.py)
sha256:416b2c15a0fe0da1457303f7ea52d44334669526792b18af24153eca0e42d708  (aiohttp/__init__.py)

## Invalidates On
- Any change to `TimeoutHandle.start()` or `TimeoutHandle.timer()` in `helpers.py` that
  drops or inverts the `timeout is not None and timeout > 0` guards, or that gives
  `None` a default deadline.
- Any change in `client.py` that substitutes a non-`None` fallback for `real_timeout.total`
  before constructing `TimeoutHandle`, or that changes `ClientTimeout.total`'s default.
- An aiohttp version bump whose `helpers.py` or `client.py` sha256 differs from the digests
  above (notably any 4.x rewrite of the client timeout plumbing).

## Checked At
2026-10-06T22:21:05Z
