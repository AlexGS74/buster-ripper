VERDICT: HOLDS
PROMOTION: ELIGIBLE

## Claim
aiohttp's cumulative total-request timeout raises bare `asyncio.TimeoutError`, which is not a subclass of `aiohttp.ClientError`

## What it actually does
Both halves are true, and they are true of the same mechanism.

1. *The `total` timeout surfaces as a bare `asyncio.TimeoutError`.* In
   `ClientSession._request`, `ClientTimeout.total` is described in-source as
   "cumulative for all request operations (request, redirects, responses, data
   consuming)" and drives a `TimeoutHandle(loop, real_timeout.total, …)`; the
   whole request/redirect loop runs under `with tm.timer():`, which is a
   `helpers.TimerContext`. When the deadline fires, `TimerContext.timeout()`
   cancels the enclosed task and sets `_cancelled`; `TimerContext.__exit__`
   turns that `CancelledError` into `raise asyncio.TimeoutError from exc_val`
   (and `TimerContext.assert_timeout()`, used by the response/stream read paths,
   is `raise asyncio.TimeoutError from None`). Nothing in that path substitutes
   an aiohttp class — `_request`'s outer handler is `except BaseException as e:
   … raise`, i.e. a re-raise. aiohttp's own code says the same thing in its
   exception reference: under `ServerTimeoutError`, "To catch all timeouts,
   including the total timeout, use `asyncio.TimeoutError`", which is only
   necessary advice because the total timeout is *not* a `ServerTimeoutError`.

2. *That exception is not a `ClientError`.* `class ClientError(Exception)` has
   `Exception` as its only base; `asyncio.TimeoutError` is defined by asyncio
   (on Python ≥ 3.11 an alias of the builtin `TimeoutError`, which derives from
   `OSError`) and appears in aiohttp's tree only as a *co-base* of
   `ServerTimeoutError`, never as a base of `ClientError`. `client.py` proves
   aiohttp relies on this non-relationship: its handler order is
   `except ClientError: raise` followed by `except OSError as exc:` with the
   guard `if exc.errno is None and isinstance(exc, asyncio.TimeoutError): raise`
   — a bare stdlib `TimeoutError` would otherwise be caught and re-wrapped as
   `ClientOSError`, so the explicit pass-through exists precisely because it is
   *not* a `ClientError`.

Consequence worth naming: catching `aiohttp.ClientError` does not catch the
total timeout. Catching `asyncio.TimeoutError` catches both the total timeout
and aiohttp's own timeout classes, since those inherit it.

Recorded edge case (does not require choosing between readings of the claim, but
it is a narrow path where the same number takes a different form): `total` is
also passed to the event loop as `ssl_handshake_timeout` in
`connector.py`'s TLS setup, inside the connector's own `ceil_timeout(...)`
scope. If that handshake is the binding constraint, the resulting
`asyncio.TimeoutError` is caught at `client.py:737` and re-raised as
`ConnectionTimeoutError`, which *is* a `ClientError` subclass (and also an
`asyncio.TimeoutError`). The general total-timeout path — `TimerContext` — still
raises bare `asyncio.TimeoutError`.

## Source
`https://raw.githubusercontent.com/aio-libs/aiohttp/v3.13.3/aiohttp/helpers.py`
— `class TimeoutHandle` (`timer()` returns `TimerContext`),
`class TimerContext` (`assert_timeout`, `__enter__`, `__exit__`, `timeout`) with
`raise asyncio.TimeoutError from exc_val` at line 713;
corroborated by `aiohttp/client.py` lines 591–624 (`TimeoutHandle(…, real_timeout.total, …)`,
`timer = tm.timer()`, `with timer:`), 737–740, 793–798 and 923–934, and by
`aiohttp/client_exceptions.py` (`class ClientError(Exception)`,
`class ServerTimeoutError(ServerConnectionError, asyncio.TimeoutError)`).
Docs statement (corroboration only): aiohttp 3.13.3 `client_reference.html`,
`aiohttp.ServerTimeoutError` — "To catch all timeouts, including the total
timeout, use `asyncio.TimeoutError`."

## Source Version
aiohttp 3.13.3 (version resolved from this workspace's committed `uv.lock`;
`requires-python >= 3.11`, so `asyncio.TimeoutError` is the builtin
`TimeoutError`/`OSError` alias). No `Source root:` / `Dependency source roots:`
locator was supplied, so the version-matching upstream tag source was read from
its exact URL instead of the installed package.

## Source Fingerprint
- `sha256:435df4ecf0849d6cf844ff1cad4c3c764e7c734605d848b7272c2429f471cf91`
  — `aiohttp/helpers.py` @ `v3.13.3` (primary: `TimerContext` raises the bare exception)
- `sha256:7ce41fc1c2142f5346015455e030e3ebec22a41cde0fc299a66ce13be2cb2a9e`
  — `aiohttp/client.py` @ `v3.13.3` (total-timeout wiring, pass-through guard, re-raise)
- `sha256:bb229bc48da979984a97b9442c1331dd47aeb0d91fa5e98f5a9185ab4afa25e3`
  — `aiohttp/client_exceptions.py` @ `v3.13.3` (`ClientError` bases)
- `sha256:bd3db604db820ed6da744d54abb1c2ef3a4e5821ccc48e27dcfb42cfbcd9664c`
  — `aiohttp/connector.py` @ `v3.13.3` (handshake edge case)
- `sha256:a56614286d9c86b6ca347057247299a7740d1ab6b417178ca46d6467a1d77fc6`
  — `https://docs.aiohttp.org/en/v3.13.3/client_reference.html` (docs corroboration)

## Invalidates On
- A change to `helpers.TimerContext.__exit__` / `assert_timeout` that raises an
  aiohttp class instead of `asyncio.TimeoutError`, or to
  `TimeoutHandle.timer()` returning a different `BaseTimerContext`.
- A change to `ClientSession._request` that maps the total timeout onto a
  `ClientError` subclass (as it already does for connector-level
  `asyncio.TimeoutError` → `ConnectionTimeoutError`), or that re-wraps the
  bare `TimeoutError` at the `except OSError` guard.
- Python's `asyncio.TimeoutError` aliasing changing again (its relationship to
  builtin `TimeoutError`/`OSError` shifted in 3.11); the not-a-`ClientError`
  half survives that, the `except OSError` reachability does not.
- aiohttp 4.0's client-exception rework, or any move off the pinned 3.13.3
  resolve in `uv.lock`.

## Checked At
2026-10-05T21:17:58Z
