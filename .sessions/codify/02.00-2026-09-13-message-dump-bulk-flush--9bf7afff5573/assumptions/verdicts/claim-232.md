VERDICT: UNCLEAR
PROMOTION: INELIGIBLE — verdict is UNCLEAR (promotion requires HOLDS or DOES-NOT-HOLD); the claim also has two referents, so no single locator settles it as written.

## Claim
aiohttp's client-side timeout errors are subclasses of `aiohttp.ClientError`

## What it actually does
Both readings are supported by 3.13.3 sources, and they point opposite ways — which
is why this is UNCLEAR and not HOLDS or DOES-NOT-HOLD.

**Reading A — "the timeout exception *classes* aiohttp defines" → true.**
`client_exceptions.py` declares exactly three timeout classes in the client namespace:

```
class ServerTimeoutError(ServerConnectionError, asyncio.TimeoutError)   # :269
class ConnectionTimeoutError(ServerTimeoutError)                        # :273
class SocketTimeoutError(ServerTimeoutError)                            # :277
```

and `ServerConnectionError(ServerConnectionError → ClientConnectionError → ClientError)`.
So all three descend from `ClientError`; the docs' "Hierarchy of exceptions" tree lists
`ServerTimeoutError`, `ConnectionTimeoutError`, `SocketTimeoutError` under `ClientError`.

**Reading B — "whatever aiohttp raises when a client-side timeout fires" → false.**
The cumulative `ClientTimeout.total` budget is not an aiohttp exception class at all.
`ClientSession._request` wraps the whole request in `timer = tm.timer(); with timer:`
(`client.py:622-624`), and `helpers.py` `TimerContext.__exit__` raises
**bare `asyncio.TimeoutError`** (`raise asyncio.TimeoutError from exc_val`, also
`__enter__`/`assert_timeout`). `asyncio.TimeoutError` is not a `ClientError` subclass,
and `client.py`'s outer handler re-raises it unchanged (`except OSError as exc: if
exc.errno is None and isinstance(exc, asyncio.TimeoutError): raise`, `client.py:796`).
Only the per-phase timeouts get converted — `asyncio.TimeoutError` from
`connector.connect()` becomes `ConnectionTimeoutError` (`client.py:737-739`), and
`sock_read` becomes `SocketTimeoutError`. aiohttp says this itself under
`ServerTimeoutError`: "To catch all timeouts, including the total timeout, use
`asyncio.TimeoutError`."

Practical consequence either way: `except ClientError` catches connect / sock_connect /
sock_read timeouts but lets a `total` timeout through as `asyncio.TimeoutError`.

## Source
Primary (reading A): `https://raw.githubusercontent.com/aio-libs/aiohttp/v3.13.3/aiohttp/client_exceptions.py`
lines 254-279 plus the `__all__` list; docs `https://docs.aiohttp.org/en/v3.13.3/client_reference.html`
("Derived from ServerConnectionError and asyncio.TimeoutError", the two "Derived from
ServerTimeoutError" entries, and the exception-hierarchy tree).
Primary (reading B): `https://raw.githubusercontent.com/aio-libs/aiohttp/v3.13.3/aiohttp/helpers.py`
`class TimerContext` (`__enter__`, `__exit__`, `assert_timeout`) and
`https://raw.githubusercontent.com/aio-libs/aiohttp/v3.13.3/aiohttp/client.py` lines 622-624, 737-739, 796.
Library claim; no repo source and no session artifact was consulted.

## Source Version
aiohttp 3.13.3 (tag `v3.13.3` upstream; the version the repo lockfile resolves).

## Source Fingerprint
sha256:bb229bc48da979984a97b9442c1331dd47aeb0d91fa5e98f5a9185ab4afa25e3 (client_exceptions.py)
sha256:435df4ecf0849d6cf844ff1cad4c3c764e7c734605d848b7272c2429f471cf91 (helpers.py)
sha256:7ce41fc1c2142f5346015455e030e3ebec22a41cde0fc299a66ce13be2cb2a9e (client.py)
sha256:a56614286d9c86b6ca347057247299a7740d1ab6b417178ca46d6467a1d77fc6 (client_reference.html)

## Invalidates On
- Ruling out either reading: an answer that means "the named timeout exception classes"
  is HOLDS; an answer that means "every exception a client-side timeout can raise" is
  DOES-NOT-HOLD. Whichever the claim was extracted to mean settles the verdict.
- An aiohttp release that re-parents `ServerTimeoutError`/`ConnectionTimeoutError`/
  `SocketTimeoutError` away from `ClientError`, or that introduces a `ClientError`
  subclass for the total timeout (e.g. an `AsyncTimeoutError` on the cumulative timer).
- A change to `helpers.py` `TimerContext.__exit__` that stops raising bare
  `asyncio.TimeoutError`.
- Any aiohttp version other than 3.13.3 — re-read the four files above.

## Checked At
2026-10-05T18:25:22Z
