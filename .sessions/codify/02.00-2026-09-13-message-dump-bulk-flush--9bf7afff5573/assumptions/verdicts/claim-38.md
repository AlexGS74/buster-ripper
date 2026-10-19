VERDICT: DOES-NOT-HOLD
PROMOTION: ELIGIBLE

## Claim
aiohttp's `ClientSession` requires its connector to be bound to the running event loop and raises otherwise

## What it actually does
In aiohttp 3.13.3 the enforced invariant is "the session's loop must equal the connector's loop", not "the connector must be bound to the running event loop". What the library does instead, in three parts:

1. When `ClientSession(connector=c)` is constructed without an explicit `loop=`, the session takes `connector._loop` as its own loop (client.py:314–316) and `asyncio.get_running_loop()` is never consulted — it is short-circuited by the truthy `connector._loop` at client.py:318. A connector bound to a different (even already-closed) loop is silently accepted; the session binds to the connector's loop, not the running one. Nothing is raised.
2. The guard that does exist — `if connector._loop is not loop: raise RuntimeError("Session and connector has to use same event loop")` (client.py:378–379) — compares the connector only against the session's own resolved loop. In the default path that comparison is vacuous (the session's loop IS the connector's loop, per part 1). It fires in practice only when `loop=` is passed explicitly to the session with a mismatched connector, or if `connector._loop` were falsy — unreachable for a successfully constructed connector.
3. The running-loop requirement exists one layer down, in the connector itself: `BaseConnector.__init__` does `loop = loop or asyncio.get_running_loop()` (connector.py:313), so constructing a connector with no running loop raises `RuntimeError` (from asyncio's `get_running_loop`) at CONNECTOR construction — not from `ClientSession`, and not about "the running event loop" of a later session. There is also no request-time loop check: the only loop comparison in the client surface is client.py:378, and the other RuntimeError sites in connector.py (1261 SSL unsupported, 1367 missing `start_tls`, 1678 proxy socket duplication, 1810 NamedPipe proactor) are unrelated.

Note on reading: "the running event loop" is standard asyncio vocabulary (`asyncio.get_running_loop()` at construction time); no term redefinition was applied. Under that reading, in the primary construction path the running loop is simply never consulted, so the claimed requirement-and-raise is not what the code does. A different, true statement would be "ClientSession requires its connector to share the session's event loop and raises otherwise" (client.py:378–379) — but that is not the claim as written, because the session derives its own loop from the connector by default.

## Source
/Users/alexhubitski/Projects/buster-ripper/.venv/lib/python3.13/site-packages/aiohttp/client.py lines 314–318 (loop resolution: `loop = connector._loop` when `loop is None`), 373–381 (default connector creation + mismatch guard); aiohttp/connector.py lines 292–313 (`BaseConnector.__init__` — `loop = loop or asyncio.get_running_loop()`), 342 (`self._loop = loop`); RuntimeError audit of client.py and connector.py as listed above. The version-pinned documentation (https://docs.aiohttp.org/en/v3.13.3/client_reference.html) is silent on any running-loop requirement for a passed connector (the `ClientSession.loop` property is only marked "Deprecated since version 3.5"), so the installed source is the establishing source.

## Source Version
aiohttp 3.13.3 (installed in the project's .venv at .venv/lib/python3.13/site-packages; matches the uv.lock pin aiohttp-3.13.3)

## Source Fingerprint
sha256:7ce41fc1c2142f5346015455e030e3ebec22a41cde0fc299a66ce13be2cb2a9e (aiohttp/client.py)
sha256:bd3db604db820ed6da744d54abb1c2ef3a4e5821ccc48e27dcfb42cfbcd9664c (aiohttp/connector.py)

## Invalidates On
- removal or change of the adoption block at client.py:314–316 (e.g., the 4.0 API break dropping the legacy `loop` parameter and always binding the session to `asyncio.get_running_loop()`)
- the guard at client.py:378–379 changing its comparison reference (e.g., comparing `connector._loop` against `asyncio.get_running_loop()` instead of the session's resolved loop)
- `BaseConnector.__init__` (connector.py:313) no longer resolving/requiring a running loop at connector construction

## Checked At
2026-09-19
