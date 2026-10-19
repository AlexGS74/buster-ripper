VERDICT: HOLDS
PROMOTION: ELIGIBLE

## Claim
aiohttp's per-request `timeout` argument to a request call replaces the session's `ClientTimeout` rather than merging component-by-component, so a request-time `ClientTimeout` whose components are all unset imposes no deadline even when the session was constructed with `total` armed

## What it actually does
`ClientSession._request` selects the timeout object wholesale: the session's `self._timeout` is used only when the argument is left at the `sentinel` default, and any `ClientTimeout` passed at request time is assigned as-is (`real_timeout = timeout`) with no per-field fallback to the session value. Every downstream deadline is then read off that replacement object alone — `TimeoutHandle(loop, real_timeout.total)`, `read_timeout=real_timeout.sock_read`, and the `ClientTimeout` handed to `connector.connect(..., timeout=real_timeout)` for `timeout.connect` / `timeout.sock_connect` — so an all-`None` request-time `ClientTimeout` (which is literally what `ClientTimeout()` is: `total`, `connect`, `sock_read`, `sock_connect` all default to `None`) leaves no timer armed at all: `TimeoutHandle.start()` returns `None` and `.timer()` returns `TimerNoop` for a `None` total, `ceil_timeout(None)` returns `asyncio.timeout(None)`, and `TCPConnector` carries no independent connect timeout to fall back on in 3.13. The session's armed `total` is silently discarded, not partially honoured.

Details of the mechanism (installed aiohttp 3.13.3):
- `client.py:584-590` — `if timeout is sentinel: real_timeout = self._timeout` / `else: ... real_timeout = timeout`. A non-`ClientTimeout` value is converted to `ClientTimeout(total=timeout)`, which likewise replaces the session object entirely. Passing an explicit `timeout=None` at request time also takes the `else` branch and produces `ClientTimeout(total=None)` — i.e. it too disables the session's total deadline.
- `client.py:202-207` — `ClientTimeout` fields `total/connect/sock_read/sock_connect` default to `None`; only `ceil_threshold` has a non-`None` default (`5`), and it is a rounding knob, not a deadline. So "components all unset" == `ClientTimeout()`.
- `client.py:217-220` (the library's own in-source note on this class) — "to create a timeout specific for a single request, either - create a completely new one to overwrite the default - or use ... `attr.evolve` to overwrite the defaults". The library documents replacement and tells callers to build the complete object themselves.
- `helpers.py:609-625` — `TimeoutHandle.start()`/`.timer()` only arm when `self._timeout is not None and > 0`, otherwise `None`/`TimerNoop`.
- `helpers.py:724-735` — `ceil_timeout(None)` → `async_timeout.timeout(None)` (no deadline).
- `connector.py:650`, `1295-1297`, `1332-1334` — connect/sock_connect deadlines come from the *request-time* `ClientTimeout` only; `connector.py` has no `self._timeout` in 3.13.3.
- Session construction (`client.py:330-352`): `ClientSession(timeout=ClientTimeout(total=30))` stores that object verbatim as `self._timeout`; there is no merge with `DEFAULT_TIMEOUT` at request time either.

Scope boundary checked: this covers `ClientSession.request/get/post/...` (and the redirect loop, which runs inside the single `TimeoutHandle` created before the loop). `ws_connect`'s `timeout` argument is a `ClientWSTimeout`, a different object and a different call, so it is not a counterexample to this claim.

## Source
`/Users/alexhubitski/Projects/buster-ripper/.venv/lib/python3.13/site-packages/aiohttp/client.py` lines 202-224 and 584-596 (decisive locator); corroborated by `aiohttp/helpers.py` lines 609-625 and 724-735, and `aiohttp/connector.py` lines 650, 1295-1297, 1332-1334 — all read from the supplied dependency source root `/Users/alexhubitski/Projects/buster-ripper/.venv/lib/python3.13/site-packages/aiohttp`. Upstream docs (`https://docs.aiohttp.org/en/stable/client_reference.html`, `ClientSession.request` parameter list) state per-request `timeout` "override the session's timeout" — fetched `stable` docs describe a newer release than installed, so they corroborate only.

## Source Version
aiohttp 3.13.3 (`aiohttp/__init__.py:1` — `__version__ = "3.13.3"`), installed source at `.venv/lib/python3.13/site-packages/aiohttp`. Corroborating upstream documentation fetched from the `stable` channel (references 3.14.4 features, i.e. newer than the installed version).

## Source Fingerprint
sha256:7ce41fc1c2142f5346015455e030e3ebec22a41cde0fc299a66ce13be2cb2a9e  (aiohttp/client.py — the exact bytes consulted)
sha256:435df4ecf0849d6cf844ff1cad4c3c764e7c734605d848b7272c2429f471cf91  (aiohttp/helpers.py)
sha256:bd3db604db820ed6da744d54abb1c2ef3a4e5821ccc48e27dcfb42cfbcd9664c  (aiohttp/connector.py)

## Invalidates On
- Any aiohttp upgrade/patch that changes `ClientSession._request`'s timeout selection at `client.py:584-590` from whole-object assignment to field-wise merge (e.g. an `attr.evolve(self._timeout, **set_fields)`), or that redefines `sentinel` so an explicit request-time `ClientTimeout` can mean "unset".
- Any change to `ClientTimeout` defaults at `client.py:202-207` that arms a field (e.g. a non-`None` `total` default on the dataclass itself, as `DEFAULT_TIMEOUT` does at `client.py:224`), so `ClientTimeout()` is no longer "all components unset".
- Any reintroduction of an independent connect timeout on `BaseConnector`/`TCPConnector` (a `self._timeout` analogue), or a fallback in `connector.py:650` / `_wrap_create_connection` to a connector-level value when `timeout.connect` / `timeout.sock_connect` is `None`.
- A change to `TimeoutHandle.start()`/`.timer()` (`helpers.py:609-625`) or `ceil_timeout()` (`helpers.py:724-735`) that arms a deadline for a `None` value.
- Verified against the three digests above; a differing digest on any of these files means the verdict must be re-checked before reuse.

## Checked At
2026-10-06 (UTC+00:00)
