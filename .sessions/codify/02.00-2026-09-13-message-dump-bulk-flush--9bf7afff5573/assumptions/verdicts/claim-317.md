VERDICT: HOLDS
PROMOTION: ELIGIBLE

## Claim
`tests/test_routes_integration.py` posts to `/v1/messages` through an aiohttp `TestClient` with the upstream session mocked

## What it actually does
All three parts are present in the file, and they are wired together by one fixture.

- **aiohttp `TestClient`**: line 6 imports `TestClient, TestServer` from
  `aiohttp.test_utils`; the `client` fixture (lines 37-57) builds
  `app = create_app()`, wraps it in `async with TestServer(app) as server:` then
  `async with TestClient(server) as client:` and yields that client.
- **posts to `/v1/messages`**: seven `await client.post("/v1/messages", ...)`
  call sites — lines 77-84, 90-94, 103-105, 124-132, 373-380, 408, 425-432
  (the string `"/v1/messages/count_tokens"` at lines 260, 274, 285, 293, 316 is a
  different route and is not counted among the seven).
- **upstream session mocked**: the route handlers' upstream client is the shared
  `aiohttp` session stored in app state — the file asserts this directly at lines
  360 and 385-388 (`assert "session" in client.server.app`, `... is not None`) and
  that the handlers use it (`session.post.assert_called()`, line 383). Tests patch
  that object's upstream call: `mocker.patch.object(client.server.app["session"],
  "post", return_value=mock_response)` at lines 71-75, 117-121, 370, 406, 419-423,
  `mocker.patch.object(client.server.app["session"], "post")` at line 102, and
  `"request"` at lines 343-347 for the passthrough route. The returned upstream
  payloads are `mocker.AsyncMock()` responses built by `_create_mock_response`
  (lines 12-27). The fixture also stubs backend detection
  (`backend.detect_backend` → `None`, `backend.get_backend_state` → `"sglang"`,
  lines 43-52) and is documented as "Create test client with mocked upstream
  backend".

Scope, stated rather than smoothed over: six of the seven `/v1/messages` posts have
the session mocked. The seventh — `test_messages_endpoint_rejects_invalid_json`
(lines 88-98), posting `data="not json"` — patches nothing, because it asserts a
`400` before any upstream call; its parametrized sibling at lines 100-108 patches
`session.post` precisely to prove `upstream.assert_not_called()`. The claim as
written does not say "every post", so this is corroboration of the mechanism, not a
counterexample.

Source basis: the claim's subject IS this test file, so test source is the
production surface for it (the P10 exception). Nothing was executed — no test run,
no import — per P1; the verdict is read off the file's own text.

## Source
`/Users/alexhubitski/Projects/buster-ripper/tests/test_routes_integration.py`
(lines 1-57 for the client/session mechanism, 60-135 and 354-436 for the
`/v1/messages` posts and their session patches).

## Source Version
Local working tree bytes of `tests/test_routes_integration.py`, 461 lines; no
package version applies (subject is the repo's own test suite). aiohttp's own
version was not read — it lives outside the supplied source root.

## Source Fingerprint
sha256:365aa9a75a07724104aecff360c65675c0ef6e9bd1cfdaf739eea7ce91a79a99
(`shasum -a 256 tests/test_routes_integration.py` — the exact file consulted.)

## Invalidates On
- Any edit to `tests/test_routes_integration.py` that changes the digest above:
  swapping `TestClient`/`TestServer` for another harness (e.g. `aiohttp.pytest_
  aiohttp`'s `aiohttp_client`, or an ASGI transport), renaming or dropping the
  `client` fixture, or changing `"/v1/messages"` to another path.
- Any change to how the app exposes its upstream client — if `create_app()` stops
  storing the shared session under `app["session"]`, or routes stop issuing
  `session.post`/`session.request`, `mocker.patch.object(client.server.app["session"],
  "post")` no longer mocks the upstream call and the third clause fails silently.
- A bump of aiohttp that relocates `TestClient`/`TestServer` out of
  `aiohttp.test_utils`, or that changes `TestServer`'s constructor signature
  (older aiohttp accepted `TestServer(app)`; newer releases require an
  `AppRunner`), which would make the import/fixture lines no longer establish the
  claim as read.

## Checked At
2026-10-05
