VERDICT: HOLDS
PROMOTION: ELIGIBLE

## Claim
The existing integration suite substitutes a mock for the upstream session rather than using a real network client

## What it actually does
The integration suite is tests/test_routes_integration.py — the repo's only module presenting itself as integration tests ("Integration tests for route handlers using aiohttp test client"); conftest.py defines no fixtures, so no other shared integration harness exists. `app["session"]` is the shared upstream client session (`buster_ripper/server.py` line 47: `app["session"] = aiohttp.ClientSession(timeout=...)`, which routes use for upstream calls). Every route test in the suite that reaches upstream patches that shared session's methods with `mocker.patch.object(client.server.app["session"], "post", ...)` (six sites: lines 64-68, 100-104, 134-138, 271, 299-303, 325-329) or `.request` for passthrough (lines 244-248), returning `_create_mock_response` AsyncMock stubs whose json/text/read/__aenter__/__aexit__ are all mocked (lines 12-27). The fixture also patches `backend.detect_backend`/`get_backend_state` (lines 36-45). No test in the suite sends a real network request toward the upstream. Precision note: the mocks are patched onto the real session object's methods rather than replacing the session object itself, and the incoming-request side does use aiohttp TestServer/TestClient (real local HTTP); it is the upstream side that is fully mocked. Related but distinct: tests/test_backend.py (backend-detection unit tests, not the integration suite) does construct real `aiohttp.ClientSession` objects against in-process `aiohttp_server` fixture apps on localhost, and test_compaction.py uses `Mock(spec=aiohttp.ClientSession)` — neither contradicts the claim as stated about the integration suite.

## Source
tests/test_routes_integration.py (lines 12-27, 30-50, 64-68, 244-248); corroborated by buster_ripper/server.py line 47 establishing app["session"] as the upstream aiohttp.ClientSession. The claim's subject is the test suite, so the test file is the production surface here.

## Source Version
unavailable (working tree; git metadata out of scope for this check)

## Source Fingerprint
sha256:7d45882d1b594da25e26edba2c3a9f3185e3c426b42ce8ab5ffd97ae57064ada (tests/test_routes_integration.py); sha256:e008d3c971db787ebd1eb890544c6d73f0479bb97d0060ee64d4f77cf150b824 (buster_ripper/server.py)

## Invalidates On
- tests/test_routes_integration.py no longer patches app["session"] post/request before exercising routes (i.e., a test performs a real upstream request)
- buster_ripper/server.py stops storing the upstream client session at app["session"], changing what the patched object is
- the integration suite moves to a different module or a second integration-level suite appears using real network clients toward the upstream

## Checked At
2026-09-19
