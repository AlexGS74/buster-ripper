VERDICT: HOLDS
PROMOTION: ELIGIBLE

## Claim
No call site under `buster_ripper/` passes a per-request `ssl` argument to a session request method

## What it actually does
A grep for `ssl` across `buster_ripper/` yields exactly one occurrence: `server.py:65`, `aiohttp.TCPConnector(ssl=not config.SKIP_TLS_VERIFY)` — a connector-constructor argument applied session-wide at construction, not a per-request argument on a request method. Reading all six session request call sites confirms their keyword arguments contain no `ssl`: `backend.py:54` (timeout), `compaction.py:36` (timeout), `routes/chat.py:125` (json, headers, timeout), `routes/messages.py:130` (json, headers), `routes/messages.py:231` (json, headers), `routes/passthrough.py:29` (method, url, data, headers, params, timeout). TLS posture is set once on the connector in `on_startup`, per that function's docstring.

## Source
buster_ripper/server.py line 65 — the sole `ssl` occurrence under `buster_ripper/`, a `TCPConnector` kwarg rather than a session-request kwarg

## Source Version
Working-tree source of the buster-ripper repo (git metadata not consulted)

## Source Fingerprint
sha256:a9e49823ea456af9c0076781c4713c956745b21b026dfa768faf947e0b130592 (buster_ripper/server.py)

## Invalidates On
An `ssl=` keyword added to any `session.get/post/request/...` call site under `buster_ripper/`
A new session request call site introduced with a per-request `ssl` argument
Removal of the connector-level `ssl` at server.py:65 (would not make the claim false, but signals the TLS-posture mechanism moved — re-check where `ssl` lands)

## Checked At
2026-09-28
