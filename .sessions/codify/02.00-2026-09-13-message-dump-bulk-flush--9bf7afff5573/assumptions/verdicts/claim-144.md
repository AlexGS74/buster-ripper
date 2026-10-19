VERDICT: HOLDS
PROMOTION: INELIGIBLE — exhaustive call-site claim requires multiple source files; no single exact source locator and fingerprint establishes it.

## Claim
All explicit per-request `ClientTimeout` call sites under production `buster_ripper/` are in `backend.py`, `compaction.py`, `routes/chat.py`, and `routes/passthrough.py`

## What it actually does
The four files construct `ClientTimeout` values passed to individual `session.get`, `session.post`, or `session.request` calls. The only other `ClientTimeout` construction under `buster_ripper/` is in `server.py`, where it configures the shared `ClientSession` default, not an individual request.

## Source
Production `buster_ripper/backend.py:54-56`, `buster_ripper/compaction.py:28-39`, `buster_ripper/routes/chat.py:114-130`, `buster_ripper/routes/passthrough.py:24-36`, and `buster_ripper/server.py:55-59` (exhaustive `ClientTimeout` search under `buster_ripper/`).

## Source Version
unavailable

## Source Fingerprint
unavailable

## Invalidates On
Any change to production `buster_ripper/` adding, removing, or moving a `ClientTimeout` call site.

## Checked At
2026-09-27
