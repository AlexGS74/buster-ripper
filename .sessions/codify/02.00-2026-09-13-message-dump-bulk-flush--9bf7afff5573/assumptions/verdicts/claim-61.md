VERDICT: HOLDS
PROMOTION: INELIGIBLE — the census requires three distinct source files, not one exact source locator.

## Claim
Exactly three outbound call sites under `buster_ripper/` pass an explicit per-request `ClientTimeout` — `compaction.py:32` (5 s, `/metrics` poll), `chat.py:122` (300 s, chat completions), `passthrough.py:26` (60 s)

## What it actually does
The three named modules construct `ClientTimeout(total=5/300/60)` respectively and pass each as `timeout=timeout` to their outbound `session.get/post/request` calls. Other outbound calls under this source root omit per-request timeouts; `server.py` instead configures a session-wide 600-second timeout.

## Source
`buster_ripper/compaction.py:32-39`; `buster_ripper/routes/chat.py:122-130`; `buster_ripper/routes/passthrough.py:26-36`; checked the other outbound calls in `buster_ripper/backend.py:54` and `buster_ripper/routes/messages.py:130-134,231-235`, and the session default in `buster_ripper/server.py:58`.

## Source Version
unavailable

## Source Fingerprint
unavailable

## Invalidates On
Any change to the outbound request calls or timeout construction under `buster_ripper/`.

## Checked At
2026-09-27
