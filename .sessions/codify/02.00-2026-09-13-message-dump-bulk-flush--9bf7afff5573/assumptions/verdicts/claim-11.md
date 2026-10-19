VERDICT: HOLDS
PROMOTION: ELIGIBLE

## Claim
`session_id(body)` derives the session key from the body's system-prompt content alone

## What it actually does
`session_id(body)` reads exactly one input — `body.get("system", "")` — flattens a list-valued `system` by joining the `text` fields of its dict blocks, and returns the first 12 hex chars of `hashlib.sha256(system.encode())`. No other body field (model, messages, metadata), no external state, and no clock enter the hash, so the key is a function of the system-prompt content alone. Refinements, not contradictions: non-dict blocks and non-`text` fields inside a list `system` are dropped from the digest input, and an absent `system` hashes the empty string to a constant key.

## Source
buster_ripper/session.py lines 93–98 (`def session_id`); callers buster_ripper/dump.py:64, buster_ripper/routes/count_tokens.py:35, buster_ripper/routes/messages.py:35 assign its return value to `sid`, the session key used for stats/dump and as the `session_stats.session_id` DB primary key (session.py lines 38–43, 59–71).

## Source Version
Repo working tree as of 2026-09-19 (no package version; single local implementation)

## Source Fingerprint
sha256:67a66b8aa733b31e982be4922350a7996ebdddb0fe5d391a6129a1cf342a9dff (buster_ripper/session.py)

## Invalidates On
- Any edit to `session_id` in buster_ripper/session.py that mixes other body fields, request metadata, or external state into the digest input
- Any change to how a list-valued `system` is flattened (e.g., hashing raw blocks instead of joined `text` fields), altering the digest input
- Any change to the digest/truncation (algorithm or `[:12]`) that changes the returned key value

## Checked At
2026-09-19
