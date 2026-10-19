VERDICT: HOLDS
PROMOTION: ELIGIBLE

## Claim
`_append_record` increments the per-session `pending_bytes` accumulator by `len(current_text)`

## What it actually does
The function adds `len(current_text)` to `state.pending_bytes`; `state` is the per-session `SessionState` passed from `dump_turn`.

## Source
`buster_ripper/dump.py:189-194` (per-session state lookup and call sites at lines 83-98, 136-147)

## Source Version
unavailable

## Source Fingerprint
sha256:c8f093c89b61c355ac182feb070cd90b84924ca0d187332117d76b5c766ad9b9

## Invalidates On
Change to `buster_ripper/dump.py` `_append_record` or its session-state lookup/call sites.

## Checked At
2026-09-27
