VERDICT: HOLDS
PROMOTION: ELIGIBLE

## Claim
`_append_record` adds a collapsed record to the session's pending list

## What it actually does
`_append_record` appends `record` to `state.pending`, the per-session `SessionState` buffer. A threshold-triggered flush can subsequently clear that list.

## Source
`buster_ripper/dump.py:32-46,189-195`

## Source Version
unavailable

## Source Fingerprint
sha256:07b0b076552f72e62c625caade1b01821b3237375900233812d14c0f275b3c2b

## Invalidates On
Changes to `buster_ripper/dump.py` `_append_record` or `SessionState.pending`.

## Checked At
2026-09-27
