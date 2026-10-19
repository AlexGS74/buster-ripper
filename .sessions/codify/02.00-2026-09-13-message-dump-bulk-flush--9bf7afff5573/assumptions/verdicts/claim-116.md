VERDICT: HOLDS
PROMOTION: ELIGIBLE

## Claim
`_append_record` computes `len(current_text.encode("utf-8"))` before mutating its pending list

## What it actually does
It assigns `rendered_bytes = len(current_text.encode("utf-8"))` immediately before calling `state.pending.append(record)`.

## Source
`buster_ripper/dump.py:189-193`

## Source Version
unavailable

## Source Fingerprint
sha256:07b0b076552f72e62c625caade1b01821b3237375900233812d14c0f275b3c2b

## Invalidates On
Changes to `_append_record` in `buster_ripper/dump.py`, particularly the order of encoding and pending-list mutation.

## Checked At
2026-09-27
