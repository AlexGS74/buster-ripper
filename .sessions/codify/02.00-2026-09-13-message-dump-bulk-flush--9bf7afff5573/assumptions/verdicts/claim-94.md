VERDICT: HOLDS
PROMOTION: ELIGIBLE

## Claim
`_flush_state` clears pending records and byte count after the append succeeds

## What it actually does
After the append context exits successfully, it clears `state.pending` and sets `state.pending_bytes` to zero. An exception before that point skips both resets.

## Source
`buster_ripper/dump.py:284-294`

## Source Version
unavailable

## Source Fingerprint
sha256:07b0b076552f72e62c625caade1b01821b3237375900233812d14c0f275b3c2b

## Invalidates On
Changes to the append or post-append reset sequence in `buster_ripper/dump.py`.

## Checked At
2026-09-27
