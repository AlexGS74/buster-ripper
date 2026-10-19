VERDICT: DOES-NOT-HOLD
PROMOTION: ELIGIBLE

## Claim
`_flush_state` appends a session's pending batch to its format-named artifact in one file write

## What it actually does
For non-`documented` states, it appends the joined pending records with one `f.write` to `<format>.log`. For `documented` states it returns without appending, even if pending records exist (for example, after a format change).

## Source
`buster_ripper/dump.py:284-294`

## Source Version
unavailable

## Source Fingerprint
sha256:07b0b076552f72e62c625caade1b01821b3237375900233812d14c0f275b3c2b

## Invalidates On
Changes to the `documented` guard or append path in `buster_ripper/dump.py`.

## Checked At
2026-09-27
