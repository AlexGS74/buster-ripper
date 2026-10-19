VERDICT: HOLDS
PROMOTION: ELIGIBLE

## Claim
`dump_turn` does not add pending records for turns processed in `documented` format

## What it actually does
For `documented`, `dump_turn` calls `_write_documented` rather than `_append_record`; the documented writer writes the turn file and index line without appending to `state.pending`. Existing pending records from any earlier processing are not addressed by this claim.

## Source
`buster_ripper/dump.py:83-98` (format dispatch); `buster_ripper/dump.py:150-156,189-195` (writer versus pending append).

## Source Version
unavailable

## Source Fingerprint
sha256:07b0b076552f72e62c625caade1b01821b3237375900233812d14c0f275b3c2b

## Invalidates On
Changes to `buster_ripper/dump.py`'s documented dispatch or `_write_documented` and its callees that add a pending-record append.

## Checked At
2026-09-27
