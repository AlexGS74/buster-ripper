VERDICT: HOLDS
PROMOTION: ELIGIBLE

## Claim
`_flush_state` returns without an artifact append when the session state's format is `documented`

## What it actually does
Its first branch returns immediately for `state.fmt == "documented"`, before resolving or opening an artifact. The pending buffer is not cleared in this branch.

## Source
`buster_ripper/dump.py:284-294` (production source)

## Source Version
unavailable

## Source Fingerprint
sha256:07b0b076552f72e62c625caade1b01821b3237375900233812d14c0f275b3c2b

## Invalidates On
Changes to `buster_ripper/dump.py`'s `_flush_state` documented-format branch.

## Checked At
2026-09-27
