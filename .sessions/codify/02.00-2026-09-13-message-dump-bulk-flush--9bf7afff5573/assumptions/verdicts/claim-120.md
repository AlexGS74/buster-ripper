VERDICT: HOLDS
PROMOTION: ELIGIBLE

## Claim
For a non-`documented` session state with pending records, `_flush_state` appends their joined batch to `<format>.log` with one `f.write`

## What it actually does
It derives `<format>.log` from the session directory, opens it in append mode, and makes one `f.write("\n\n".join(state.pending) + "\n")` call. This is conditional on the writable-target guard and successful file I/O; a nonregular or symlink target is rejected instead.

## Source
`buster_ripper/dump.py:269-271,284-294,303-305` (production source)

## Source Version
unavailable

## Source Fingerprint
sha256:07b0b076552f72e62c625caade1b01821b3237375900233812d14c0f275b3c2b

## Invalidates On
Changes to `buster_ripper/dump.py`'s `_flush_state`, `_artifact_path`, or writable-target guard.

## Checked At
2026-09-27
