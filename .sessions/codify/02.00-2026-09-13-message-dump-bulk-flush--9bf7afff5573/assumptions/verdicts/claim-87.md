VERDICT: HOLDS
PROMOTION: ELIGIBLE

## Claim
`_write_documented` writes a turn file and appends one index line per turn

## What it actually does
On each successful invocation, it writes `turn_000.json` for turn zero or `turn_{N:03d}.diff` thereafter, then appends exactly one summary line to `index.txt`. An intervening write error can prevent the append.

## Source
`buster_ripper/dump.py:150-186` (production source)

## Source Version
unavailable

## Source Fingerprint
sha256:c8f093c89b61c355ac182feb070cd90b84924ca0d187332117d76b5c766ad9b9

## Invalidates On
Changes to `buster_ripper/dump.py`, especially `_write_documented`, `_write_full_turn`, `_write_diff_turn`, or `_append_index_line`.

## Checked At
2026-09-27
