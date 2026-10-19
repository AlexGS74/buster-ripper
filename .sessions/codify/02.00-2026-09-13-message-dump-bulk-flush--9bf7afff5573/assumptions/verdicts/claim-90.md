VERDICT: HOLDS
PROMOTION: ELIGIBLE

## Claim
`_append_index_line` appends a summary line to `index.txt`

## What it actually does
It opens `session_dir / "index.txt"` in append mode and writes `summary + "\n"` after checking writability.

## Source
`buster_ripper/dump.py:181-186`

## Source Version
unavailable

## Source Fingerprint
sha256:c8f093c89b61c355ac182feb070cd90b84924ca0d187332117d76b5c766ad9b9

## Invalidates On
Changes to `buster_ripper/dump.py`, particularly `_append_index_line` or `_require_writable`.

## Checked At
2026-09-27
