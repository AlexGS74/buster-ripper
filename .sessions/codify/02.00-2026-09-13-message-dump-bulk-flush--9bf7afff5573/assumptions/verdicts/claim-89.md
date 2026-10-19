VERDICT: HOLDS
PROMOTION: ELIGIBLE

## Claim
`_write_diff_turn` writes later turns' unified rendered-body diff to `turn_{turn:03d}.diff`

## What it actually does
For turns after zero, the documented writer calls `_write_diff_turn`, which obtains a unified diff of the prior and current rendered body text and writes its joined lines to the named `.diff` file.

## Source
`buster_ripper/dump.py:150-175,250-258`

## Source Version
unavailable

## Source Fingerprint
sha256:c8f093c89b61c355ac182feb070cd90b84924ca0d187332117d76b5c766ad9b9

## Invalidates On
Changes to `buster_ripper/dump.py` affecting documented-format dispatch, `_write_diff_turn`, or `_unified_diff`.

## Checked At
2026-09-27
