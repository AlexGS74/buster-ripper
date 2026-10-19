VERDICT: HOLDS
PROMOTION: ELIGIBLE

## Claim
The dump-format dispatch under `buster_ripper/` reads `config.DUMP_FORMAT` when processing a turn

## What it actually does
`dump_turn` calls `_resolve_format()` for each turn, which reads `config.DUMP_FORMAT`; it then dispatches to the documented, change-focused, or self-contained writer based on that value.

## Source
`buster_ripper/dump.py:83-98,128-133` (production source)

## Source Version
unavailable

## Source Fingerprint
sha256:c8f093c89b61c355ac182feb070cd90b84924ca0d187332117d76b5c766ad9b9

## Invalidates On
Change to `buster_ripper/dump.py` affecting `dump_turn` or `_resolve_format`.

## Checked At
2026-09-27
