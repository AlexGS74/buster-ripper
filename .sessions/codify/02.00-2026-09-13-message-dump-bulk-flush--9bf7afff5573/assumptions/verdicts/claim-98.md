VERDICT: HOLDS
PROMOTION: ELIGIBLE

## Claim
`_resolve_format` reads `config.DUMP_FORMAT` and rejects values outside `config.DUMP_FORMATS`

## What it actually does
It reads `config.DUMP_FORMAT` into `fmt`, raises `ValueError` when `fmt not in config.DUMP_FORMATS`, and otherwise returns `fmt`.

## Source
`buster_ripper/dump.py:128-133`

## Source Version
unavailable

## Source Fingerprint
sha256:c8f093c89b61c355ac182feb070cd90b84924ca0d187332117d76b5c766ad9b9

## Invalidates On
Changes to `buster_ripper/dump.py`, especially `_resolve_format`.

## Checked At
2026-09-27
