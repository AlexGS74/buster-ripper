VERDICT: HOLDS
PROMOTION: ELIGIBLE

## Claim
`dump_turn` selects the documented writer or one of two collapsed record builders using the resolved format

## What it actually does
It obtains `fmt` from `_resolve_format`; `documented` invokes `_write_documented`, `change-focused` invokes `_change_focused_record` through `_append_record`, and the remaining allowed format invokes `_self_contained_record` through `_append_record`. `_resolve_format` rejects unknown formats.

## Source
buster_ripper/dump.py:83-98,128-133

## Source Version
unavailable

## Source Fingerprint
sha256:c8f093c89b61c355ac182feb070cd90b84924ca0d187332117d76b5c766ad9b9

## Invalidates On
Changes to `dump_turn`'s format dispatch or `_resolve_format` in `buster_ripper/dump.py`.

## Checked At
2026-09-27
