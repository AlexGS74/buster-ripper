VERDICT: DOES-NOT-HOLD
PROMOTION: ELIGIBLE

## Claim
`SessionState` tracks pending records and their rendered-text byte count per session

## What it actually does
It tracks pending records per session, but `_append_record` increments `pending_bytes` by `len(current_text)`, which counts Python string characters rather than encoded bytes. Non-ASCII rendered text can therefore contribute fewer counted units than its UTF-8 byte length.

## Source
`buster_ripper/dump.py:32-46,189-194`

## Source Version
unavailable

## Source Fingerprint
sha256:c8f093c89b61c355ac182feb070cd90b84924ca0d187332117d76b5c766ad9b9

## Invalidates On
Changes to `buster_ripper/dump.py`'s `SessionState` fields or `_append_record` accounting.

## Checked At
2026-09-27
