VERDICT: DOES-NOT-HOLD
PROMOTION: ELIGIBLE

## Claim
`_append_record` compares pending rendered-text bytes with `config.DUMP_FLUSH_BYTES` before triggering a flush

## What it actually does
It accumulates `len(current_text)` into `state.pending_bytes` and compares that count to `config.DUMP_FLUSH_BYTES` before calling `flush_all()`. Python string `len` counts Unicode code points, not encoded bytes, so non-ASCII rendered text makes the purported byte count inaccurate.

## Source
`buster_ripper/dump.py:189-194`

## Source Version
unavailable

## Source Fingerprint
sha256:c8f093c89b61c355ac182feb070cd90b84924ca0d187332117d76b5c766ad9b9

## Invalidates On
Changes to `buster_ripper/dump.py` affecting `_append_record`'s accounting or flush comparison.

## Checked At
2026-09-27
