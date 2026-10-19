VERDICT: HOLDS
PROMOTION: ELIGIBLE

## Claim
`_append_record` increments `pending_bytes` by the rendered text's UTF-8 encoded length

## What it actually does
It computes `rendered_bytes` from `len(current_text.encode("utf-8"))`, then adds that amount to `state.pending_bytes`; it does not use the byte length of `record`.

## Source
`buster_ripper/dump.py:189-194`

## Source Version
unavailable

## Source Fingerprint
sha256:07b0b076552f72e62c625caade1b01821b3237375900233812d14c0f275b3c2b

## Invalidates On
Changes to `_append_record` in `buster_ripper/dump.py`, particularly the source or accumulation of `rendered_bytes`.

## Checked At
2026-09-27
