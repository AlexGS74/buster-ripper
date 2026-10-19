VERDICT: HOLDS
PROMOTION: ELIGIBLE

## Claim
`_append_record` calls `flush_all()` when accumulated `pending_bytes >= config.DUMP_FLUSH_BYTES`

## What it actually does
`_append_record` adds the UTF-8 byte length of `current_text` to `state.pending_bytes` and calls `flush_all()` when that session's accumulated value reaches or exceeds `config.DUMP_FLUSH_BYTES`.

## Source
`buster_ripper/dump.py:189-195`

## Source Version
unavailable

## Source Fingerprint
sha256:07b0b076552f72e62c625caade1b01821b3237375900233812d14c0f275b3c2b

## Invalidates On
Changes to `buster_ripper/dump.py` `_append_record` threshold or accumulation logic.

## Checked At
2026-09-27
