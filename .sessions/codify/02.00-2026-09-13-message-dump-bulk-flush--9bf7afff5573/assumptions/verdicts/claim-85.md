VERDICT: HOLDS
PROMOTION: ELIGIBLE

## Claim
The size-based dump-flush trigger under `buster_ripper/` reads its configured byte threshold at runtime

## What it actually does
For collapsed dump formats, `_append_record` adds the rendered text's UTF-8 byte length to the session's pending count and compares it directly to `config.DUMP_FLUSH_BYTES` on each append, calling `flush_all()` when reached. The documented format does not use this buffered trigger.

## Source
`buster_ripper/dump.py:189-195` (production source)

## Source Version
unavailable

## Source Fingerprint
sha256:07b0b076552f72e62c625caade1b01821b3237375900233812d14c0f275b3c2b

## Invalidates On
Change to `buster_ripper/dump.py` that alters `_append_record`'s threshold lookup or comparison.

## Checked At
2026-09-27
