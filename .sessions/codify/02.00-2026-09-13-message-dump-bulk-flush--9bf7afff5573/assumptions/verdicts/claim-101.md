VERDICT: HOLDS
PROMOTION: ELIGIBLE

## Claim
`poll_flush` repeatedly calls `flush_all` then awaits `asyncio.sleep(config.DUMP_FLUSH_INTERVAL)`

## What it actually does
The `while True` loop calls `flush_all()` inside a `try` block, logs any `Exception`, and then awaits `asyncio.sleep(config.DUMP_FLUSH_INTERVAL)` before repeating.

## Source
`buster_ripper/dump.py:114-125`

## Source Version
unavailable

## Source Fingerprint
sha256:c8f093c89b61c355ac182feb070cd90b84924ca0d187332117d76b5c766ad9b9

## Invalidates On
Change to `buster_ripper/dump.py` affecting `poll_flush` or its loop structure.

## Checked At
2026-09-27
