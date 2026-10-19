VERDICT: HOLDS
PROMOTION: ELIGIBLE

## Claim
The periodic dump-flush trigger under `buster_ripper/` reads its configured interval at runtime

## What it actually does
`poll_flush()` repeatedly flushes and then evaluates `config.DUMP_FLUSH_INTERVAL` as the argument to `asyncio.sleep`, rather than capturing its value when the module loads. The interval is read on every loop iteration.

## Source
`buster_ripper/dump.py:114-125`

## Source Version
unavailable

## Source Fingerprint
sha256:c8f093c89b61c355ac182feb070cd90b84924ca0d187332117d76b5c766ad9b9

## Invalidates On
Changes to `buster_ripper/dump.py` affecting `poll_flush()` or its configuration access.

## Checked At
2026-09-27
