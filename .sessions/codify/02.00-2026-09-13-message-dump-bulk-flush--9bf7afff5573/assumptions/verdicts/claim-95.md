VERDICT: HOLDS
PROMOTION: ELIGIBLE

## Claim
`on_cleanup` invokes `_final_dump_flush` after cancelling background tasks and before closing the shared session

## What it actually does
On the normal cleanup path, `on_cleanup` awaits cancellation of both registered background tasks, calls `_final_dump_flush()`, then awaits `app["session"].close()`. An externally propagated cancellation can interrupt that sequence.

## Source
`buster_ripper/server.py:74-102`

## Source Version
unavailable

## Source Fingerprint
sha256:cb30bc733586b0f3263c410c37499483a5c59ef8c6bec2aeee38f54f7edb5b7d

## Invalidates On
Change to `buster_ripper/server.py` cleanup ordering or cancellation behavior.

## Checked At
2026-09-27
