VERDICT: HOLDS
PROMOTION: ELIGIBLE

## Claim
`spawn_dump_flush` stores a `dump.poll_flush()` task in app state for cleanup

## What it actually does
`spawn_dump_flush` creates an asyncio task from `dump.poll_flush()` and stores it at `app["dump_flush_task"]`; `on_cleanup` cancels and awaits the task under that key.

## Source
`buster_ripper/server.py:37-45,74-91` (production source)

## Source Version
unavailable

## Source Fingerprint
sha256:fbd579a7995d0f4a2ae8a6d2994bb9e5fda344c1a900ec581caee38f599647f8

## Invalidates On
Changes to `buster_ripper/server.py` task spawning or cleanup logic.

## Checked At
2026-09-27
