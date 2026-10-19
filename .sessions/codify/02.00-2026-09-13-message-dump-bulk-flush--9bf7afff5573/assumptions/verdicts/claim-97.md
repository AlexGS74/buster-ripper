VERDICT: HOLDS
PROMOTION: ELIGIBLE

## Claim
`on_cleanup` cancels and awaits each background task through `_cancel_background_task`

## What it actually does
`on_cleanup` calls `_cancel_background_task` for both background tasks spawned by this module (`kv_monitor_task` and `dump_flush_task`); the helper calls `task.cancel()` and awaits the task if its key is present. It does not enumerate arbitrary other tasks.

## Source
`buster_ripper/server.py:25-45,67-89`

## Source Version
unavailable

## Source Fingerprint
sha256:cb30bc733586b0f3263c410c37499483a5c59ef8c6bec2aeee38f54f7edb5b7d

## Invalidates On
Change to `buster_ripper/server.py` background task spawning, cleanup calls, or `_cancel_background_task` implementation.

## Checked At
2026-09-27
