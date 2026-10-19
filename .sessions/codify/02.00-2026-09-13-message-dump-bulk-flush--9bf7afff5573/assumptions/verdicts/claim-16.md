VERDICT: HOLDS
PROMOTION: ELIGIBLE

## Claim
`server.py`'s `spawn_kv_cache_monitor`/`on_cleanup` pair establishes the background-task lifecycle: `create_task` at startup, an `app`-state key, cancel-and-await at cleanup

## What it actually does
`on_startup` calls `spawn_kv_cache_monitor`, which creates a task and stores it under `app["kv_monitor_task"]`. `on_cleanup` passes that key to `_cancel_background_task`, which cancels and awaits the task, swallowing `CancelledError`.

## Source
`buster_ripper/server.py:25-34,55-91,102-106` (production source)

## Source Version
unavailable

## Source Fingerprint
sha256:fbd579a7995d0f4a2ae8a6d2994bb9e5fda344c1a900ec581caee38f599647f8

## Invalidates On
Changes to `buster_ripper/server.py`'s startup registration, monitor spawning, task storage, cleanup registration, or cancellation helper.

## Checked At
2026-09-27
