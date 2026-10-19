VERDICT: HOLDS
PROMOTION: ELIGIBLE

## Claim
`_cancel_background_task` snapshots the cleanup task's cancellation count before cancelling and awaiting the stored background task

## What it actually does
After retrieving the stored task, it takes `asyncio.current_task()`, saves `cleanup.cancelling()`, calls `task.cancel()`, then awaits `task`.

## Source
`buster_ripper/server.py:82-91`

## Source Version
unavailable

## Source Fingerprint
sha256:6a3e45ef69413838dc53de2939ff1bb06b268746231bf63f3589be407f3d3a5a

## Invalidates On
Change to `buster_ripper/server.py` affecting `_cancel_background_task` or its task storage.

## Checked At
2026-09-27
