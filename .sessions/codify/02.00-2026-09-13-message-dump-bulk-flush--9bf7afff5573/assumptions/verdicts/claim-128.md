VERDICT: HOLDS
PROMOTION: ELIGIBLE

## Claim
`_cancel_background_task` re-raises a child's `CancelledError` when the cleanup task's cancellation count increased during the await

## What it actually does
It catches `asyncio.CancelledError` from awaiting the stored task and re-raises it if `cleanup.cancelling()` exceeds the count saved before `task.cancel()` and the await. The branch does not distinguish the exception's origin; it applies to any `CancelledError` caught there.

## Source
`buster_ripper/server.py:82-95`

## Source Version
unavailable

## Source Fingerprint
sha256:6a3e45ef69413838dc53de2939ff1bb06b268746231bf63f3589be407f3d3a5a

## Invalidates On
Change to `buster_ripper/server.py` affecting `_cancel_background_task` cancellation handling.

## Checked At
2026-09-27
