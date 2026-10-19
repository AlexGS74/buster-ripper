VERDICT: HOLDS
PROMOTION: ELIGIBLE

## Claim
`_cancel_background_task` re-raises `CancelledError` when its current cleanup task is cancelling

## What it actually does
Its `except asyncio.CancelledError` branch checks `asyncio.current_task().cancelling()` and re-raises when that count is nonzero; otherwise it suppresses the awaited task's cancellation.

## Source
`buster_ripper/server.py:82-94`

## Source Version
unavailable

## Source Fingerprint
sha256:cb30bc733586b0f3263c410c37499483a5c59ef8c6bec2aeee38f54f7edb5b7d

## Invalidates On
Changes to `buster_ripper/server.py`, especially `_cancel_background_task`'s `CancelledError` handler.

## Checked At
2026-09-27
