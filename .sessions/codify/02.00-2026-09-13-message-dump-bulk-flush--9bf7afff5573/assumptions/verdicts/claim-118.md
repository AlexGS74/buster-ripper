VERDICT: HOLDS
PROMOTION: ELIGIBLE

## Claim
`_cancel_background_task` catches ordinary exceptions raised by awaiting the stored task

## What it actually does
After cancelling the stored task, it awaits it inside a try block; `except Exception` logs and suppresses ordinary exceptions from that await.

## Source
`buster_ripper/server.py:82-94`

## Source Version
unavailable

## Source Fingerprint
sha256:cb30bc733586b0f3263c410c37499483a5c59ef8c6bec2aeee38f54f7edb5b7d

## Invalidates On
Changes to `buster_ripper/server.py`, especially `_cancel_background_task`'s try/except around `await task`.

## Checked At
2026-09-27
