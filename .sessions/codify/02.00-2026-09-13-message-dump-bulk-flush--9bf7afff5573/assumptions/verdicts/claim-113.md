VERDICT: HOLDS
PROMOTION: ELIGIBLE

## Claim
Cancelling an already-completed `asyncio.Task` does not replace its stored exception with cancellation

## What it actually does
`Task.cancel()` returns `False` when `self.done()` is true, before requesting cancellation or changing its completion state; its only preceding assignment clears the traceback-logging flag, not the stored exception.

## Source
Installed CPython `asyncio/tasks.py`, `Task.cancel`, lines 183–223 (especially 205–207).

## Source Version
CPython 3.14.7 (Homebrew installed standard library)

## Source Fingerprint
sha256:a97ffdb60a42104a86f2ece781878b3371bd88e9eea8f1ac1fa8f397cc11cca5

## Invalidates On
Change to installed `asyncio/tasks.py` `Task.cancel` implementation or replacement of the selected Python runtime.

## Checked At
2026-09-27
