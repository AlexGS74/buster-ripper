VERDICT: HOLDS
PROMOTION: ELIGIBLE

## Claim
`tests/test_dump_write_count.py`'s `_WriteHook` records a write-bearing operation only for `Path.open` calls whose mode string contains no `"r"` and whose target path lies under the instrumented dump root

## What it actually does
`_WriteHook.covers` (the sole gate for recording) returns `"r" not in mode and path.is_relative_to(self.root)` — a `Path.open` call is wrapped in the recording `_RecordingFile` only when both conditions hold, and only that wrapper's `write` appends events. Modes containing `"r"` (including write-capable ones like `"r+"`) and opens outside the hook's `root` constructor argument return the bare handle and are never recorded.

## Source
tests/test_dump_write_count.py — `_WriteHook.covers` (lines 106-108) and `_WriteHook.install`/`recording_open` (lines 92-104); recording occurs only via `_RecordingFile.write` (lines 74-76)

## Source Version
unavailable (working-tree repo source; git metadata out of scope for this role)

## Source Fingerprint
sha256:63b2c4c8e049f9462d4b44277cb1a3c387680da2820154a3e583d601c814822e

## Invalidates On
- Any edit to `_WriteHook.covers` that changes the mode filter (e.g., admitting `"r+"`) or drops the `is_relative_to(self.root)` check
- Any edit to `_WriteHook.install`/`recording_open` that wraps handles without consulting `covers`, or that records at open time rather than via `_RecordingFile.write`
- Note: this is a claim about the test's instrument, so the test file itself is the production surface (P10 exception)

## Checked At
2026-09-28
