# Organize Phase Correction - Debug Request Dumping

**Session ID**: 01.01_2026-03-16_api-relay-deep-dive
**Iteration**: 002-organize (correction to 001-organize)
**Date**: 2026-03-16

## Correction Summary

**CORRECTION**: Debug request dumping was incorrectly reported as MISSING in iteration 001.

**Reality**: Buster-ripper already has `dump.py` which implements session-aware prompt diffing that is SUPERIOR to cloud-relay's raw JSON approach.

## What Changed

### Capability Matrix Update
- **Before**: Debug dumping listed as "No" for buster-ripper
- **After**: Debug dumping listed as "Yes - `dump.py` with session-aware diffs"

### Effort Estimates Recalculated

| Priority | Before | After | Change |
|----------|--------|-------|--------|
| P0 | 14 hours | 14 hours | Unchanged |
| P1 | 7 hours | 4 hours | -3 hours |
| **Total P0+P1** | **21 hours** | **18 hours** | **-3 hours** |

### Tasks Removed from P1

**T005: Debug request dumping (3h)** - REMOVED
- Was: Add `DEBUG_DIR` config, implement `dump_request()` function
- Reality: Already implemented in `buster_ripper/dump.py`:
  - Turn 0: Saves full JSON to `turn_000.json`
  - Turn N+: Saves unified diff to `turn_NNN.diff`
  - Human-readable rendered sections ([tools], [system], [message N])
  - Already wired into `routes/messages.py` at line 41
  - Uses `DUMP_DIR` config for output directory
  - Maintains session state with turn counting and index tracking

## Files Updated

1. **001-organize.md**: Updated capability matrix, P1 section, task list
2. **plan-summary.md**: Updated P1 gaps table, task descriptions
3. **handoff-context.md**: Updated P1 findings, priority summary

## Learning Moment

**Absence of evidence is not evidence of absence.**

The initial analysis looked for cloud-relay's exact pattern (raw JSON to `debug/` directory) and concluded buster-ripper was missing this feature. However:
- The feature EXISTS in `buster_ripper/dump.py`
- It uses a DIFFERENT approach (session-aware diffs vs raw JSON)
- It's actually SUPERIOR (delta compression + human-readable rendering)

This correction demonstrates the importance of thorough code verification before declaring features "missing."

## Questions Asked

None - this was a straightforward correction based on verifying existing implementation.

## Handoff to Decide

The decide agent should review the corrected effort estimates:
- **P0**: 14 hours (unchanged) - Client session pooling, context overflow retry, backend detection
- **P1**: 4 hours (reduced from 7h) - Per-request IDs, enhanced error handling
- **Total P0+P1**: 18 hours (reduced from 21h)

The implementation plan remains valid with adjusted P1 scope.
