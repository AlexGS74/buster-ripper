# Forward Iteration 008-01: Fix Codex Findings

**Session**: 01.02_2026-03-16_core-infrastructure
**Phase**: Forward (Implementation)
**Date**: 2026-03-04
**Status**: COMPLETE

## Summary

Fixed 3 codex-review findings (2 P1, 1 P2) identified in previous session validation.

## Findings Fixed

### [P1] Streaming Error Handling - messages.py
**Issue**: Hard-coded `status=200` before upstream call. If upstream returns error, response already prepared with wrong status.

**Fix**: Check for error response BEFORE preparing the StreamResponse. Only prepare with status=200 after we know upstream succeeded.

**Files Modified**:
- `buster_ripper/routes/messages.py` - `_stream_single_request()` now checks upstream status before preparing response

### [P1] Retry Check Timing - messages.py
**Issue**: `ContextOverflowError` raised after `response.prepare()` called. Can't retry after headers sent.

**Fix**: Added `_probe_for_context_overflow()` function that probes upstream BEFORE committing to streaming. Retry loop now happens before any response is sent to client.

**Files Modified**:
- `buster_ripper/routes/messages.py` - New `_probe_for_context_overflow()` function
- `buster_ripper/routes/messages.py` - `handle_streaming()` now probes before streaming

### [P2] Image Cache Eviction - image.py
**Issue**: Never evicts by count despite `IMAGE_CACHE_MAX_SIZE` config existing.

**Fix**: Added `_evict_by_lru()` function that removes oldest entries when cache exceeds max size.

**Files Modified**:
- `buster_ripper/image.py` - New `_evict_by_lru()` function
- `buster_ripper/image.py` - `strip_and_cache_images()` now calls eviction

## Files Modified

| File | Changes |
|------|---------|
| `buster_ripper/routes/messages.py` | Added `_probe_for_context_overflow()`, fixed error handling in `_stream_single_request()`, updated `handle_streaming()` retry logic |
| `buster_ripper/image.py` | Added `_evict_by_lru()` function, integrated into `strip_and_cache_images()` |
| `tests/test_image.py` | New test file with 20 tests for image caching |

## L→N→C→P Compliance

### New Functions

| Function | L | N | C | P | Status |
|----------|---|---|---|---|--------|
| `_probe_for_context_overflow` | 9 | 1 | 3 | 2 | ✅ Compliant |
| `_evict_by_lru` | 7 | 1 | 3 | 1 | ✅ Compliant |

### Modified Functions

| Function | L | N | C | P | Status |
|----------|---|---|---|---|--------|
| `handle_streaming` | 15 | 1 | 5 | 4 | ✅ Compliant |
| `_stream_single_request` | 25 | 2 | 7 | 4 | ✅ Compliant |
| `strip_and_cache_images` | 24 | 2 | 6 | 2 | ✅ Compliant |

All functions within acceptable thresholds.

## Tests Written

### New Test File: `tests/test_image.py`

**Test Classes**:
- `TestImageDetection` (5 tests) - Image detection in messages
- `TestImageCaching` (5 tests) - Stripping and caching images
- `TestImageCacheTTL` (2 tests) - TTL-based expiration
- `TestImageCacheLRUEviction` (3 tests) - LRU-based eviction by count
- `TestClearSessionCache` (2 tests) - Session cache clearing
- `TestMakeDataUrl` (3 tests) - Data URL construction

**Coverage**: 20 new tests covering all new and modified image functionality.

## Test Results

```
105 passed, 20 warnings in 1.56s
```

**Breakdown**:
- 85 existing tests (from previous T001/T002/T003 work)
- 20 new image tests

## Decisions Made

1. **Probe-first approach for streaming retry**: Instead of catching exception after `response.prepare()`, probe upstream first. This allows proper retry before committing to a response.

2. **LRU eviction by timestamp**: Used timestamp-based LRU since we already track timestamps for TTL. Simple and effective.

3. **Error response handling**: Non-overflow errors now return proper error response with correct status code instead of sending 200 then error body.

## Issues Encountered

1. **mocker.patch.dict on module**: Initial tests failed because `mocker.patch.dict(config, {...})` doesn't work on modules. Fixed by using `mocker.patch.object(config, "ATTR", value)`.

## Questions Asked

None - all findings had clear fixes.

## Deferred Questions

None - straightforward implementation.

## Next Steps

Await Yield validation of codex fixes.
