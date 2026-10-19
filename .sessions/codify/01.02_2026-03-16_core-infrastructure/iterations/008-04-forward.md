# Forward Iteration 008-04: Fix Naming and Add Tests

## Summary

Fixed inaccurate naming in `server.py` and added tests for the KV cache monitoring functions.

## Changes Since Iteration 008-03

### Naming Fix

**Problem**: `poll_kv_cache_loop` was misleading - it's NOT a loop. It's a thin wrapper that delegates to `compaction.poll_kv_cache()` which contains the actual `while True` loop.

**Solution**: Renamed to accurately describe the functions:

| Old Name | New Name | Rationale |
|----------|----------|-----------|
| `poll_kv_cache_loop` | `start_kv_cache_monitoring` | Describes what it DOES (starts monitoring), not incorrectly implying it IS the loop |
| `spawn_compaction_task` | `spawn_kv_cache_monitor` | Uses consistent terminology ("monitor") and clarifies it spawns a monitor, not a compaction task |

**Docstrings Updated**:
- `start_kv_cache_monitoring`: Now clearly states it "Delegates to compaction.poll_kv_cache() which polls /metrics every KV_POLL_INTERVAL seconds"
- `spawn_kv_cache_monitor`: Now mentions "periodic KV cache monitoring" and clarifies the polling interval

### Tests Added

Created `tests/test_server_hooks.py` with 4 tests:

1. `test_returns_none_immediately` - Verifies `spawn_kv_cache_monitor` returns `None` without awaiting
2. `test_creates_task_with_correct_coroutine` - Verifies it creates an asyncio task
3. `test_delegates_to_compaction_module` - Verifies `start_kv_cache_monitoring` calls `compaction.poll_kv_cache`
4. `test_passes_session_from_app_context` - Verifies session is correctly passed

## Files Modified

| File | Change |
|------|--------|
| `buster_ripper/server.py` | Renamed `poll_kv_cache_loop` → `start_kv_cache_monitoring`, `spawn_compaction_task` → `spawn_kv_cache_monitor` |
| `tests/test_server_hooks.py` | NEW: 4 tests for spawn and monitoring functions |

## L→N→C→P Compliance

| Function | L | N | C | P | Status |
|----------|---|---|---|---|--------|
| `start_kv_cache_monitoring` | 4 | 0 | 1 | 1 | Compliant |
| `spawn_kv_cache_monitor` | 3 | 0 | 1 | 1 | Compliant |

## Test Results

- Tests: 109/109 passing (100%)
- New tests: 4
- Previous tests: 105

## Deferred Questions

None - straightforward naming fix requested by user.

## Checkpoint Details

- Breaking change: No - internal function rename only
- WIP: No
- Non-trivial: No
- Complex: No
