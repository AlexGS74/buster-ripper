# Forward Iteration 010-01: KV Monitor Task Cleanup

## Summary
Fixed P2 issue: `spawn_kv_cache_monitor()` now stores the background task handle in `app["kv_monitor_task"]` and `on_cleanup()` properly cancels and awaits the task before closing the session.

## Changes Made

### Files Modified
1. **`buster_ripper/server.py`** (lines 25-34, 60-72):
   - `spawn_kv_cache_monitor()`: Now stores task in `app["kv_monitor_task"]`
   - `on_cleanup()`: Now cancels and awaits KV monitor task before closing session

2. **`tests/test_server_hooks.py`** (lines 15-36, 83-145):
   - Fixed `test_returns_none_immediately`: Use dict instead of Mock for item assignment
   - Added `test_stores_task_in_app_state`: Verifies task storage
   - Added `TestOnCleanup` class with 4 new tests:
     - `test_cancels_kv_monitor_task`: Verifies task cancellation
     - `test_closes_session`: Verifies session cleanup
     - `test_handles_missing_kv_monitor_task`: Verifies graceful handling when no task
     - `test_handles_cancelled_error_gracefully`: Verifies CancelledError is caught

### Code Changes

**spawn_kv_cache_monitor** (before):
```python
def spawn_kv_cache_monitor(app: web.Application) -> None:
    asyncio.create_task(start_kv_cache_monitoring(app))
```

**spawn_kv_cache_monitor** (after):
```python
def spawn_kv_cache_monitor(app: web.Application) -> None:
    task = asyncio.create_task(start_kv_cache_monitoring(app))
    app["kv_monitor_task"] = task
```

**on_cleanup** (before):
```python
async def on_cleanup(app: web.Application) -> None:
    await app["session"].close()
```

**on_cleanup** (after):
```python
async def on_cleanup(app: web.Application) -> None:
    if "kv_monitor_task" in app:
        task = app["kv_monitor_task"]
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
    await app["session"].close()
```

## L->N->C->P Compliance

| Function | L | N | C | P | Status |
|----------|---|---|---|---|--------|
| `spawn_kv_cache_monitor` | 2 | 0 | 1 | 1 | IDEAL |
| `on_cleanup` | 8 | 2 | 3 | 1 | IDEAL |

All functions well within thresholds.

## Test Results

- **Tests Added**: 5 new tests (1 fixed, 4 new)
- **Total Tests**: 127 tests
- **Pass Rate**: 100% (127/127 passing)

### New Tests Coverage
- Task creation and storage
- Task cancellation on cleanup
- CancelledError handling
- Missing task handling
- Session cleanup still works

## Issues Resolved
- **P2 - KV Monitor Task Leak**: Background task was never stored or cancelled on shutdown
- **Test Fix**: Mock object didn't support item assignment - switched to dict

## Assumptions Made
- The task should be cancelled gracefully, awaiting it after cancellation
- CancelledError is the expected exception when awaiting a cancelled task
- The cleanup order (task cancel, then session close) is correct

## Questions Asked
None - straightforward fix per the task specification.

## Next Steps
Await Yield agent validation.
