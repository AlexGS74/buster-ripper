# Forward Iteration 008-03: Naming Improvements

**Date**: 2026-03-04
**Task**: Improve naming and extract task startup

## Summary

Refactored `server.py` to improve naming clarity:
- Renamed `background_task` to `poll_kv_cache_loop` (describes WHAT it does)
- Extracted `spawn_compaction_task` function for task activation (single-line with clear purpose)

## Changes Since Iteration 008-02

### NEW: Function Renaming and Extraction

**Before:**
```python
async def background_task(app: web.Application) -> None:
    """Background task runner for KV cache polling."""
    # Run the poll_kv_cache task with shared session
    session = app["session"]
    await compaction.poll_kv_cache(session)
```

**After:**
```python
async def poll_kv_cache_loop(app: web.Application) -> None:
    """Poll KV cache for compaction heuristics. Runs for app lifetime."""
    session = app["session"]
    await compaction.poll_kv_cache(session)


def spawn_compaction_task(app: web.Application) -> None:
    """Spawn background task for KV cache polling.

    Non-blocking: creates task and returns immediately.
    Task runs for entire app lifetime, polling at intervals
    to check if compaction is needed.
    """
    asyncio.create_task(poll_kv_cache_loop(app))
```

**And in `on_startup`:**
```python
    # Spawn background compaction polling (non-blocking)
    spawn_compaction_task(app)
```

## Files Modified

| File | Change |
|------|--------|
| `buster_ripper/server.py` | Renamed `background_task` to `poll_kv_cache_loop`, added `spawn_compaction_task` |

## L→N→C→P Compliance

| Function | L | N | C | P | Status |
|----------|---|---|---|---|--------|
| `poll_kv_cache_loop` | 2 | 0 | 1 | 1 | Ideal |
| `spawn_compaction_task` | 1 | 0 | 1 | 1 | Ideal |

All functions within ideal thresholds.

## Tests

**Result**: 105/105 passing (100%)

No new tests needed - this is a pure naming refactoring with no behavioral changes.

## Deferred Questions

None - straightforward naming improvement task.

## Notes

- Naming now describes WHAT the function does, not generic terms
- Task activation is extracted into a dedicated function with clear docstring
- `on_startup` comment updated to match new function name
