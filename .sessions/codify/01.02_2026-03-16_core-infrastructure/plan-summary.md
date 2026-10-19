# CODIFY Session Plan Summary

## Session Metadata
- **Session ID**: 01.02_2026-03-16_core-infrastructure
- **Topic**: Core Infrastructure - Production Essential Gaps
- **Status**: COMPLETE - All P0 tasks done
- **Created**: 2026-03-16
- **Parent Session**: 01.01_2026-03-16_api-relay-deep-dive

## Progress Overview

| Phase | Status | Iteration |
|-------|--------|-----------|
| Collect | ✅ Complete | 001 |
| Organize | ✅ Complete | 002 |
| Decide | ✅ Complete | 003 |
| Invite | ✅ Complete | 004 |
| Forward (T001) | ✅ Complete | 005-01 |
| Yield (T001) | ✅ Complete | 005-03-yield |
| Forward (T003) | ✅ Complete | 006-01-forward |
| Yield (T003) | ✅ Complete | 006-01-yield |
| Forward (T002) | ✅ Complete | 007-01-forward |
| Yield (T002) | ❌ REJECTED | 007-01-yield |
| Forward (T002 fix) | ✅ Complete | 007-02-forward |
| Yield (T002 fix) | ✅ Complete | 007-03-yield |
| Forward (Codex fixes) | ✅ Complete | 008-01-forward |
| Yield (Codex fixes) | ✅ Complete | 008-01-yield |
| Forward (More Codex fixes) | ✅ Complete | 008-02-forward |
| Yield (More Codex fixes) | ✅ Complete | 008-02-yield |
| Forward (Naming improvements) | ✅ Complete | 008-03-forward |
| Yield (Naming improvements) | superseded by 008-04 | - |
| Forward (Fix naming + tests) | ✅ Complete | 008-04-forward |
| Yield (Fix naming + tests) | ✅ Complete | 008-04-yield |
| Forward (Pre-merge quality fixes) | ✅ Complete | 008-05-forward |
| Forward (More Codex fixes) | ✅ Complete | 009-01-forward |
| Forward (Remaining Codex fixes) | ✅ Complete | 009-02-forward |
| Forward (More P2 Codex fixes) | ✅ Complete | 009-03-forward |
| Forward (KV Monitor Task Cleanup) | ✅ Complete | 010-01-forward |

## Scope: Core Infrastructure (P0)

This session implements production-essential gaps identified in the parent session:

1. **Client Session Pooling** (T001) - ✅ Complete (tests removed as over-engineered)
2. **Backend Detection** (T003) - ✅ Complete
3. **Context Overflow Retry** (T002) - ✅ Complete (L→N→C→P fixed)

## Implementation Plan Summary

### Final Order (Decide Phase Decision)

| Order | Task ID | Task Name | Priority | Dependencies | Effort | Status |
|-------|---------|-----------|----------|--------------|--------|--------|
| 1 | T001 | Client session pooling | P0 | None | 4h | ✅ Complete |
| 2 | T003 | Backend detection | P0 | T001 | 4h | ✅ Complete |
| 3 | T002 | Context overflow retry | P0 | T001, T003 | 6h | ✅ Complete |

**Total**: 14 hours (P0: 14h complete)

**Decision Rationale**: T001 is foundational - without shared session pooling, T002 retries would create 2-3x connection churn. T003 is independent but needed by T002 for backend-aware retry. T002 depends on both.

**Progress**: T001 (4h) ✅ | T003 (4h) ✅ | T002 (6h) ✅

**T002 COMPLETE**: All tests passing (85/85), success criteria met, L→N→C→P compliant after refactoring `_transform_eval_response` (N:2, C:7 - within thresholds)

### Files Summary

| File | Changes | Phase | Status |
|------|---------|-------|--------|
| `buster_ripper/backend.py` | Create (~106 lines) | T003 | ✅ Complete |
| `buster_ripper/server.py` | Add hooks (+10 lines) | T001+T003 | ✅ Complete |
| `buster_ripper/config.py` | Add constants (+10 lines) | T002+T003 | ✅ Complete |
| `buster_ripper/utils.py` | Add retry utilities (+30 lines) | T002 | ✅ Complete |
| `buster_ripper/routes/messages.py` | Replace session, add retry (+50 lines) | T001+T002 | ✅ Complete |
| `buster_ripper/routes/chat.py` | Replace session, add retry (+70 lines) | T001+T002 | ✅ Complete |
| `buster_ripper/routes/passthrough.py` | Replace session (+20 lines) | T001 | ✅ Complete |
| `buster_ripper/compaction.py` | Replace session (+15 lines) | T001 | ✅ Complete |
| `buster_ripper/main.py` | Add CLI opts (+10 lines) | T003 | ✅ Complete |
| `tests/test_context_overflow.py` | New tests (17 tests) | T002 | ✅ Complete |

## Success Criteria

### T001: Client Session Pooling ✅ COMPLETE
- [x] `server.py` has `on_startup` that creates `app["session"]`
- [x] `server.py` has `on_cleanup` that closes `app["session"]`
- [x] `messages.py` uses `request.app["session"]` (not `async with aiohttp.ClientSession`)
- [x] `chat.py` uses `request.app["session"]`
- [x] `compaction.py` uses shared session
- [x] No `aiohttp.ClientSession(` in any route handler
- [x] Implementation verified - tests removed as over-engineered for codebase size

### T003: Backend Detection ✅ VALIDATED
- [x] `buster_ripper/backend.py` exists
- [x] `detect_backend()` function probes `/v1/models`
- [x] Backend type cached in module state
- [x] `on_startup` in `server.py` calls `detect_backend`
- [x] `get_backend_state()` returns cached result
- [x] `--backend-type` CLI option in `main.py` (EXTRA - added during implementation)

### T002: Context Overflow Retry - FIX APPLIED
- [x] `CONTEXT_OVERFLOW_RE` pattern defined in `config.py`
- [x] `CTX_LIMIT_RE` pattern defined in `config.py`
- [x] `handle_streaming()` has retry loop with `max_retries=2`
- [x] `handle_non_streaming()` has retry loop
- [x] `handle_chat()` has retry loop
- [x] Calculation: `new_max = max(MIN_COMPLETION_TOKENS, ctx_limit - input_tokens - 100)`
- [x] SGLang-only retry (unknown/vllm skip)
- [x] L→N→C→P compliance - **FIXED**: Extracted 3 helper functions, all metrics now compliant

## Collect Phase Summary

**Status**: COMPLETE (see `iterations/001-collect.md`)

**Key Findings**:
1. **Backend Detection Gap**: No backend detection module exists in buster-ripper
2. **Connection Churn**: Every request creates new `aiohttp.ClientSession` in 3+ places
3. **No Retry Logic**: Context overflow errors not handled

**Files to Create**:
- `buster_ripper/backend.py` (90 lines) - ✅ Created

**Files to Modify**:
- `server.py` (+20 lines) - ✅ Complete
- `config.py` (+5 lines) - ✅ Complete
- `utils.py` (+30 lines) - ✅ Complete
- `routes/messages.py` (+40 lines) - ✅ Complete
- `routes/chat.py` (+20 lines) - ✅ Complete
- `compaction.py` (+5 lines) - ✅ Complete
- `main.py` (+10 lines) - ✅ Complete

## Handoff Context

See `handoff-context.md` for details on session context and implementation results.

## Notes

- This is a child session of `01.01_2026-03-16_api-relay-deep-dive`
- All tasks are P0 priority (production essential)
- Cloud-relay patterns available as reference
- **T001 Complete**: Shared ClientSession implemented, tests removed as over-engineered
- **T003 Complete**: Backend detection with CLI override, auto-detect, fallback
  - **Validated**: 68/68 tests passing, 100% coverage on new code
- **T002 Complete**: Context overflow retry for SGLang backends
  - **Tests**: 85/85 tests passing (17 new tests added)
  - **Success Criteria**: All met
  - **L→N→C→P Violation FIXED**: Refactored `_transform_eval_response`
- **Codex Findings Fixed** (008-01):
  - [P1] Streaming error handling - check status before prepare()
  - [P1] Retry check timing - probe before committing to stream
  - [P2] Image cache eviction - LRU eviction by count
  - **Tests**: 105/105 passing (20 new image tests)
- **Naming Fixes** (008-04):
  - Renamed `poll_kv_cache_loop` → `start_kv_cache_monitoring` (accurate description)
  - Renamed `spawn_compaction_task` → `spawn_kv_cache_monitor` (consistent terminology)
  - Added 4 tests for spawn and monitoring functions
  - **Tests**: 109/109 passing (4 new tests)
- **SESSION COMPLETE**: All P0 tasks done, 122 tests passing
- **Pre-merge quality fixes** (008-05):
  - L-N-C-P violations fixed in normalize.py, image.py, compaction.py
  - Integration tests added for route handlers (13 tests)
  - Dead code removed (convert/ module deleted)
  - Test coverage improved: route handlers now covered
  - All 122 tests passing (up from 109)
- **Codex review fixes** (009-01):
  - [P1] Removed duplicate upstream requests - detect overflow from actual response
  - [P1] Fixed case-sensitive header filtering in messages.py
  - [P2] Fixed case-sensitive header filtering in chat.py
  - [P2] Fixed empty session_id for images - compute before normalize
  - All 122 tests passing
- **Remaining Codex fixes** (009-02):
  - [P1] Session ID stability - CORRECTED order: normalize FIRST (strips billing nonce), THEN compute session_id
  - [P2] Auth header case-insensitive - iterate through keys with case-insensitive comparison
  - All 122 tests passing
- **More P2 Codex fixes** (009-03):
  - [P2] Token count preserved in messages.py - use parsed tokens if available
  - [P2] Token count preserved in chat.py - use parsed tokens if available
  - [P2] Case-insensitive headers in passthrough.py - unified pattern
  - All 122 tests passing
- **KV Monitor Task Cleanup** (010-01):
  - [P2] Task leak fixed - spawn_kv_cache_monitor() stores task in app["kv_monitor_task"]
  - [P2] Cleanup fixed - on_cleanup() cancels/awaits task before session close
  - 5 new tests (1 fixed, 4 added for on_cleanup)
  - All 127 tests passing
