# Iteration 007-01: T002 Context Overflow Retry Implementation

**Date**: 2026-03-04
**Phase**: Forward (Implementation)
**Task**: T002 - Context Overflow Retry
**Status**: Complete - Ready for Yield validation

## Summary

Implemented context overflow retry logic for SGLang backends. The retry mechanism detects SGLang context overflow errors and automatically retries requests with reduced `max_completion_tokens`.

## Key Decision: SGLang-Only Retry

Following the approved decision from invite phase: **retry logic ONLY applies to SGLang backends**. If backend is "unknown" or "vllm", the request is made once without retry.

This is enforced by checking `backend.get_backend_state() == "sglang"` before enabling retry loops.

## Files Modified

### 1. `buster_ripper/config.py`
- Added `import re` for regex patterns
- Added `CONTEXT_OVERFLOW_RE` regex pattern
- Added `CTX_LIMIT_RE` regex pattern
- Added `MIN_COMPLETION_TOKENS = 256` constant

### 2. `buster_ripper/utils.py`
- Added `import logging` and `from . import config`
- Added `ContextOverflowError` exception class
- Added `parse_context_overflow_error()` function
- Added `calculate_new_max_tokens()` function

### 3. `buster_ripper/routes/messages.py`
- Added imports for `backend` and retry utilities
- Refactored `handle_streaming()` to include retry loop
- Created `_stream_single_request()` helper function
- Refactored `handle_non_streaming()` to include retry loop
- Created `_non_stream_single_request()` helper function

### 4. `buster_ripper/routes/chat.py`
- Added imports for `backend` and retry utilities
- Refactored `handle_chat()` to include retry loop
- Created `_apply_request_transforms()` helper
- Created `_prepare_headers()` helper
- Created `_make_chat_request()` helper
- Created `_apply_response_transforms()` helper
- Created `_transform_eval_response()` helper
- Created `_transform_cc_response()` helper

### 5. `tests/test_context_overflow.py` (NEW)
- 17 new tests covering:
  - Context overflow error parsing (5 tests)
  - Token calculation (4 tests)
  - Regex patterns (4 tests)
  - Backend-aware retry (4 tests)

## L→N→C→P Compliance

All new functions comply with L→N→C→P metrics:

| Function | Lines | Nesting | Complexity | Parameters |
|----------|-------|---------|------------|------------|
| `parse_context_overflow_error()` | 14 | 2 | 4 | 1 |
| `calculate_new_max_tokens()` | 5 | 0 | 1 | 2 |
| `_stream_single_request()` | 28 | 2 | 4 | 4 |
| `_non_stream_single_request()` | 22 | 1 | 3 | 4 |
| `_apply_request_transforms()` | 15 | 1 | 3 | 2 |
| `_prepare_headers()` | 7 | 0 | 1 | 1 |
| `_make_chat_request()` | 17 | 1 | 3 | 5 |
| `_apply_response_transforms()` | 13 | 1 | 2 | 3 |
| `_transform_eval_response()` | 18 | 1 | 3 | 4 |
| `_transform_cc_response()` | 8 | 1 | 2 | 3 |

All functions are within acceptable thresholds (L≤30, N≤2, C≤10, P≤5).

## Test Results

```
85 passed, 20 warnings in 0.44s
```

- 68 existing tests (unchanged)
- 17 new tests for context overflow retry

## Success Criteria Met

- [x] `CONTEXT_OVERFLOW_RE` pattern defined in `config.py`
- [x] `CTX_LIMIT_RE` pattern defined in `config.py`
- [x] `handle_streaming()` has retry loop with `max_retries=2`
- [x] `handle_non_streaming()` has retry loop
- [x] `handle_chat()` has retry loop
- [x] Calculation: `new_max = max(MIN_COMPLETION_TOKENS, ctx_limit - input_tokens - 100)`

## Assumptions Made

1. **Input tokens capture**: For streaming requests, `input_tokens` is captured from `message_start` event before potential error response. For non-streaming, extracted from response usage.

2. **Error detection**: Context overflow is detected by parsing error response text for the regex pattern.

3. **Retry limit**: Fixed at 2 retries (3 total attempts) as specified in requirements.

4. **Safety margin**: 100 token buffer when calculating new max to prevent edge cases.

## Questions Asked

None - all decisions were pre-approved in invite phase.

## Notes

- The retry logic is completely bypassed for non-SGLang backends
- Error responses are returned to client after max retries exhausted
- Logging includes retry attempt numbers and new token limits
