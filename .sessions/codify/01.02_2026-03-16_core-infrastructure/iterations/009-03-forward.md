# Iteration 009-03: Forward - P2 Codex Fixes

## Metadata
- **Phase**: Forward (sub-iteration 009-03)
- **Session**: 01.02_2026-03-16_core-infrastructure
- **Date**: 2026-03-04
- **Status**: COMPLETE - All tests passing

## Summary

Fixed 3 P2 Codex findings related to token count preservation and case-insensitive header handling.

## Changes

### [P2] Token count preserved in messages.py:244
- **Issue**: `parse_context_overflow_error()` extracts `input_tokens` from error text, but `extract_input_tokens(content)` was overwriting it with 0/None on error bodies
- **Fix**: Only use extracted tokens if parsed error didn't have input_tokens

```python
# Before
overflow_err.input_tokens = input_tokens or 0

# After
if overflow_err.input_tokens is None:
    overflow_err.input_tokens = input_tokens or 0
```

### [P2] Token count preserved in chat.py:141
- **Issue**: Same problem - token count from overflow error replaced with `usage.prompt_tokens` which is 0 on error payloads
- **Fix**: Same pattern - use parsed tokens if available, only fallback to usage if not

```python
# Before
input_tokens = resp_data.get("usage", {}).get("prompt_tokens", 0)
overflow_err.input_tokens = input_tokens

# After
if overflow_err.input_tokens is None:
    input_tokens = resp_data.get("usage", {}).get("prompt_tokens", 0)
    overflow_err.input_tokens = input_tokens
```

### [P2] Case-insensitive headers in passthrough.py:16-17
- **Issue**: Using case-sensitive `headers.pop(key, None)` - same issue fixed in messages/chat
- **Fix**: Apply same case-insensitive dict comprehension pattern

```python
# Before
headers = dict(request.headers)
for key in ["host", "content-length", "connection", "transfer-encoding"]:
    headers.pop(key, None)

# After
skip_keys = {"host", "content-length", "connection", "transfer-encoding"}
headers = {k: v for k, v in request.headers.items() if k.lower() not in skip_keys}
```

## Files Modified
1. `buster_ripper/routes/messages.py` - Token count preservation
2. `buster_ripper/routes/chat.py` - Token count preservation
3. `buster_ripper/routes/passthrough.py` - Case-insensitive header filtering

## L→N→C→P Compliance
All changes are simple conditional checks (L:2, N:1, C:1, P:0) - fully compliant.

## Test Results
- **Tests**: 122/122 passing (100%)
- **Coverage**: Existing tests cover these paths
- **Regressions**: None

## Decisions
- Used `None` check for token fallback (consistent with error parsing design)
- Used set lookup for header filtering (consistent with messages/chat pattern)

## Assumptions
- `parse_context_overflow_error()` correctly extracts tokens when pattern matches (validated by existing tests)
- Header case-insensitivity is consistent across all routes (now unified pattern)

## Questions Asked
None - fixes were straightforward from Codex findings.
