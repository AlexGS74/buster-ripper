# Forward Iteration 009-01: Fix Codex Findings

## Session
- **Session ID**: 01.02_2026-03-16_core-infrastructure
- **Iteration**: 009-01-forward
- **Date**: 2026-03-04

## Summary

Fixed 4 issues identified by codex review:

1. **[P1] Duplicate upstream requests** - Removed probe approach, detect overflow from actual streaming response
2. **[P1] Case-sensitive header filtering** - Fixed in messages.py
3. **[P2] Case-sensitive header filtering** - Fixed in chat.py
4. **[P2] Empty session_id for images** - Compute session_id BEFORE normalize

## Files Modified

### buster_ripper/routes/messages.py

**[P1] Duplicate upstream requests (lines 67-112)**:
- Removed `_probe_for_context_overflow()` function entirely
- Modified `handle_streaming()` to detect overflow from actual streaming response
- No longer makes duplicate requests - overflow detected from error response

**[P2] Empty session_id for images (lines 30-34)**:
- Changed: Compute `session_id(body)` BEFORE calling `normalize()`
- Rationale: `session_id()` uses system prompt which normalize doesn't modify
- Images now cached under correct session ID instead of empty string

**[P1] Case-sensitive header filtering (lines 55-57)**:
- Changed from: `headers.pop(key, None)` with lowercase keys
- Changed to: Dict comprehension with case-insensitive key matching
```python
skip_keys = {"host", "content-length", "connection", "transfer-encoding"}
headers = {k: v for k, v in request.headers.items() if k.lower() not in skip_keys}
```

### buster_ripper/routes/chat.py

**[P2] Case-sensitive header filtering (lines 98-109)**:
- Same fix as messages.py - case-insensitive header filtering
- Preserved authorization handling logic

## L-N-C-P Compliance

### messages.py - `handle_streaming`
- **Length (L)**: ~20 functional lines (acceptable)
- **Nesting (N)**: 2 levels (acceptable)
- **Complexity (C)**: ~7 (acceptable)
- **Parameters (P)**: 4 (acceptable)

### chat.py - `_prepare_headers`
- **Length (L)**: 8 functional lines
- **Nesting (N)**: 1 level
- **Complexity (C)**: 3
- **Parameters (P)**: 1

## Tests

All 122 tests passing after changes.

```
122 passed, 21 warnings in 1.60s
```

## Deferred Questions

No questions - all issues had clear fixes from codex review.

## Decisions and Assumptions

1. **Session ID ordering**: Assumed `session_id()` only uses system prompt which normalize doesn't modify. Verified by reading `session.py` and `normalize.py` - session_id uses `body.get("system", "")` and normalize strips billing header but doesn't change the system content structure.

2. **Streaming overflow detection**: Instead of probing before streaming, now rely on the existing error handling in `_stream_single_request` which checks status BEFORE preparing response. This allows retry without duplicate requests.

3. **Header filtering**: Used set-based skip list with case-insensitive comparison for O(1) lookup and clean code.
