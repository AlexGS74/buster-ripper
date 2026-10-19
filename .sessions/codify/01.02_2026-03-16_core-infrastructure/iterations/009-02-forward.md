# Forward Iteration 009-02: Fix Remaining Codex Issues

## Session
- **Session ID**: 01.02_2026-03-16_core-infrastructure
- **Iteration**: 009-02-forward
- **Date**: 2026-03-04

## Summary

Fixed 2 remaining issues from codex review:

1. **[P1] Session ID stability** - CORRECTED order: normalize FIRST, then compute session_id
2. **[P2] Auth header case-insensitive** - Fixed case-insensitive auth header lookup in chat.py

## Changes Since Iteration 009-01

### [P1] Session ID Stability - CORRECTED

**Previous (incorrect) approach in 009-01**:
- Computed `session_id(body)` BEFORE `normalize()`
- Rationale was wrong: assumed normalize doesn't modify system prompt

**Problem discovered**:
- `normalize()` DOES modify system prompt - it strips the billing nonce
- `session_id()` hashes system prompt content
- Billing nonce changes every request
- So session_id was changing every request → breaks stats/compaction

**Correct approach (this iteration)**:
1. First: `normalize(body, session_id="")` - strips billing nonce
2. Then: `session_id(body)` - computed from normalized body (stable)
3. Finally: Handle images with proper session_id

### buster_ripper/routes/messages.py

**[P1] Session ID stability (lines 30-39)**:
- Changed order: normalize FIRST (empty session_id), THEN compute session_id
- Added image handling AFTER session_id is computed
- Added import for `image_handler`

```python
# Normalize FIRST to strip billing nonce, THEN compute session_id.
body, changes = normalize(body, session_id="")
sid = session_id(body)

# Handle images AFTER computing session_id (for proper cache isolation)
if config.STRIP_IMAGES and image_handler.has_images(body):
    image_handler.strip_and_cache_images(body, sid)
    changes.append("stripped and cached images")
```

### buster_ripper/normalize.py

**Modified normalize() (lines 189-202)**:
- Image handling now conditional on both STRIP_IMAGES AND non-empty session_id
- Empty session_id skips image handling (route will handle images after computing sid)

```python
# 0. Image handling - strip and cache images if enabled AND session_id provided
if config.STRIP_IMAGES and session_id and image_handler.has_images(body):
    image_handler.strip_and_cache_images(body, session_id)
    changes.append("stripped and cached images")
```

### buster_ripper/routes/chat.py

**[P2] Case-insensitive auth header lookup (lines 98-109)**:
- Changed from: `headers.get("authorization", "")` (case-sensitive)
- Changed to: Iterate through keys with case-insensitive comparison

```python
# Handle empty Bearer token (case-insensitive lookup)
for k in list(headers.keys()):
    if k.lower() == "authorization":
        auth = headers[k]
        if auth and auth.strip() in ("Bearer", "Bearer "):
            del headers[k]
        break
```

## L-N-C-P Compliance

### messages.py - `handle_messages`
- **Length (L)**: ~20 functional lines (acceptable)
- **Nesting (N)**: 2 levels (acceptable)
- **Complexity (C)**: ~5 (acceptable)
- **Parameters (P)**: 1 (compliant)

### chat.py - `_prepare_headers`
- **Length (L)**: 9 functional lines
- **Nesting (N)**: 2 levels
- **Complexity (C)**: 4
- **Parameters (P)**: 1

### normalize.py - `normalize`
- **Length (L)**: ~15 functional lines
- **Nesting (N)**: 1 level
- **Complexity (C)**: 3
- **Parameters (P)**: 2

## Tests

All 122 tests passing after changes.

```
122 passed, 21 warnings in 1.61s
```

## Deferred Questions

No questions - all issues had clear fixes from codex review.

## Decisions and Assumptions

1. **Session ID ordering**: Verified that `normalize()` DOES strip billing nonce from system prompt via `_BILLING_RE.sub()`. Therefore session_id must be computed AFTER normalize to be stable.

2. **Two-phase image handling**: Modified normalize() to skip images when session_id is empty, then handle images in route handler after session_id is computed. This ensures images are cached under the correct session.

3. **Auth header case-insensitivity**: HTTP headers are case-insensitive per RFC 7230. The fix iterates through header keys and compares lowercase versions.
