# Collect Phase Report

**Session ID**: 01.02_2026-03-16_core-infrastructure
**Topic**: Core Infrastructure - Production Essential Gaps
**Date**: 2026-03-16
**Iteration**: 001

---

## Verification Update

**Status**: Collect phase findings verified against current source files (2026-03-16)

**Files Verified**:
- `/Users/alexhubitski/Projects/buster-ripper/buster_ripper/routes/chat.py` - Line 60: Creates NEW `ClientSession` per request
- `/Users/alexhubitski/Projects/buster-ripper/buster_ripper/routes/messages.py` - Lines 77, 114: Creates NEW `ClientSession` per request
- `/Users/alexhubitski/Projects/buster-ripper/buster_ripper/compaction.py` - Line 33: Creates NEW `ClientSession` per request
- `/Users/alexhubitski/Projects/buster-ripper/buster_ripper/server.py` - No `on_startup` session creation, no `on_cleanup` cleanup
- `/Users/alexhubitski/Projects/buster-ripper/buster_ripper/config.py` - No backend detection config, no retry regex patterns
- `/Users/alexhubitski/Projects/buster-ripper/buster_ripper/main.py` - No backend TTL CLI options
- `/Users/alexhubitski/Projects/buster-ripper/buster_ripper/backend.py` - Does NOT exist (confirmed gap)

**Conclusion**: All Collect findings are accurate and current.

---

## Information Gathered

### 1. Backend Detection (T003)

#### Cloud-Relay Pattern (`/Users/alexhubitski/Projects/claude-relay/claude_proxy/backend.py`)

**Detection Logic** (lines 36-79):
- Probes `/v1/models` endpoint on startup
- Checks `owned_by` field in response:
  - `"sglang"` → backend_type = "sglang"
  - `"vllm"` → backend_type = "vllm"
- Fallback: tries `/get_model_info` endpoint if `owned_by` is inconclusive
- Result cached in global `_state` object with TTL (30 seconds default)
- State includes: `model`, `backend_type`, `last_check`

```python
async def detect_backend(session: aiohttp.ClientSession, backend_url: str) -> BackendState:
    # Uses /v1/models endpoint
    # Checks model_info["owned_by"] for "sglang" or "vllm"
    # Falls back to /get_model_info if needed
    # Caches result in global _state with TTL
```

**Key Implementation Details**:
- Line 49: `owned_by = model_info.get("owned_by", "").lower()`
- Lines 51-54: Backend type detection via substring match
- Lines 56-64: Fallback to `/get_model_info` probe
- Line 66: Timestamp cache update
- Lines 82-88: `init_state()` and `get_state()` for state management

**Usage in server.py** (line 17, 352):
- Imported from `backend` module
- Called in `on_startup()` (line 352): `await detect_backend(app["session"], config.backend_url)`
- Result accessed via `get_state()` in `send_to_backend()` (line 102)

#### Buster-Ripper Gap Analysis

**Current State**:
- `/Users/alexhubitski/Projects/buster-ripper/buster_ripper/config.py` - No backend detection
- `/Users/alexhubitski/Projects/buster-ripper/buster_ripper/server.py` - No startup backend probe
- `/Users/alexhubitski/Projects/buster-ripper/buster_ripper/main.py` - No CLI option for backend TTL

**Where Backend Detection Should Be Added**:
1. **New file**: `buster_ripper/backend.py` (mirror cloud-relay pattern)
2. **server.py**: Add `on_startup` backend detection call
3. **main.py**: Add `--backend-ttl` CLI option (optional)

---

### 2. Context Overflow Retry (T002)

#### Cloud-Relay Pattern (`/Users/alexhubitski/Projects/claude-relay/claude_proxy/server.py`)

**Detection and Retry Logic** (lines 132-210):

```python
# Lines 132-136: Max retry setup
max_retries = 2
for attempt in range(max_retries + 1):

# Lines 175-193: Error parsing
error_msg = "Backend returned empty response"
context_overflow = False
for ev in buffered:
    # Check for context overflow error pattern
    m = re.search(r"(\d+) tokens from the input.*?(\d+) tokens for the completion", error_msg)
    if m:
        context_overflow = True
        input_tokens = int(m.group(1))
        m2 = re.search(r"maximum context length of (\d+)", error_msg)
        ctx_limit = int(m2.group(1)) if m2 else config.context_limit

# Lines 195-199: Retry calculation
if context_overflow and attempt < max_retries:
    new_max = max(config.min_completion_tokens, ctx_limit - input_tokens - 100)
    openai_body["max_completion_tokens"] = new_max
    continue  # retry
```

**Key Implementation Details**:
- Line 185: Error regex pattern for token extraction
- Lines 189-190: Context limit extraction from error message
- Line 196: Token calculation: `new_max = max(min_tokens, ctx_limit - input_tokens - 100)`
- Line 199: Body update: `openai_body["max_completion_tokens"] = new_max`
- Line 200: Continue to retry

**Integration Points**:
- `handle_messages()` function (line 58) contains the retry logic
- SSE stream is parsed to detect empty/error responses (lines 155-210)
- Backend session is passed from `app["session"]` (line 60)

#### Buster-Ripper Gap Analysis

**Current State**:
- `/Users/alexhubitski/Projects/buster-ripper/buster_ripper/routes/messages.py`:
  - Line 77-82: Creates NEW `aiohttp.ClientSession` per request (no pooling)
  - No retry logic for context overflow errors
  - No error parsing for token counts

- `/Users/alexhubitski/Projects/buster-ripper/buster_ripper/routes/chat.py`:
  - Line 60-62: Creates NEW `aiohttp.ClientSession` per request
  - No retry logic

**Where Context Overflow Retry Should Be Added**:
1. **messages.py**: Add retry logic to `handle_streaming()` and `handle_non_streaming()` (lines 58-146)
2. **chat.py**: Add retry logic to `handle_chat()` (line 16)
3. **config.py**: Add `CONTEXT_OVERFLOW_REGEX`, `MIN_COMPLETION_TOKENS` constants
4. **Requires**: Shared ClientSession (T001) for efficient retries

**Error Pattern to Implement**:
```python
CONTEXT_OVERFLOW_RE = re.compile(r"(\d+) tokens from the input.*?(\d+) tokens for the completion")
CTX_LIMIT_RE = re.compile(r"maximum context length of (\d+)")
```

---

### 3. Client Session Pooling (T001)

#### Cloud-Relay Pattern (`/Users/alexhubitski/Projects/claude-relay/claude_proxy/server.py`)

**Shared Session Creation** (lines 349-356):

```python
async def on_startup(app: web.Application):
    app["session"] = aiohttp.ClientSession()  # Line 350
    config: ProxyConfig = app["config"]
    await detect_backend(app["session"], config.backend_url)

async def on_cleanup(app: web.Application):
    await app["session"].close()  # Line 356
```

**Usage in Handlers**:
- Line 60: `session: aiohttp.ClientSession = request.app["session"]`
- Line 136: `resp = await send_to_backend(session, config, openai_body, req_id=req_id)`
- Session is passed through to `send_to_backend()` which uses it for all backend calls

#### Buster-Ripper Gap Analysis

**Current State** (ALL create NEW sessions per request):

**messages.py** (lines 77, 114):
```python
# Line 77: Streaming
async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=None)) as session:
    async with session.post(...)

# Line 114: Non-streaming
async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=300)) as session:
```

**chat.py** (line 60):
```python
async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=300)) as session:
```

**compaction.py** (line 33):
```python
async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=5)) as session:
```

**Where Shared Session Should Be Added**:
1. **server.py**: Create session in `on_startup()`, cleanup in `on_shutdown()`
2. **messages.py**: Replace `async with aiohttp.ClientSession(...)` with `request.app["session"]`
3. **chat.py**: Same pattern
4. **compaction.py**: Use shared session instead of creating new one

**Connection Churn Issue**:
- Currently creates 3-4 new sessions per request (messages streaming/non-streaming, chat, compaction poll)
- Each session creates new TCP connections
- No connection reuse, potential resource exhaustion under load

---

## File Inventory

### Files to Create (New)

| File | Purpose | Lines |
|------|---------|-------|
| `buster_ripper/backend.py` | Backend detection and state management | ~90 lines (mirror cloud-relay) |

### Files to Modify

| File | Current Lines | Changes Required |
|------|---------------|------------------|
| `buster_ripper/server.py` | 72 lines | Add `on_startup` session creation, backend detection, `on_cleanup` |
| `buster_ripper/config.py` | 97 lines | Add backend TTL config, context overflow config |
| `buster_ripper/routes/messages.py` | 147 lines | Replace session creation, add retry logic |
| `buster_ripper/routes/chat.py` | 150 lines | Replace session creation, add retry logic |
| `buster_ripper/compaction.py` | 76 lines | Use shared session |
| `buster_ripper/main.py` | 128 lines | Add CLI options for backend TTL |

### Files Already Analyzed (Reference Only)

| File | Purpose |
|------|---------|
| `claude-relay/claude_proxy/backend.py` | Backend detection reference |
| `claude-relay/claude_proxy/server.py` | Session pooling and retry reference |
| `buster_ripper/buster_ripper/session.py` | Session state pattern reference |
| `buster_ripper/buster_ripper/utils.py` | Token extraction utilities |

---

## Implementation Targets

### T003: Backend Detection (4h)

**Target Location**: `buster_ripper/backend.py`

**Implementation Steps**:
1. Copy `backend.py` pattern from cloud-relay
2. Adapt constants to match buster-ripper naming conventions
3. Add to `server.py` startup sequence
4. Test with both sglang and vLLM backends

**Key Functions**:
- `detect_backend(session, backend_url)` - Probe and cache backend type
- `get_state()` - Return cached state
- `send_to_backend(session, config, body, req_id)` - Send request with backend-specific params

---

### T002: Context Overflow Retry (6h)

**Target Locations**:
- `buster_ripper/config.py` - Add regex patterns and config constants
- `buster_ripper/routes/messages.py` - Add retry logic to streaming/non-streaming handlers
- `buster_ripper/routes/chat.py` - Add retry logic to chat handler

**Implementation Steps**:
1. Add regex patterns to config.py (or utils.py)
2. Add retry wrapper around backend requests
3. Parse error messages for token counts
4. Calculate reduced `max_completion_tokens`
5. Retry with reduced tokens

**Key Calculation**:
```python
new_max = max(config.min_completion_tokens, ctx_limit - input_tokens - 100)
```

---

### T001: Client Session Pooling (4h)

**Target Location**: `buster_ripper/server.py`

**Implementation Steps**:
1. Add `on_startup` to create shared session
2. Add `on_cleanup` to close session
3. Update `messages.py` to use `request.app["session"]`
4. Update `chat.py` to use `request.app["session"]`
5. Update `compaction.py` to use shared session

**Session Access Pattern**:
```python
# In handler
session = request.app["session"]

# In background task (poll_kv_cache)
# Need to pass session as parameter or store globally
```

---

## Dependencies Between Tasks

```
T003 (Backend Detection) ──────────────┐
    │                                   │
    ▼                                   │
T002 (Context Overflow Retry) ◀─────────┘
    │                                   │
    │ Requires backend type to know     │
    │ if retry is needed (sglang only)  │
    │                                   │
    └───────────────────────────────────┘
                                        │
    ┌───────────────────────────────────┘
    │
    ▼
T001 (Client Session Pooling) ──────────┘
    │
    │ Required for efficient retry
    │ (T002 needs to reuse session for retries)
    │
    ▼
Foundation for both T002 and T003
```

**Recommended Order**:
1. **T001 First**: Session pooling is foundational - both T002 and T003 need it
2. **T003 Second**: Backend detection is independent and needed for T002
3. **T002 Third**: Depends on both T001 (for retries) and T003 (for backend-aware behavior)

---

## Patterns Observed

### Cloud-Relay → Buster-Ripper Translation

| Cloud-Relay | Buster-Ripper Equivalent |
|-------------|-------------------------|
| `claude_proxy/backend.py` | `buster_ripper/backend.py` (new) |
| `ProxyConfig` | `config` module |
| `send_to_backend()` | Need to add |
| `app["session"]` | Need to add |
| `app["config"]` | `app["config"]` (implicit via module) |

### Module Structure Differences

**Cloud-Relay**:
- Centralized `server.py` with all handlers
- Separate `backend.py` for detection
- Uses `app["config"]` and `app["session"]`

**Buster-Ripper**:
- Routes split across `routes/messages.py`, `routes/chat.py`, etc.
- No backend detection module
- Uses module-level `config` import everywhere
- Creates new `ClientSession` per request in each handler

---

## Questions for Next Phase

1. **Backend Detection**: Should we probe on every request (with TTL cache) or only on startup?
   - Cloud-relay uses TTL (30s), but buster-ripper could use startup-only

2. **Session Sharing in Background Tasks**: `poll_kv_cache()` runs as background task
   - Should session be passed as parameter?
   - Or stored in `app` state?

3. **Retry Scope**: Should context overflow retry apply to both streaming and non-streaming?
   - Cloud-relay implements both

---

## Files Requiring Changes Summary

### New Files
- [ ] `buster_ripper/backend.py` (90 lines - backend detection)

### Modified Files
- [ ] `buster_ripper/server.py` (+20 lines - session creation, cleanup)
- [ ] `buster_ripper/config.py` (+5 lines - backend/overflow config)
- [ ] `buster_ripper/routes/messages.py` (+40 lines - retry logic, session reuse)
- [ ] `buster_ripper/routes/chat.py` (+20 lines - retry logic, session reuse)
- [ ] `buster_ripper/compaction.py` (+5 lines - session reuse)
- [ ] `buster_ripper/main.py` (+10 lines - CLI options)

---

## Session Context for Handoff

**Session Handling**: New session created from parent `01.01_2026-03-16_api-relay-deep-dive`

**Previous Progress**:
- Parent session completed comprehensive analysis of API relay migration
- This session focuses on P0 production essentials (T001, T002, T003)

**Files Examined**:
- `/Users/alexhubitski/Projects/claude-relay/claude_proxy/backend.py`
- `/Users/alexhubitski/Projects/claude-relay/claude_proxy/server.py`
- `/Users/alexhubitski/Projects/buster-ripper/buster_ripper/server.py`
- `/Users/alexhubitski/Projects/buster-ripper/buster_ripper/config.py`
- `/Users/alexhubitski/Projects/buster-ripper/buster_ripper/routes/messages.py`
- `/Users/alexhubitski/Projects/buster-ripper/buster_ripper/routes/chat.py`
- `/Users/alexhubitski/Projects/buster-ripper/buster_ripper/session.py`
- `/Users/alexhubitski/Projects/buster-ripper/buster_ripper/compaction.py`
- `/Users/alexhubitski/Projects/buster-ripper/buster_ripper/main.py`

**Key Finding**: Buster-ripper creates new `aiohttp.ClientSession` for EVERY request in multiple files, causing connection churn. Cloud-relay uses shared session pattern.
