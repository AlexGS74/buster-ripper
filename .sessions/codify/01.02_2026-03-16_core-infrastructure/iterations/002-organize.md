# Organize Phase Report

**Session ID**: 01.02_2026-03-16_core-infrastructure
**Topic**: Core Infrastructure - Production Essential Gaps
**Date**: 2026-03-16
**Iteration**: 002

---

## Synthesis: Key Patterns and Gaps

### 1. Foundational Infrastructure Gap (T001)

**Pattern Identified**: Buster-ripper violates the shared session pattern that cloud-relay implements correctly.

**Current State**:
```python
# messages.py, chat.py, compaction.py - ALL do this:
async with aiohttp.ClientSession(...) as session:
    async with session.post(...)
```

**Problem**: Every HTTP request creates a new ClientSession, which means:
- New TCP connection(s) established per request
- No connection reuse across requests
- Resource exhaustion risk under load (4 sessions per user request)
- No centralized lifecycle management

**Gap**: No `on_startup`/`on_cleanup` hooks in server.py for session management.

**Cloud-Relay Reference**:
```python
# server.py lines 349-356
async def on_startup(app):
    app["session"] = aiohttp.ClientSession()
    await detect_backend(app["session"], config.backend_url)

async def on_cleanup(app):
    await app["session"].close()
```

**Synthesis**: T001 is the **foundation** - both T002 (retry logic) and T003 (backend detection) need efficient session access. Without T001, retries would create NEW sessions per attempt, compounding the connection churn problem.

---

### 2. Backend Detection Gap (T003)

**Pattern Identified**: Cloud-relay probes backend type on startup, caches with TTL, uses for request routing.

**What Buster-Ripper Misses**:
- No `backend.py` module exists
- No startup probe to `/v1/models`
- No `owned_by` field inspection
- No backend type caching

**Why It Matters**:
```
SGLang backends require `enable_thinking` parameter
vLLM backends do NOT use this parameter
Without detection: wrong params sent → errors or suboptimal behavior
```

**Implementation Target**: `buster_ripper/backend.py` (~90 lines)
- `detect_backend(session, backend_url)` - async probe function
- `get_backend_state()` - returns cached result
- `send_to_backend(session, config, body, req_id)` - backend-aware request sender

**Synthesis**: T003 is **independent** of T001 but **required by** T002. Backend type determines whether context overflow retry is needed (SGLang-specific behavior).

---

### 3. Context Overflow Retry Gap (T002)

**Pattern Identified**: Cloud-relay parses error messages, extracts token counts, calculates reduced max_tokens, retries.

**Error Pattern**:
```python
# From cloud-relay server.py lines 175-193
error_msg = "BadRequestError: 4000 tokens from the input... exceed context length of 8192"
# Regex extracts: input_tokens=4000, ctx_limit=8192
# Calculation: new_max = max(min_tokens, 8192 - 4000 - 100) = 4092
```

**Current State in Buster-Ripper**:
- `messages.py` lines 58-146: No retry logic, no error parsing
- `chat.py` line 16+: No retry logic
- No `CONTEXT_OVERFLOW_RE` pattern defined
- No `CTX_LIMIT_RE` pattern defined

**Where Retry Logic Goes**:
```python
# messages.py handle_streaming() and handle_non_streaming()
# chat.py handle_chat()

max_retries = 2
for attempt in range(max_retries + 1):
    try:
        async with session.post(...) as resp:
            # Stream response, parse SSE
            # If error detected:
            match = CONTEXT_OVERFLOW_RE.search(error_msg)
            if match and attempt < max_retries:
                input_tokens = int(match.group(1))
                ctx_limit = int(CTX_LIMIT_RE.search(error_msg).group(1))
                new_max = max(MIN_COMPLETION_TOKENS, ctx_limit - input_tokens - 100)
                body["max_completion_tokens"] = new_max
                continue  # retry
    except Exception:
        if attempt == max_retries:
            raise
```

**Synthesis**: T002 **depends on T001** (shared session for efficient retries) and **depends on T003** (backend type to know if retry applies). Without T001, each retry creates a new session → 2-3x connection churn per request.

---

## Dependency Graph (Updated Analysis)

```
                    ┌─────────────────────────┐
                    │   T001: Session Pooling │
                    │   (Foundation Layer)    │
                    └───────────┬─────────────┘
                                │
                    ┌───────────┴─────────────┐
                    │                         │
                    ▼                         │
        ┌───────────────────┐                 │
        │ T003: Backend     │                 │
        │ Detection         │                 │
        │ (Independent)     │                 │
        └───────────┬───────┘                 │
                    │                         │
                    └────────────┬────────────┘
                                 │
                    ┌────────────▼────────────┐
                    │ T002: Context Overflow  │
                    │ Retry                   │
                    │ (Depends on T001 + T003)│
                    └─────────────────────────┘

Implementation Order:
1. T001 - Foundation (session pooling)
2. T003 - Independent (backend detection)
3. T002 - Dependent (retry logic needs both)
```

---

## File-by-File Plan

### Files to Create (1)

| File | Lines | Purpose | Key Functions |
|------|-------|---------|---------------|
| `buster_ripper/backend.py` | ~90 | Backend detection and state management | `detect_backend()`, `get_backend_state()`, `send_to_backend()` |

### Files to Modify (6)

| File | Lines Changed | Change Type | Details |
|------|---------------|-------------|---------|
| `server.py` | +25 | Add hooks | `on_startup` creates session, calls `detect_backend`; `on_cleanup` closes session |
| `config.py` | +10 | Add constants | Backend TTL, retry config, regex patterns |
| `routes/messages.py` | +45 | Replace + Add | Use `request.app["session"]`, add retry loop |
| `routes/chat.py` | +25 | Replace + Add | Use `request.app["session"]`, add retry loop |
| `compaction.py` | +5 | Replace | Use `request.app["session"]` or pass as param |
| `main.py` | +10 | Add CLI opts | `--backend-ttl`, `--min-completion-tokens` |

---

## Risk Assessment

### High-Risk Areas

| Risk | Impact | Mitigation |
|------|--------|------------|
| **Session not closed on crash** | Resource leak | Ensure `on_cleanup` hook registered; add explicit cleanup in `on_shutdown` |
| **Backend detection fails silently** | Wrong backend assumed | Add logging; fallback to "unknown" backend type; probe only on startup |
| **Retry loop creates infinite loop** | Request hangs | Hard `max_retries=2`; decrement or break; timeout on POST request |
| **Background task (poll_kv_cache) loses session access** | Compaction breaks | Pass session as parameter to `poll_kv_cache`; store in app state if needed |

### Medium-Risk Areas

| Risk | Impact | Mitigation |
|------|--------|------------|
| **Regex pattern mismatch** | Retry doesn't trigger on valid error | Test against real SGLang error messages; add fallback to generic retry |
| **Backend type changes at runtime** | Cached type becomes stale | TTL cache (30s); re-probe if backend returns unexpected error |
| **Session timeout on long idle** | First request after idle fails | aiohttp handles reconnection; add keepalive if needed |

### Low-Risk Areas

| Risk | Impact | Mitigation |
|------|--------|------------|
| **CLI options parsing** | Config not loaded | aiohttp already uses argparse; add defaults |
| **Import errors from new module** | Server won't start | Test import in isolation first; no circular dependencies |

---

## Success Criteria (Measurable)

### T001: Client Session Pooling
- [ ] `server.py` has `on_startup` that creates `app["session"]`
- [ ] `server.py` has `on_cleanup` that closes `app["session"]`
- [ ] `messages.py` uses `request.app["session"]` (not `async with aiohttp.ClientSession`)
- [ ] `chat.py` uses `request.app["session"]`
- [ ] `compaction.py` uses shared session
- [ ] No `aiohttp.ClientSession(` in any route handler

### T003: Backend Detection
- [ ] `buster_ripper/backend.py` exists
- [ ] `detect_backend()` function probes `/v1/models`
- [ ] Backend type cached in module state
- [ ] `on_startup` in `server.py` calls `detect_backend`
- [ ] `get_backend_state()` returns cached result

### T002: Context Overflow Retry
- [ ] `CONTEXT_OVERFLOW_RE` pattern defined in `config.py`
- [ ] `CTX_LIMIT_RE` pattern defined in `config.py`
- [ ] `handle_streaming()` has retry loop with `max_retries=2`
- [ ] `handle_non_streaming()` has retry loop
- [ ] `handle_chat()` has retry loop
- [ ] Calculation: `new_max = max(MIN_COMPLETION_TOKENS, ctx_limit - input_tokens - 100)`

---

## Implementation Strategy

### Phase 1: T001 - Session Pooling (Foundation)
**Target**: `server.py`, `config.py`, `messages.py`, `chat.py`, `compaction.py`
**Steps**:
1. Add `on_startup`/`on_cleanup` to `server.py`
2. Replace `async with aiohttp.ClientSession(...)` with `request.app["session"]` in all handlers
3. Test: Single session reused across multiple requests (verify with logs)

### Phase 2: T003 - Backend Detection
**Target**: `backend.py` (new), `server.py`, `config.py`
**Steps**:
1. Create `buster_ripper/backend.py` with `detect_backend()`, `get_backend_state()`
2. Call `detect_backend()` from `server.py on_startup`
3. Add `--backend-ttl` CLI option to `main.py`
4. Test: Backend type detected and logged on startup

### Phase 3: T002 - Context Overflow Retry
**Target**: `config.py`, `messages.py`, `chat.py`
**Steps**:
1. Add regex patterns to `config.py`
2. Wrap backend POST in retry loop in `messages.py`
3. Wrap backend POST in retry loop in `chat.py`
4. Test: Simulate context overflow error; verify retry with reduced tokens

---

## Questions for Next Phase

**No questions asked** - analysis was straightforward based on collect findings.

**Deferred observations**:
1. **Backend TTL configuration**: Should default to 30s (cloud-relay) or startup-only? Cloud-relay uses 30s TTL, but buster-ripper backends rarely change. **Agent recommendation**: Startup-only probe (no TTL), but this can be revisited.

2. **Session in background tasks**: `poll_kv_cache()` in `compaction.py` runs as background task. Should receive session as parameter from `app["session"]` rather than creating new one. **Agent recommendation**: Pass `request.app["session"]` when spawning background task.

---

## Handoff Preparation

**For Decide Phase**:
- Recommended order: T001 → T003 → T002
- Risk mitigation: All high-risk items have mitigations
- Effort estimate: 14 hours total (P0)
- Key decision: Any deviation from cloud-relay pattern? (naming, structure)

**For Invite Phase**:
- User approval needed before implementation
- Clarify: Backend TTL strategy (startup-only vs 30s TTL)
- Clarify: Background task session handling pattern

---

## Files Requiring Changes (Summary)

### Create
- [ ] `buster_ripper/backend.py` (90 lines)

### Modify
- [ ] `buster_ripper/server.py` (+25 lines)
- [ ] `buster_ripper/config.py` (+10 lines)
- [ ] `buster_ripper/routes/messages.py` (+45 lines)
- [ ] `buster_ripper/routes/chat.py` (+25 lines)
- [ ] `buster_ripper/compaction.py` (+5 lines)
- [ ] `buster_ripper/main.py` (+10 lines)

**Total**: 7 files, ~215 lines of changes
