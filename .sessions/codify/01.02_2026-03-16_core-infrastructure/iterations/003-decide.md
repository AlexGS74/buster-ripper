# Decide Phase Report

**Session ID**: 01.02_2026-03-16_core-infrastructure
**Topic**: Core Infrastructure - Production Essential Gaps
**Date**: 2026-03-16
**Iteration**: 003

---

## Decision Summary

This section documents the final decisions for implementing the three P0 tasks identified in the organize phase.

### Implementation Order Decision

**Decision**: T001 → T003 → T002

The organize phase analysis revealed a critical insight: while the parent session suggested T003 → T002 → T001, the actual dependency chain is different:

| Order | Task | Rationale |
|-------|------|-----------|
| 1 | **T001 (Session Pooling)** | Foundation layer - all HTTP requests need efficient session access. Without this, T002 retries would create 2-3x connection churn per request. |
| 2 | **T003 (Backend Detection)** | Independent module, but needed by T002 to determine if retry logic applies (SGLang-specific). |
| 3 | **T002 (Context Overflow Retry)** | Depends on both T001 (efficient session reuse) and T003 (backend-aware retry). |

**Why this order over parent's suggestion:**
- Parent suggested T003 first (foundation for correct behavior), but T001 is the actual foundation for efficient HTTP operations
- T002 without T001 would work but be inefficient (new session per retry = 3x connections)
- T001 first ensures all subsequent work benefits from connection reuse

---

## Key Decisions

### Decision 1: Backend Detection Strategy

**Options Considered:**
| Option | Description | Pros | Cons |
|--------|-------------|------|------|
| **CLI override + auto-detect (Recommended)** | Auto-detect by default, `--backend-type` flag overrides | Best of both: automatic for most users, manual override when needed | Slightly more complex logic |
| **Startup-only probe** | Probe once on startup, cache indefinitely | Simple, no TTL management | No manual override option |
| **TTL-based (cloud-relay)** | Re-probe every 30s | Handles backend changes | Complexity, unnecessary for stable backends |
| **CLI flag only** | User must specify backend type | Explicit, no probing | Extra configuration burden |

**Decision**: **Auto-detection with CLI override** (hierarchical detection)

**Rationale**:
1. Auto-detection provides zero-config default for most users
2. CLI override (`--backend-type`) allows manual control when needed
3. Fallback to "vllm" if probe fails (most common backend)
4. Hierarchical approach: CLI flag → auto-detect → fallback

**Detection Priority Chain**:
```
1. CLI flag (--backend-type)  ← Highest priority
   ↓
2. Auto-detect (/v1/models probe)
   ↓
3. Fallback ("vllm")          ← Lowest priority
```

**Implementation**:
```python
# backend.py module state
_backend_type: Optional[str] = None
_backend_state_cached: bool = False
_cli_override: Optional[str] = None  # Set from config

async def detect_backend(session: aiohttp.ClientSession, backend_url: str,
                         cli_override: Optional[str] = None) -> str:
    global _backend_type, _backend_state_cached

    # 1. CLI override takes priority
    if cli_override:
        _backend_type = cli_override
        return _backend_type

    # 2. Return cached if available
    if _backend_state_cached:
        return _backend_type

    # 3. Auto-detect via probe
    try:
        async with session.get(f"{backend_url}/v1/models") as resp:
            resp.raise_for_status()
            data = await resp.json()
            owned_by = data.get("data", [{}])[0].get("owned_by", "")
            if "sglang" in owned_by.lower():
                _backend_type = "sglang"
            elif "vllm" in owned_by.lower():
                _backend_type = "vllm"
            else:
                _backend_type = "unknown"
    except Exception as e:
        # 4. Fallback on probe failure
        log.warning(f"Backend probe failed: {e}, defaulting to 'vllm'")
        _backend_type = "vllm"

    _backend_state_cached = True
    return _backend_type
```

**CLI Option** (`main.py`):
```python
@click.option('--backend-type', type=click.Choice(['sglang', 'vllm']),
              help='Override backend type (default: auto-detect)')
def main(backend_type: Optional[str] = None, ...):
    config.BACKEND_TYPE_OVERRIDE = backend_type
    ...
```

---

### Decision 2: Session Access in Background Tasks

**Context**: The `poll_kv_cache()` function in `compaction.py` runs as a background task and needs HTTP session access.

**Options Considered:**
| Option | Description | Pros | Cons |
|--------|-------------|------|------|
| **Pass as parameter (Recommended)** | `request.app["session"]` passed to background task | Explicit, testable, follows cloud-relay pattern | Requires parameter propagation |
| **Global app state** | Store session in module-level global | Simple access | Hidden dependency, harder to test |
| **Create new session** | Background task creates own session | Isolated | Defeats pooling purpose, connection churn |

**Decision**: **Pass session as parameter**

**Rationale**:
1. Matches cloud-relay pattern (proven in production)
2. Makes dependency explicit in function signature
3. Enables testability (mock session in tests)
4. Background task spawned from request context, has access to `request.app["session"]`

**Implementation pattern**:
```python
# In routes or handler
async def some_handler(request):
    session = request.app["session"]
    # Spawn background task with session reference
    asyncio.create_task(poll_kv_cache(session, ...))
```

---

### Decision 3: Effort Estimate Validation

**Original**: 14 hours total (T001: 4h, T003: 4h, T002: 6h)

**Validation**: After reviewing organize phase analysis:
- T001: 4h realistic (server.py hooks + 3 file updates + testing)
- T003: 4h realistic (new module ~90 lines + integration + testing)
- T002: 6h realistic (retry logic in 2 handlers + regex patterns + testing)

**Decision**: **14 hours confirmed** (no adjustment needed)

---

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Session not closed on crash | Medium | Resource leak | `on_cleanup` hook registered in server.py; explicit cleanup in shutdown handler |
| Backend detection fails silently | Low | Wrong backend assumed | Log warning; fallback to "unknown"; probe only on startup |
| Retry loop creates infinite loop | Medium | Request hangs | Hard `max_retries=2`; break on success; timeout on POST |
| Background task loses session | Low | Compaction breaks | Pass session as parameter ( Decision 2) |
| Regex pattern mismatch | Low | Retry doesn't trigger | Test against real SGLang errors; add fallback pattern |

---

## Open Questions for Invite Phase

The following should be presented to the user for approval:

### Question 1: Implementation Order Approval

**Present**: Recommended order T001 → T003 → T002 vs parent's T003 → T002 → T001

**Recommendation**: Proceed with T001 first (foundation for connection reuse)

### Question 2: Backend Detection Strategy

**Present**: Hierarchical detection with CLI override + auto-detect + fallback

- **CLI flag** (`--backend-type`): Manual override when needed
- **Auto-detect**: Probe `/v1/models` and parse `owned_by` field
- **Fallback**: "vllm" if probe fails

**Recommendation**: Approve hierarchical approach (zero-config default + manual override)

### Question 3: Retry Behavior

**Present**: Retry only on SGLang backends (when backend type detected)

**Default**: If backend type is "unknown", skip retry logic (safety first)

---

## Success Criteria

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

## Finalized Implementation Plan

| Order | Task ID | Task Name | Priority | Dependencies | Effort | Files |
|-------|---------|-----------|----------|--------------|--------|-------|
| 1 | T001 | Client session pooling | P0 | None | 4h | `server.py`, `messages.py`, `chat.py`, `compaction.py`, `config.py` |
| 2 | T003 | Backend detection | P0 | T001 | 4h | `backend.py` (new), `server.py`, `main.py` |
| 3 | T002 | Context overflow retry | P0 | T001, T003 | 6h | `config.py`, `messages.py`, `chat.py` |

**Total**: 14 hours (P0: 14h)

**Files Summary**:
| File | Changes | Phase |
|------|---------|-------|
| `buster_ripper/backend.py` | Create (~90 lines) | T003 |
| `buster_ripper/server.py` | Add hooks (+25 lines) | T001+T003 |
| `buster_ripper/config.py` | Add constants (+10 lines) | T001+T002 |
| `buster_ripper/routes/messages.py` | Replace session + retry (+45 lines) | T001+T002 |
| `buster_ripper/routes/chat.py` | Replace session + retry (+25 lines) | T001+T002 |
| `buster_ripper/compaction.py` | Replace session (+5 lines) | T001 |
| `buster_ripper/main.py` | Add CLI opts (+10 lines) | T003 |

---

## Assumptions Made

1. **Backend stability**: Backends (SGLang/vLLM) are stable infrastructure that don't change type at runtime
2. **Session lifecycle**: aiohttp's ClientSession handles reconnection automatically on timeout
3. **Error format consistency**: SGLang context overflow errors follow predictable regex pattern
4. **No breaking changes**: All changes are additive (no API changes for clients)
5. **Cloud-relay as reference**: Patterns from cloud-relay are applicable and proven

---

## Implementation Strategy

**Phase 1: T001 - Session Pooling (4h)**
1. Add `on_startup`/`on_cleanup` hooks to `server.py`
2. Replace `async with aiohttp.ClientSession(...)` with `request.app["session"]` in all handlers
3. Test: Single session reused across multiple requests

**Phase 2: T003 - Backend Detection (4h)**
1. Create `buster_ripper/backend.py` with `detect_backend()`, `get_backend_state()`
2. Call `detect_backend()` from `server.py on_startup`
3. Add `--backend-ttl` CLI option to `main.py`
4. Test: Backend type detected and logged on startup

**Phase 3: T002 - Context Overflow Retry (6h)**
1. Add regex patterns to `config.py`
2. Wrap backend POST in retry loop in `messages.py`
3. Wrap backend POST in retry loop in `chat.py`
4. Test: Simulate context overflow error; verify retry with reduced tokens

---

**📋 Detailed Analysis**: For comprehensive decision rationale, assumptions, and alternative approaches, see `iterations/003-decide.md`
