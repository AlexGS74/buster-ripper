# Invite Phase Report

**Session ID**: 01.02_2026-03-16_core-infrastructure
**Topic**: Core Infrastructure - Production Essential Gaps
**Date**: 2026-03-16
**Iteration**: 004

---

## User Collaboration Summary

The invite phase obtained user approval for all planned decisions via deferred mode. The user provided three key approvals that define the implementation approach.

### User Decisions Received

| Decision | User Response | Impact on Plan |
|----------|---------------|----------------|
| Implementation order | "Approve T001→T003→T002" | Confirms foundation-first approach - session pooling first |
| Backend detection strategy | "I approve implementing Auto-detect with flag overrides." | Enables zero-config default with manual control option |
| Retry behavior scope | "A" (SGLang only) | Limits retry logic to known-safety backend |

### Questions Asked

No questions were asked in deference mode - user provided all three approvals directly.

### Plan Modifications

No plan modifications requested. All decisions align with the decide phase recommendations.

### Final User Approval Status

**APPROVED** - All decisions accepted, ready to proceed to Forward phase.

---

## Approved Implementation Plan

### Task Order: T001 → T003 → T002

| Order | Task | Effort | Description |
|-------|------|--------|-------------|
| 1 | T001 (Session Pooling) | 4h | Shared ClientSession for connection reuse |
| 2 | T003 (Backend Detection) | 4h | Auto-detect backend with CLI flag override |
| 3 | T002 (Context Overflow Retry) | 6h | SGLang-only retry on token limit errors |

### Backend Detection Strategy

User approved hierarchical detection approach:

```
1. CLI flag (--backend-type)  ← Manual override
   ↓
2. Auto-detect (/v1/models)    ← Probe on startup
   ↓
3. Fallback ("vllm")           ← Default if probe fails
```

### Retry Behavior Scope

User selected "A" - SGLang only:

- Retry logic only activates when backend type is "sglang"
- Skips retry if backend type is "unknown" (safety first)
- Reduces max_completion_tokens on context overflow errors
- Max retries: 2 with token recalculation

---

## Files to Modify/Create

| File | Action | Lines | Phase |
|------|--------|-------|-------|
| `buster_ripper/backend.py` | Create | ~90 | T003 |
| `buster_ripper/server.py` | Add hooks | +25 | T001+T003 |
| `buster_ripper/config.py` | Add constants | +10 | T001+T002+T003 |
| `buster_ripper/routes/messages.py` | Session + retry | +45 | T001+T002 |
| `buster_ripper/routes/chat.py` | Session + retry | +25 | T001+T002 |
| `buster_ripper/compaction.py` | Replace session | +5 | T001 |
| `buster_ripper/main.py` | Add CLI opts | +10 | T003 |

---

## Success Criteria to Verify

### T001: Client Session Pooling
- `server.py` has `on_startup` that creates `app["session"]`
- `server.py` has `on_cleanup` that closes `app["session"]`
- messages.py, chat.py, compaction.py use `request.app["session"]`
- No `aiohttp.ClientSession(` in any route handler

### T003: Backend Detection
- `buster_ripper/backend.py` exists with `detect_backend()` function
- Backend type probed via `/v1/models` on startup
- `--backend-type` CLI flag available for override
- `get_backend_state()` returns cached result

### T002: Context Overflow Retry
- Regex patterns (`CONTEXT_OVERFLOW_RE`, `CTX_LIMIT_RE`) in `config.py`
- Retry loops in `handle_streaming()`, `handle_non_streaming()`, `handle_chat()`
- Retry only activates for "sglang" backend
- Calculation: `new_max = max(MIN_COMPLETION_TOKENS, ctx_limit - input_tokens - 100)`

---

## Handoff to Forward Phase

**Next Agent**: forward

**Session Data**:
- Session ID: 01.02_2026-03-16_core-infrastructure
- Session Directory: .sessions/codify/01.02_2026-03-16_core-infrastructure/
- Ready for: Incremental implementation of T001 → T003 → T002

**Approved User Choices for Forward**:

1. **decide - Implementation order**:
   - Question: Which task order to use?
   - User Choice: T001 → T003 → T002
   - Impact: Session pooling first establishes foundation for efficient retries
   - Status: RESPECTED

2. **decide - Backend detection strategy**:
   - Question: How should backend detection work?
   - User Choice: Auto-detect with CLI flag overrides
   - Impact: Zero-config default with manual control option
   - Status: RESPECTED

3. **decide - Retry scope**:
   - Question: Which backends should get retry logic?
   - User Choice: SGLang only (option A)
   - Impact: Safer approach, limits to known-good backends
   - Status: RESPECTED
