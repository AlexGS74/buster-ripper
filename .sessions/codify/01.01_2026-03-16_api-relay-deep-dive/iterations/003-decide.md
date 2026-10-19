# CODIFY Decide Phase - API Relay Deep Dive Analysis

**Session ID**: 01.01_2026-03-16_api-relay-deep-dive
**Session Directory**: /Users/alexhubitski/Projects/buster-ripper/.sessions/codify/01.01_2026-03-16_api-relay-deep-dive/

## Decision Summary

### Problem Statement

The organize phase has identified critical gaps between buster-ripper's current implementation and production-ready reliability. The core architectural insight is that buster-ripper is a **native Anthropic proxy** (not a translation proxy like cloud-relay), which means:
- Raw SSE passthrough is CORRECT (no conversion needed)
- vLLM's `/anthropic/v1/messages` handles internal format conversion
- The gaps are operational, not architectural

**Essential gaps to address**:
1. No client session reuse (connection churn, latency waste)
2. No context overflow retry (breaks agentic workflows with growing context)
3. No backend detection (blindly injecting SGLang-specific kwargs)
4. No per-request IDs (limited correlation/tracing)
5. Minimal error handling (client disconnects logged as errors)

### Assumptions Made

| Assumption | Rationale | Risk if Wrong |
|------------|-----------|---------------|
| P0 tasks are production-essential | Without these, relay has resource churn, fails on long contexts, incorrect behavior | If incorrect, deployment delayed but no breaking change |
| Backend detection prerequisite for retry | Context overflow parsing requires knowing backend's error format | If wrong, retry may fail but graceful fallback exists |
| 18-hour estimate acceptable | P0+P1 scoped to reasonable increments | If wrong, scope can be reduced |
| Debug dumping already implemented | dump.py exists with session-aware diffing | CORRECTED during organize phase |
| Image agent intercept is niche | Only relevant for multi-modal agentic workflows | If wrong, can be elevated later |

### Viable Options

**Option A: Phased P0-First (Recommended)**
- Implement all P0 tasks first (14h), then P1 (4h)
- Sequence: Backend detection → Session pooling → Context overflow retry → Per-request IDs → Error handling
- Rationale: Production essentials first, then operational excellence

**Option B: Single PR with All Tasks**
- Implement P0+P1 together (18h total)
- Single PR with multiple commits, one per task
- Rationale: Reduces review cycles, all changes tested together

**Option C: Dependency-Ordered Implementation**
- Sequence by dependencies: backend detection → context overflow retry → session pooling → per-request IDs → error handling
- Rationale: Each task builds on previous; backend detection enables correct retry behavior

### Trade-off Analysis

| Option | Pros | Cons | Recommendation |
|--------|------|------|----------------|
| **A: P0-First** | Production-ready after P0; clear milestone | May create artificial cutoff | Good for incremental deployment |
| **B: Single PR** | Atomic change; cohesive review | Larger PR; harder to review | Best for small teams |
| **C: Dependency-Ordered** | Natural flow; each task enables next | Similar to A but clearer rationale | **RECOMMENDED** |

### Risk Assessment

| Risk | Impact | Likelihood | Mitigation |
|------|--------|------------|------------|
| Backend detection fails | Incorrect kwargs injected | Low | Graceful fallback to current behavior |
| Context overflow parsing wrong | Retry fails on overflow errors | Medium | Log original error for debugging |
| Session pooling causes leaks | Connection exhaustion under load | Low | Proper cleanup in on_shutdown |
| Breaking changes to existing clients | Clients may depend on current behavior | Low | Anthropic format is stable |

### Recommended Approach

**Option C (Dependency-Ordered Implementation)** is recommended:

1. **Backend detection first** (4h) - Foundation for correct behavior
2. **Context overflow retry** (6h) - Depends on backend detection for error parsing
3. **Client session pooling** (4h) - Independent infrastructure improvement
4. **Per-request IDs** (2h) - Enables correlation in logs
5. **Enhanced error handling** (2h) - Improves UX but not blocking

This sequence ensures each task builds on the previous, and the critical production essentials (backend detection, retry, pooling) are implemented before operational improvements.

## Implementation Strategy

### Phase 1: P0 Essential (14 hours)

| Task | Dependencies | Files to Modify | Effort |
|------|--------------|-----------------|--------|
| **T003: Backend detection** | None | `server.py` (startup), `config.py` | 4h |
| **T002: Context overflow retry** | T003 | `routes/messages.py`, `config.py` | 6h |
| **T001: Client session pooling** | None | `server.py` (app factory, on_shutdown) | 4h |

### Phase 2: P1 Operational (4 hours)

| Task | Dependencies | Files to Modify | Effort |
|------|--------------|-----------------|--------|
| **T004: Per-request IDs** | None | `routes/messages.py`, `session.py` | 2h |
| **T005: Error handling** | None | `routes/messages.py` | 2h |

### Deferred (P2/P3)

| Task | Reason for Deferral | Revisit When |
|------|---------------------|--------------|
| Image agent intercept | Niche multi-modal use case | Agentic vision workflows needed |
| Keepalive pings | Only useful with image agent | If image agent implemented |
| OpenTelemetry | Substantial work (16h) | Observability becomes critical |
| Full traffic capture | Foundation exists in dump.py | Analytics/monitoring needed |

## Success Criteria

**P0 Essential**:
- [ ] Backend type detected and cached on startup
- [ ] Context overflow errors trigger automatic retry with reduced tokens
- [ ] Single ClientSession reused across all requests
- [ ] No connection churn or resource leaks

**P1 Operational**:
- [ ] All log statements include per-request correlation ID
- [ ] Client disconnections logged at WARNING level
- [ ] Graceful cleanup on disconnect

## Open Questions for Invite

The following questions should be presented to the user during the invite phase:

### Q1: Implementation Sequencing
Should the implementation proceed as:
- **A**: P0 tasks first (14h), then P1 (4h) - phased deployment
- **B**: Single PR with all P0+P1 (18h) - atomic change
- **C**: Dependency-ordered (backend → retry → pooling → IDs → errors) - recommended

**Agent Recommendation**: Option C (dependency-ordered) as it ensures each task builds on the previous.

### Q2: P0 Scope Confirmation
Are all three P0 tasks (backend detection, context overflow retry, session pooling) considered essential before production deployment?

**Agent Recommendation**: Yes - all three address production-readiness gaps.

### Q3: Per-Request ID Format
Should request IDs use:
- **A**: `os.urandom(3).hex()` (6-char hex, like cloud-relay)
- **B**: UUID4 (36-char, more unique)
- **C**: Timestamp-based (readable, less unique)

**Agent Recommendation**: Option A (6-char hex) for consistency with cloud-relay patterns and log readability.

### Q4: Backend Caching Strategy
Should backend detection cache:
- **A**: 30s TTL (like cloud-relay BackendState)
- **B**: Startup-only probe (backend won't change)
- **C**: Per-request probe (always fresh, high overhead)

**Agent Recommendation**: Option B (startup-only) since backend type doesn't change at runtime.

## Questions Asked

No questions were asked during the decide phase - the analysis was based on organized findings from the previous phase.

## Finalized Implementation Plan

| Task ID | Task Name | Priority | Dependencies | Effort | Status |
|---------|-----------|----------|--------------|--------|--------|
| T003 | Backend detection | P0 | None | 4h | Pending |
| T002 | Context overflow retry | P0 | T003 | 6h | Pending |
| T001 | Client session pooling | P0 | None | 4h | Pending |
| T004 | Per-request IDs | P1 | None | 2h | Pending |
| T005 | Enhanced error handling | P1 | None | 2h | Pending |

**Total Effort**: 18 hours (P0: 14h, P1: 4h)
**Recommended Sequence**: T003 → T002 → T001 → T004 → T005
**Deferred**: Image agent intercept (12h), OpenTelemetry (16h), Traffic capture (8h)

## Risk Mitigation Plan

| Risk | Trigger | Action |
|------|---------|--------|
| Backend detection fails | `/v1/models` probe fails | Log warning, default to vLLM behavior |
| Context overflow retry infinite loop | Multiple retries on same request | Cap retries at 2, log original error |
| Session pool exhaustion | Connection errors increase | Add connection health checks |
| Request ID collision | Duplicate IDs in logs | Increase entropy (4 bytes instead of 3) |
