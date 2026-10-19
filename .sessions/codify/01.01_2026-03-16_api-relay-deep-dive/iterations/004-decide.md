# CODIFY Decide Phase - Multi-Session Split Plan

**Session ID**: 01.01_2026-03-16_api-relay-deep-dive
**Session Directory**: /Users/alexhubitski/Projects/buster-ripper/.sessions/codify/01.01_2026-03-16_api-relay-deep-dive/
**Topic**: Session Split Strategy - No Deferrals, Focused Parallel Sessions

## Critical Instruction: NO DEFERRALS

The user has explicitly requested that we do **NOT** defer any tasks. Instead, the work is split into **multiple focused sessions** that can be executed in parallel or sequence.

All identified tasks (P0, P1, P2, P3) from the organize phase will be executed across five independent sessions.

## Task Inventory (Complete)

| Task ID | Description | Priority | Effort | Dependencies |
|---------|-------------|----------|--------|--------------|
| T001 | Client session pooling | P0 | 4h | None |
| T002 | Context overflow retry | P0 | 6h | T003 (backend detection helpful) |
| T003 | Backend detection | P0 | 4h | None |
| T004 | Per-request IDs | P1 | 2h | None |
| T005 | Enhanced error handling | P1 | 2h | None |
| T006 | Image agent intercept | P2 | 12h | SSE state tracking |
| T007 | Keepalive pings | P2 | 2h | T006 (or standalone) |
| T008 | OpenTelemetry integration | P3 | 16h | T004 (per-request IDs helpful) |
| T009 | Traffic capture enhancement | P3 | 8h | dump.py foundation exists |

**Total Effort**: 72 hours across all sessions (56h implementation + 16h tests)

## Test Coverage Requirements

Each session is responsible for delivering **unit tests for all new functions** and **integration tests for new behaviors** introduced by its tasks. Tests are not a separate session — they are a per-session exit criterion.

### Per-Session Test Scope

| Session | Test Scope | Effort | Key Test Cases |
|---------|-----------|--------|----------------|
| 01.02 (Core) | Unit: backend probe, overflow regex, session lifecycle. Integration: retry flow end-to-end | 6h | `test_backend_detection_vllm`, `test_backend_detection_sglang`, `test_overflow_retry_reduces_tokens`, `test_overflow_retry_max_attempts`, `test_session_reuse_across_requests`, `test_session_cleanup_on_shutdown` |
| 01.03 (Ops) | Unit: request ID generation, disconnect detection. Integration: ID propagation in logs | 2h | `test_request_id_uniqueness`, `test_request_id_in_log_output`, `test_client_disconnect_warning_level`, `test_upstream_cleanup_on_disconnect` |
| 01.04 (Image) | Unit: analyzeImage detection, keepalive timing. Integration: tool call interception flow | 4h | `test_analyze_image_detection_in_stream`, `test_tool_call_suppression`, `test_vision_model_stitching`, `test_keepalive_ping_interval` |
| 01.05 (OTel) | Unit: span creation, attribute extraction. Integration: OTLP export verification | 2h | `test_span_attributes_gen_ai`, `test_metrics_counters`, `test_otlp_exporter_config` |
| 01.06 (Traffic) | Unit: schema validation, diff compression. Integration: capture pipeline | 2h | `test_traffic_schema_fields`, `test_differential_compression`, `test_sql_aggregation_views` |

### Test Infrastructure

- **Framework**: `pytest` with `pytest-asyncio` (existing pattern)
- **Mocking**: `mocker` fixture for aiohttp, `AsyncMock` for SSE streams
- **Coverage target**: >80% for new code per session
- **Test location**: `tests/test_<module>.py` following existing convention

## Session Split Rationale

The session boundaries are designed around **independence** and **thematic coherence**:

1. **Core Infrastructure** (Session 1): Production essentials that must work before anything else
2. **Operational Excellence** (Session 2): Debugging and tracing improvements
3. **Multi-Modal Agent Support** (Session 3): Specialized vision workflows
4. **Observability Foundation** (Session 4): Full telemetry stack
5. **Traffic Analytics** (Session 5): Analytics and monitoring layer

Each session is self-contained with its own collect→organize→decide→invite→forward→yield flow.

## Session Dependencies

| Session | Depends On | Rationale |
|---------|------------|-----------|
| 01.02 (Core) | None | Foundation layer |
| 01.03 (Ops) | None | Independent improvements |
| 01.04 (Image) | None | Specialized feature |
| 01.05 (OTel) | None | Standalone integration |
| 01.06 (Traffic) | None | Builds on existing dump.py |

**All sessions are independent** - no strict ordering required. However, the execution order below is recommended for logical flow.

## Execution Order Recommendation

| Order | Session | Reason |
|-------|---------|--------|
| 1 | 01.02 (Core Infrastructure) | Production essentials first |
| 2 | 01.03 (Operational Excellence) | Improves debugging for rest of work |
| 3 | 01.04 (Image Agent) | Specialized feature, depends on stable core |
| 4 | 01.05 (Observability) | Full observability after core is stable |
| 5 | 01.06 (Traffic Analytics) | Analytics layer on top of existing foundation |

**Note**: Sessions can be executed in parallel by different agents if resources allow.

## Session Summaries

### Session 1: Core Infrastructure (01.02)

**Tasks**: T003 (backend detection), T002 (context overflow retry), T001 (session pooling)
**Effort**: 14h implementation + 6h tests = **20 hours**
**Priority**: P0 (Production essential)

This session implements the non-negotiable production requirements:
- Backend type detection enables correct behavior (SGLang vs vLLM)
- Context overflow retry prevents failures with growing context
- Client session pooling eliminates connection churn

**Test exit criteria**: Unit tests for backend probe parsing, overflow regex extraction, session lifecycle; integration test for retry flow end-to-end.

**Why first**: These gaps cause immediate production failures (connection exhaustion, context errors, incorrect kwargs).

### Session 2: Operational Excellence (01.03)

**Tasks**: T004 (per-request IDs), T005 (enhanced error handling)
**Effort**: 4h implementation + 2h tests = **6 hours**
**Priority**: P1 (High value)

This session adds operational polish:
- Per-request correlation IDs enable log tracing
- Client disconnect handling improves error UX

**Test exit criteria**: Unit tests for ID generation and disconnect detection; integration test for ID propagation in log output.

**Why second**: Easier to debug the remaining sessions with proper request IDs and error handling in place.

### Session 3: Multi-Modal Agent Support (01.04)

**Tasks**: T006 (image agent intercept), T007 (keepalive pings)
**Effort**: 14h implementation + 4h tests = **18 hours**
**Priority**: P2 (Specialized feature)

This session enables advanced multi-modal workflows:
- Intercept `analyzeImage` tool calls
- Stitch vision model calls with tool results
- Keepalive pings prevent timeout during long calls

**Test exit criteria**: Unit tests for tool call detection and keepalive timing; integration test for full interception flow with mocked vision model.

**Why third**: Only needed for agentic vision workflows; depends on core being stable.

### Session 4: Observability Foundation (01.05)

**Tasks**: T008 (OpenTelemetry integration)
**Effort**: 16h implementation + 2h tests = **18 hours**
**Priority**: P3 (Observability)

This session adds full observability:
- Distributed tracing with OpenTelemetry
- Span attributes for all request/response flow
- Metrics export to OTLP endpoint

**Test exit criteria**: Unit tests for span creation and attribute extraction; integration test for OTLP exporter config.

**Why fourth**: Substantial standalone work; core metrics/observability patterns already exist in vLLM.

### Session 5: Traffic Analytics (01.06)

**Tasks**: T009 (traffic capture enhancement)
**Effort**: 8h implementation + 2h tests = **10 hours**
**Priority**: P3 (Analytics)

This session enhances traffic analytics:
- Build on existing dump.py foundation
- Session-aware prompt diffing
- Traffic pattern analysis

**Test exit criteria**: Unit tests for schema validation and diff compression; integration test for capture pipeline.

**Why fifth**: Foundation exists (dump.py), enhancement work can happen after core is stable.

## Handoff Instructions

To initiate each new session, use the following pattern:

```bash
# Session 1: Core Infrastructure
claude-refactor codify create-new-session --topic "core-infrastructure" --parent 01

# Session 2: Operational Excellence
claude-refactor codify create-new-session --topic "operational-excellence" --parent 01

# Session 3: Image Agent
claude-refactor codify create-new-session --topic "image-agent" --parent 01

# Session 4: Observability
claude-refactor codify create-new-session --topic "observability" --parent 01

# Session 5: Traffic Analytics
claude-refactor codify create-new-session --topic "traffic-analytics" --parent 01
```

Each session will:
1. Start with its own collect phase
2. Reference this session's findings as context
3. Execute independent collect→organize→decide→invite→forward→yield flow
4. Merge results back to main branch independently

## Session ID Format

Following the CODIFY naming convention:
- `01.02` = Child session of parent 01, first child (core infrastructure)
- `01.03` = Child session of parent 01, second child (operational excellence)
- And so on...

## Questions Asked

No questions were asked - the task inventory and session split rationale was clear from the organize phase findings.

## Assumptions Made

| Assumption | Rationale | Risk if Wrong |
|------------|-----------|---------------|
| All sessions can execute independently | No shared state between task groups | If dependencies exist, merge sessions or reorder |
| 72h total effort is acceptable | 56h implementation + 16h tests; user requested no deferrals | If scope is too large, user can prioritize |
| Parallel execution is possible | Sessions are independent | Resource constraints may require sequential execution |
| Image agent requires stable core | Complex SSE state tracking depends on basic streaming | If core is unstable, image agent work may fail |
| Tests are per-session, not standalone | Each session delivers tested code as exit criterion | If test infrastructure needs shared setup, add conftest early |

## Risk Mitigation

| Risk | Mitigation |
|------|------------|
| Session duplication | Each session has clear boundaries |
| Dependency conflicts | Session 1 (core) should complete first |
| Merge conflicts | Small, focused PRs per session |
| Context loss between sessions | Each session starts with collect phase |
