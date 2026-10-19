# CODIFY Session Plan Summary

## Session Metadata
- **Session ID**: 01.01_2026-03-16_api-relay-deep-dive
- **Topic**: API Relay Deep Dive with Cloud Relay, vLLM, and Observability
- **Status**: Collect phase complete
- **Created**: 2026-03-16

## Progress Overview

| Phase | Status | Iteration |
|-------|--------|-----------|
| Collect | Complete | 000-collect |
| Organize | Complete (CORRECTED) | 001-organize, 002-organize |
| Decide | Complete | 003-decide, 004-decide |
| Invite | Pending | - |

## Scope: Expanded API Relay Migration Analysis

This session extends previous analysis (01.00) to include:
1. Cloud Relay (claude-relay) feature gap analysis
2. vLLM Anthropic/OpenAI endpoint capability assessment
3. OpenTelemetry integration blueprint
4. Streaming requirements expansion
5. Traffic extraction architecture recommendations

## Session Type
- **New Session** - Continuing from 01.00 with expanded scope
- **Parent Session**: 01.00_2026-03-03_api-relay-migration-analysis
- **Purpose**: Deeper investigation into cloud relay patterns and vLLM capabilities

## Files Modified/Created This Session
- Created: `.sessions/codify/01.01_2026-03-16_api-relay-deep-dive/iterations/000-collect.md`
- Created: `.sessions/codify/01.01_2026-03-16_api-relay-deep-dive/iterations/001-organize.md`
- Created: `.sessions/codify/01.01_2026-03-16_api-relay-deep-dive/iterations/002-organize.md`
- Created: `.sessions/codify/01.01_2026-03-16_api-relay-deep-dive/iterations/003-decide.md`
- Updated: `.sessions/codify/01.01_2026-03-16_api-relay-deep-dive/handoff-context.md`
- Updated: `.sessions/codify/01.01_2026-03-16_api-relay-deep-dive/plan-summary.md`

## Key Architectural Insights (Organize Phase)

### Core Finding: Native Anthropic Proxy vs Translation Proxy

**Cloud-relay**: Translation proxy (OpenAI requests to backend, Anthropic SSE to client). Requires complex SSE conversion (`convert_stream.py`).

**Buster-ripper**: Native Anthropic proxy (Anthropic format to vLLM's `/anthropic/v1/messages`). Raw passthrough is CORRECT approach.

**Implication**: We do NOT need to port cloud-relay's streaming conversion logic. vLLM's endpoint handles internal conversion.

### P0 Gaps: Production Essentials

| Gap | Impact | Effort |
|-----|--------|--------|
| Client session reuse | High - connection churn, latency waste | Low (4h) |
| Context overflow retry | High - breaks agentic workflows | Medium (6h) |
| Backend detection | High - incorrect SGLang kwargs | Medium (4h) |

### P1 Gaps: Operational Excellence (REVISED)

| Gap | Impact | Effort |
|-----|--------|--------|
| Per-request IDs | High - correlation foundation | Low (2h) |
| Enhanced error handling | Medium - better UX | Low (2h) |

**Note**: Debug dumping REMOVED - already implemented in `buster_ripper/dump.py` with session-aware diffing (superior to cloud-relay's approach).

### Deferred (P2/P3)

Image agent intercept, keepalive pings, OpenTelemetry, full traffic capture.

## Implementation Plan Summary (Decide Phase Approved)

**Update**: User requested NO deferrals. All tasks split into 5 independent sessions.

**Recommended Sequence**: T003 → T002 → T001 → T004 → T005 (dependency-ordered)

### Session Split Strategy

| Session | ID | Tasks | Impl | Tests | Total | Priority |
|---------|-----|-------|------|-------|-------|----------|
| Core Infrastructure | 01.02 | T001, T002, T003 | 14h | 6h | 20h | P0 |
| Operational Excellence | 01.03 | T004, T005 | 4h | 2h | 6h | P1 |
| Multi-Modal Agent | 01.04 | T006, T007 | 14h | 4h | 18h | P2 |
| Observability Foundation | 01.05 | T008 | 16h | 2h | 18h | P3 |
| Traffic Analytics | 01.06 | T009 | 8h | 2h | 10h | P3 |

**Total**: 72 hours across all sessions (56h implementation + 16h tests)

**Test coverage requirement**: Each session delivers >80% coverage for new code. Tests are a per-session exit criterion, not a standalone task.

### P0 Tasks (14h impl + 6h tests = 20h total)

**T003: Backend detection (4h)** - *FIRST - enables correct behavior*
- Probe /v1/models on startup
- Check owned_by for "sglang" vs "vllm"
- Cache result (startup-only, backend doesn't change)
- Conditionally inject enable_thinking only for SGLang

**T002: Context overflow retry (6h)** - *Requires T003*
- Implement error regex: `(\d+) tokens from the input.*?(\d+) tokens for the completion`
- Calculate new max_completion_tokens = context_limit - input_tokens - 100
- Add retry loop with max_retries=2
- Integrate into handle_streaming() and handle_non_streaming()

**T001: Client session pooling (4h)** - *Independent*
- Create shared ClientSession in server.py during startup
- Store in app["session"] per cloud-relay pattern
- Update all route handlers to use shared session
- Add on_shutdown cleanup

**Tests (6h)**: `test_backend_detection_vllm`, `test_backend_detection_sglang`, `test_overflow_retry_reduces_tokens`, `test_overflow_retry_max_attempts`, `test_session_reuse_across_requests`, `test_session_cleanup_on_shutdown`

### P1 Tasks (4h impl + 2h tests = 6h total)

**T004: Per-request IDs (2h)**
- Add generate_request_id() using os.urandom(3).hex()
- Generate at start of handle_messages()
- Log with request ID in all log statements

**T005: Enhanced error handling (2h)**
- Detect ClientDisconnectedError
- Log at WARNING (not ERROR)
- Clean up upstream connections on disconnect

**Tests (2h)**: `test_request_id_uniqueness`, `test_request_id_in_log_output`, `test_client_disconnect_warning_level`, `test_upstream_cleanup_on_disconnect`

### P2/P3 Tasks (Now Scheduled - No Deferrals)

Per user request, all tasks are scheduled across multiple independent sessions:

| Task | Session | Effort |
|------|---------|--------|
| Image agent intercept | 01.04 (Multi-Modal) | 12h |
| Keepalive pings | 01.04 (Multi-Modal) | 2h |
| OpenTelemetry integration | 01.05 (Observability) | 16h |
| Traffic capture enhancement | 01.06 (Traffic Analytics) | 8h |

**For detailed decision rationale, see** `iterations/003-decide.md`
**For session split plan, see** `iterations/004-decide.md`

## Handoff Context
See `handoff-context.md` for details on passing to `decide` agent

## Correction Note (Organize Phase Update)

**CORRECTION APPLIED**: Debug request dumping was incorrectly listed as "missing" in initial analysis.

**Discovery**: Buster-ripper already has `dump.py` which implements session-aware prompt diffing - a SUPERIOR approach to cloud-relay's raw JSON dumping:
- Turn 0: Saves full JSON to `turn_000.json`
- Turn N+: Saves unified diff (delta compression)
- Human-readable rendered sections
- Index tracking with session state

**Effort Impact**:
- P1 tasks: 7 hours → **4 hours** (reduced by 3h)
- Total P0+P1: 21 hours → **~18 hours**

**Learning**: Absence of evidence ≠ evidence of absence. The feature exists but looks different.
