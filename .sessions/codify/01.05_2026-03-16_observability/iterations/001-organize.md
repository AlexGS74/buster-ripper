# CODIFY Organize Phase - OpenTelemetry Integration Analysis

**Session ID**: 01.05_2026-03-16_observability
**Session Directory**: /Users/alexhubitski/Projects/buster-ripper/.sessions/codify/01.05_2026-03-16_observability/
**Parent Session**: 01.01_2026-03-16_api-relay-deep-dive

## Executive Summary

This analysis synthesizes the OpenTelemetry integration requirements for buster-ripper based on findings from the parent session (01.01). The task is **P3 priority** (observability foundation) with an estimated **16 hours** of implementation effort.

**Core Finding**: OpenTelemetry integration is a **substantial standalone integration**, not a simple "turn on" feature. While vLLM provides reference patterns and bundles OTel packages, the relay-layer tracing must be implemented from scratch with proper span hierarchy, semantic conventions, and metrics export.

**Key Insight**: The integration can be decomposed into **four logical layers** that can be implemented incrementally:
1. **Tracing SDK setup** - OTLP exporter configuration, span processors
2. **Span creation** - Request lifecycle spans with proper parent-child relationships
3. **Attribute capture** - AI semantic conventions (gen_ai.*) and relay-specific attributes
4. **Metrics export** - Counters, gauges, and histograms for operational visibility

**Critical Dependencies**:
- Per-request IDs (T004 from session 01.03) - Required for span correlation
- This task is P3 priority, meaning P0-P1 tasks should ideally be complete first

## Current State Analysis

### What Exists in Buster-Ripper

**Logging Foundation** (from `buster_ripper_legacy.py` and aiohttp refactoring):
- Request logging via middleware (`logging_middleware` in `server.py`)
- Per-request latency tracking in `count_tokens` handler
- Session-level stats tracking in `session.py`

**Request Lifecycle** (from `routes/messages.py`):
```python
async def handle_messages(request: web.Request) -> web.StreamResponse:
    t_start = time.monotonic()
    # ... normalization, dump, session_id extraction ...
    # ... streaming or non-streaming logic ...
    elapsed_ms = (time.monotonic() - t_start) * 1000
    input_tokens = _extract_input_tokens(resp_body)
    _update_session_stats(sid, input_tokens, elapsed_ms)
```

**What's Missing**:
- No distributed tracing capability
- No standardized attribute extraction
- No metrics export to OTLP endpoint
- No span hierarchy or correlation IDs

### vLLM Reference Implementation

**From vLLM's OTel patterns** (per parent session analysis):
- vLLM bundles `opentelemetry-api`, `opentelemetry-sdk`, `opentelemetry-exporter-otlp`
- Semantic conventions defined in `vllm/tracing/utils.py`
- Example configuration in `examples/online_serving/opentelemetry/`
- **Important**: vLLM's OTel traces **model inference** (prefill, decode spans), not relay-layer concerns

**What vLLM provides**:
- SDK dependencies and setup patterns
- `gen_ai.*` semantic convention definitions
- OTLP exporter configuration examples

**What vLLm does NOT provide**:
- Relay-specific spans (`relay.client_request`, `relay.upstream_request`)
- SSE streaming conversion spans
- Session statistics tracking integration

## Gap Analysis

### Tracing Gaps

| Trace Element | vLLM Provides | Buster-Ripper Needs | Gap |
|--------------|---------------|---------------------|-----|
| **SDK Setup** | Reference patterns | Full implementation | **Create from scratch** |
| **Span Hierarchy** | Model spans (prefill/decode) | Relay spans | **Design & implement** |
| **Attribute Extraction** | `gen_ai.*` conventions | Request/response attribute capture | **Integrate into handlers** |
| **OTLP Export** | Example config | Configurable export | **Add CLI flags** |
| **Request ID Correlation** | N/A (model-internal) | Per-request correlation IDs | **Requires T004 first** |

### Metrics Gaps

| Metric | Type | Purpose | Gap |
|--------|------|---------|-----|
| `relay_requests_inflight` | Gauge | Concurrent request tracking | **Create from scratch** |
| `relay_requests_total` | Counter | Total requests (labels: status, endpoint) | **Create from scratch** |
| `relay_request_duration_seconds` | Histogram | End-to-end latency (p50/p95/p99) | **Create from scratch** |
| `relay_request_ttft_seconds` | Histogram | Time-to-first-token | **Create from scratch** |
| `relay_request_tokens_input` | Histogram | Input token counts | **Create from scratch** |
| `relay_request_tokens_output` | Histogram | Output token counts | **Create from scratch** |
| `relay_request_errors_total` | Counter | Error tracking (labels: error_type) | **Create from scratch** |

### Semantic Convention Gaps

| Attribute | Source | Capture Point |
|-----------|--------|---------------|
| `gen_ai.request.id` | Generated | Request start |
| `gen_ai.response.model` | Response JSON | Response parsing |
| `gen_ai.usage.prompt_tokens` | Response `usage.input_tokens` | Response parsing |
| `gen_ai.usage.completion_tokens` | Response `usage.output_tokens` | Response parsing |
| `gen_ai.latency.time_to_first_token` | SSE `message_start` event | Streaming handler |
| `gen_ai.latency.e2e` | Request lifecycle | Request completion |
| `relay.session.id` | Computed from system prompt | `handle_messages` |
| `relay.backend.type` | Backend detection (T003) | If implemented |

## Implementation Strategy

### Recommended Order of Operations

**Phase 1: Foundation (4h)**
1. Add `opentelemetry-api`, `opentelemetry-sdk`, `opentelemetry-exporter-otlp` dependencies
2. Create `opentelemetry.py` module with tracer/meter setup
3. Add OTLP exporter configuration (CLI flags)
4. Basic span creation test

**Phase 2: Span Hierarchy (6h)**
5. Create `relay.client_request` entry span (covers full request lifecycle)
6. Create `relay.upstream_request` child span (covers upstream call)
7. Create `relay.normalize_request` child span (covers normalization)
8. Create `relay.session_update` child span (covers stats persistence)
9. Integrate into `handle_messages()` and `handle_chat()`

**Phase 3: Attribute Capture (4h)**
10. Extract `gen_ai.request.id` from request
11. Capture `gen_ai.response.model` from response
12. Parse `gen_ai.usage.*` from response JSON
13. Capture TTFT from SSE `message_start` event
14. Add relay-specific attributes (session ID, backend type)

**Phase 4: Metrics Export (2h)**
15. Create metric instruments (gauge, counter, histogram)
16. Wire up metric recording at appropriate points
17. Test metrics export to OTLP endpoint

### Dependency Analysis

**Hard Dependencies**:
- T004 (Per-Request IDs) from session 01.03 - **Required** for `gen_ai.request.id`
- No dependency on T001 (Connection Pooling) or T002 (Context Overflow Retry)

**Soft Dependencies**:
- T003 (Backend Detection) helpful for `relay.backend.type` attribute (can use "unknown" if not available)

**Implementation can proceed without T003, but T004 is required first.**

## File-by-File Implementation Plan

### New Files to Create

| File | Purpose | Lines |
|------|---------|-------|
| `buster_ripper/opentelemetry.py` | OTel SDK setup, tracer/meter providers, span helpers | ~150 |
| `buster_ripper/metrics.py` | Prometheus-style metrics wrappers (or use OTel metrics API) | ~100 |

### Files to Modify

| File | Changes |
|------|---------|
| `buster_ripper/main.py` | Add OTel CLI flags (`--otlp-traces-endpoint`, `--otlp-metrics-endpoint`, `--enable-tracing`) |
| `buster_ripper/server.py` | Initialize OTel providers on startup, cleanup on shutdown |
| `buster_ripper/routes/messages.py` | Wrap handlers with spans, capture attributes |
| `buster_ripper/routes/chat.py` | Wrap handlers with spans, capture attributes |
| `buster_ripper/session.py` | Optionally span-annotate session updates |

### CLI Flags to Add

```python
@click.option("--enable-tracing", is_flag=True,
              help="Enable OpenTelemetry tracing and metrics export.")
@click.option("--otlp-traces-endpoint", default="http://localhost:4317",
              show_default=True, help="OTLP gRPC endpoint for traces.")
@click.option("--otlp-metrics-endpoint", default="http://localhost:4317",
              show_default=True, help="OTLP gRPC endpoint for metrics.")
@click.option("--service-name", default="buster-ripper",
              show_default=True, help="Service name for OTel spans.")
```

## Success Criteria

**Must Have (Complete)**:
- [ ] All requests traced with `relay.client_request` entry span
- [ ] Upstream requests traced with `relay.upstream_request` child span
- [ ] `gen_ai.*` semantic conventions captured (request ID, model, token counts)
- [ ] Metrics exported to OTLP endpoint (configurable)
- [ ] Opt-in via CLI flag (disabled by default)

**Should Have (High Quality)**:
- [ ] TTFT captured from SSE `message_start` event
- [ ] Span attributes include session ID for correlation
- [ ] Metrics include proper labels (endpoint, status, model)

**Nice to Have**:
- [ ] Streaming conversion spans (if SSE state tracking added later)
- [ ] Image agent intercept spans (P2 feature, defer)

## Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| **OTel SDK version conflicts** | Low | Medium | Pin exact versions in `pyproject.toml` |
| **Performance overhead** | Medium | Medium | Benchmark with tracing enabled vs disabled; make opt-in |
| **Complexity in streaming handlers** | Medium | Low | Start with non-streaming path, extend to streaming |
| **Missing per-request IDs (T004 not done)** | - | **Blocking** | **MUST complete T004 first** |

## Cross-Session References

**Requires**: Session `01.03_2026-03-16_operational-excellence` (T004: Per-Request IDs)
- Per-request ID generation is foundational for span correlation
- Cannot properly implement `gen_ai.request.id` without this

**Independent from**: T001 (Connection Pooling), T002 (Context Overflow Retry)
- Can implement OTel integration without these, though they would add useful spans

## Task Identification Summary

### T008: OpenTelemetry Integration (16h total)

**Subtask 8.1: SDK Foundation (4h)**
- Add `opentelemetry-api`, `opentelemetry-sdk`, `opentelemetry-exporter-otlp` to dependencies
- Create `buster_ripper/opentelemetry.py` with:
  - TracerProvider setup
  - MeterProvider setup
  - OTLPExporterConfig (traces and metrics)
  - Span processor configuration
  - Context propagation setup
- Add CLI flags for OTel configuration
- Estimated effort: 4 hours

**Subtask 8.2: Span Hierarchy (6h)**
- Implement `relay.client_request` span (full request lifecycle)
- Implement `relay.upstream_request` span (upstream call duration)
- Implement `relay.normalize_request` span (normalization overhead)
- Implement `relay.session_update` span (stats persistence)
- Integrate into route handlers (`handle_messages`, `handle_chat`)
- Implement proper parent-child span relationships
- Handle span context propagation in streaming handlers
- Estimated effort: 6 hours

**Subtask 8.3: Attribute Capture (4h)**
- Capture `gen_ai.request.id` (from per-request ID generator)
- Capture `gen_ai.response.model` (from response JSON)
- Capture `gen_ai.usage.prompt_tokens` and `gen_ai.usage.completion_tokens`
- Capture `gen_ai.latency.time_to_first_token` (from SSE message_start)
- Capture `gen_ai.latency.e2e` (end-to-end latency)
- Capture relay-specific attributes (`relay.session.id`, `relay.backend.type`)
- Estimated effort: 4 hours

**Subtask 8.4: Metrics Export (2h)**
- Create `relay_requests_inflight` gauge (increment on request start, decrement on end)
- Create `relay_requests_total` counter (with labels: status, endpoint)
- Create `relay_request_duration_seconds` histogram (buckets: 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0)
- Create `relay_request_ttft_seconds` histogram (streaming only)
- Create `relay_request_tokens_input` histogram
- Create `relay_request_tokens_output` histogram
- Create `relay_request_errors_total` counter (labels: error_type)
- Wire up metric recording in appropriate handlers
- Estimated effort: 2 hours

### Implementation Sequence

```
T004 (Per-Request IDs) - MUST BE COMPLETE FIRST
  ↓
8.1 SDK Foundation
  ↓
8.2 Span Hierarchy (non-streaming first, then streaming)
  ↓
8.3 Attribute Capture
  ↓
8.4 Metrics Export
```

## Questions Asked

No questions were asked - the analysis was based on parent session findings and established OpenTelemetry patterns. The implementation path is clear with T004 (Per-Request IDs) as a prerequisite.

## Session Handoff

**Next phase**: `decide` agent should review these findings and finalize the implementation approach.

**Key decisions needed**:
1. Confirm T004 (Per-Request IDs) completion status - this is blocking
2. Confirm OTLP endpoint defaults (localhost:4317 gRPC is standard)
3. Confirm tracing is opt-in (disabled by default, enabled via CLI flag)
4. Confirm metric histogram bucket boundaries
5. Approve 16h effort estimate or adjust scope

**For detailed parent findings**: See `01.01_2026-03-16_api-relay-deep-dive/iterations/001-organize.md`
