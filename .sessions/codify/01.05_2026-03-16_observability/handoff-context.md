# Session Context

## Session Metadata
- **Session ID**: 01.05_2026-03-16_observability
- **Topic**: Observability Foundation - OpenTelemetry Integration
- **Question Mode**: deferred (default)
- **Created**: 2026-03-16
- **Parent Session**: 01.01_2026-03-16_api-relay-deep-dive

## Current Status
- **Current Phase**: Ready for Collect
- **Progress**: Session created from parent session split

## Parent Session Context

This session is a child of `01.01_2026-03-16_api-relay-deep-dive`, which performed comprehensive analysis of API relay migration requirements.

## Scope: Observability Foundation (P3)

**Tasks inherited from parent session**:
| Task ID | Description | Effort | Dependencies |
|---------|-------------|--------|--------------|
| T008 | OpenTelemetry integration | 16h | Per-request IDs helpful |

**Total Effort**: 16 hours
**Priority**: P3 (Observability)

## User Choices Resolved
- **Parent Session Decision**: All P0-P3 tasks scheduled (no deferrals)
  - Impact: This session handles P3 observability foundation
  - Status: RESPECTED

## Key Findings from Parent

1. **OpenTelemetry Spans to Create**:
   - `relay.client_request` - Entry span for incoming requests
   - `relay.upstream_connect` - Time to connect to upstream
   - `relay.upstream_request` - Upstream request duration
   - `relay.stream_conversion` - Time spent in SSE conversion (if applicable)
   - `relay.normalize_request` - Request normalization time
   - `relay.session_update` - Session stats persistence time

2. **Attributes to Capture**:
   - `gen_ai.request.id` - Request/thread ID from client
   - `gen_ai.response.model` - Model name
   - `gen_ai.usage.prompt_tokens` - Prompt token count
   - `gen_ai.usage.completion_tokens` - Completion token count
   - `gen_ai.latency.time_to_first_token` - TTFT in milliseconds
   - `gen_ai.latency.e2e` - End-to-end latency
   - `relay.session.id` - Buster-ripper session ID
   - `relay.backend.type` - "vllm" or "sglang" (if implemented)

3. **Metrics to Track**:
   - `relay_requests_inflight` - Gauge of concurrent requests
   - `relay_requests_total` - Counter (with labels: status, endpoint)
   - `relay_requests_duration_seconds` - Histogram (buckets for p50/p95/p99)
   - `relay_request_ttft_seconds` - Histogram for TTFT
   - `relay_request_tokens_input` - Histogram for input tokens
   - `relay_request_tokens_output` - Histogram for output tokens
   - `relay_request_errors_total` - Counter (labels: error_type)

4. **vLLM Reference**:
   - vLLM bundles OTel packages (documented in examples/online_serving/opentelemetry/README.md)
   - Semantic conventions defined in `vllm/tracing/utils.py`
   - OTel support is opt-in via `--otlp-traces-endpoint` flag

## Recommended Implementation Order

1. **Review vLLM OTel patterns** - Understand semantic conventions
2. **Add core spans** - Entry span, upstream request spans
3. **Capture attributes** - AI semantic conventions (gen_ai.*)
4. **Export metrics** - OTLP endpoint integration
5. **Add observability config** - Enable/disable flags

## Success Criteria

- [ ] All requests traced with entry span
- [ ] Upstream requests traced with correct attributes
- [ ] AI semantic conventions (gen_ai.*) captured
- [ ] Metrics export to OTLP endpoint
- [ ] Configurable via flags (opt-in)

## Notes
- See parent session for full analysis: `01.01_2026-03-16_api-relay-deep-dive`
- This is a substantial standalone integration (16h)
- vLLM provides reference patterns for OTel integration
