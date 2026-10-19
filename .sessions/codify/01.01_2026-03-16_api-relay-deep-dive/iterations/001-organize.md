# CODIFY Organize Phase - API Relay Deep Dive Analysis

**Session ID**: 01.01_2026-03-16_api-relay-deep-dive
**Session Directory**: /Users/alexhubitski/Projects/buster-ripper/.sessions/codify/01.01_2026-03-16_api-relay-deep-dive/

## Executive Summary

The analysis reveals a fundamental architectural difference between cloud-relay and buster-ripper that drives most other decisions:

**Core Finding**: Cloud-relay is a **translation proxy** (OpenAI format in, Anthropic format out) while buster-ripper is a **native Anthropic proxy** (Anthropic format both sides, leveraging vLLM's Anthropic endpoint). This architectural choice has cascading implications for streaming, normalization, and feature parity.

**Key Insight**: vLLM provides a native `/anthropic/v1/messages` endpoint that handles Anthropic's SSE format internally, including thinking tokens. This means buster-ripper's "raw passthrough" approach is actually correct for the current architecture and does NOT require the complex conversion logic from cloud-relay's `convert_stream.py`.

**Critical Gaps Identified** (in order of impact):
1. **Connection pooling** - buster-ripper creates new ClientSession per request, which is wasteful and can lead to connection churn
2. **Context overflow retry** - No automatic token reduction on errors; relies on user/app to manage token budgets
3. **Backend detection** - SGLang-specific kwargs (`enable_thinking`) are always injected regardless of backend
4. **Request-scoped IDs** - Only session IDs, no per-request correlation ID
5. **Enhanced error handling** - Client disconnection not detected early

**CORRECTION**: Debug dumping was incorrectly listed as missing. Buster-ripper has `dump.py` which implements session-aware prompt diffing (superior to cloud-relay's raw JSON approach).

## Capability Matrix

| Feature | vLLM Provides | buster-ripper | Cloud Relay | Notes |
|---------|---------------|---------------|-------------|-------|
| **Streaming (thinking tokens)** | Yes - via `/anthropic/v1/messages` | Yes - raw passthrough | Yes - via conversion | vLLM's endpoint does the conversion internally |
| **Streaming (tool calls)** | Yes | Yes (passthrough) | Yes | All support parallel tools |
| **Structured outputs** | Yes (`response_format`) | No (passthrough) | No (passthrough) | vLLM feature if forwarded through |
| **Logprobs** | Yes (`logprobs`, `top_logprobs`) | No (not exposed) | No | vLLM feature if forwarded |
| **Priority scheduling** | Yes (`priority` param) | No | No | vLLM backend feature |
| **Request ID correlation** | Generates `request_id` | Session ID only | Per-request ID | Different scope |
| **Error recovery (context overflow)** | No (returns error) | No | Yes (auto-retry) | Cloud relay独家 |
| **Backend detection** | No ("vllm" only) | No (hardcoded) | Yes (sglang vs vllm) | Cloud relay独家 |
| **Connection pooling** | N/A (requires relay) | No | Yes | Cloud relay uses app["session"] |
| **Debug dumping** | No | **Yes** - `dump.py` with session-aware diffs | No | **CORRECTION**: buster-ripper has SUPERIOR impl (delta compression + session tracking) |
| **Image agent intercept** | No | No | Yes (analyzeImage stitch) | Cloud relay独家 |
| **Keepalive pings** | No | No | Yes | Cloud relay独家 |

### Architecture Comparison Matrix

| Aspect | Cloud Relay | Buster-Ripper | vLLM Endpoint |
|--------|-------------|---------------|---------------|
| **Client API** | Anthropic SSE | Anthropic SSE | Anthropic/UIS openai |
| **Backend API** | OpenAI format | Anthropic format | Anthropic format |
| **Needs conversion** | Yes (OpenAI→Anthropic) | No | No (native) |
| **Streaming logic** | `convert_stream.py` | Raw passthrough | Internal to vLLM |
| **Normalization** | Pre-request | Pre-request | N/A |

## Sufficiency Analysis

### What vLLM Provides "Good Enough"

**Streaming (SSE format)**: vLLM's `/anthropic/v1/messages` endpoint already emits proper Anthropic SSE format including `message_start`, `content_block_delta`, `message_delta` events with thinking tokens. Buster-ripper's raw passthrough is sufficient and actually simpler than cloud-relay's conversion path.

**Message elements**: vLLM supports all Anthropic elements (thinking, text, tool_use, tool_result, image) in its streaming output. No additional logic needed.

**Tool calls**: Full support including parallel tools and streaming tool results.

### What We Need to Build Ourselves

**Connection pooling (P0)**: vLLM doesn't provide relay-layer optimizations. We MUST add this to avoid:
- TCP connection churn
- TLS handshake overhead
- Port exhaustion under load
- Increased latency from cold connections

**Context overflow retry (P0)**: vLLM returns error on overflow; we must:
- Parse error to extract context_limit and input_tokens
- Calculate new `max_completion_tokens`
- Retry request with reduced budget
- This is CRITICAL for agentic scenarios with growing context

**Backend detection (P0)**: Currently we blindly inject `enable_thinking` kwargs which is SGLang-specific. Need to:
- Probe `/v1/models` on startup
- Check `owned_by` field for backend type
- Cache result with TTL
- Only inject backend-specific kwargs appropriately

**Debug dumping (P1)**: **CORRECTION - ALREADY IMPLEMENTED**. Buster-ripper has `dump.py` which provides:
- Session-aware prompt diffing (turn 0 = full JSON, turn N+ = unified diff)
- Human-readable rendered sections ([tools], [system], [message N])
- Index tracking with `index.txt`
- This is SUPERIOR to cloud-relay's raw JSON dumping approach

**Request-scoped IDs (P1)**: Current session IDs are insufficient for:
- Request-level correlation in logs
- Per-request OTel tracing (if/when added)
- Tracking individual requests within a session

**Image agent intercept (P2)**: SGLang/vLLM don't do multi-modal stitching. Need:
- Detect `analyzeImage` tool calls in stream
- Suppress from client
- Call vision model separately
- Stream integrated response
- This is niche but valuable for agentic vision scenarios

**Keepalive pings (P2)**: vLLM doesn't emit them. Need:
- Emit `event: ping` every 5s during long vision calls
- Prevents client-side timeouts
- Only relevant when implementing image agent

### OpenTelemetry Readiness

**vLLM provides patterns, not implementation**:
- vLLM bundles `opentelemetry-api`, `opentelemetry-sdk`, `opentelemetry-exporter-otlp`
- vLLM defines `gen_ai.*` semantic conventions in `vllm/tracing/utils.py`
- vLLM has working examples in `examples/online_serving/opentelemetry/`
- BUT: vLLM's OTel is for model inference, not the relay layer

**We would need to implement**:
- Our own span creation (`relay.client_request`, `relay.upstream_request`, etc.)
- Custom attribute extraction from request/response
- Metrics for in-flight requests, latency, tokens
- OTLP exporter setup

**Recommendation**: Defer to P3. Foundation exists, but full OTel integration is substantial work. Start with logging/per-request IDs and add OTel later.

## Priority Clustering

### P0: Critical for Production Use

| Feature | Dependencies | Effort | Impact |
|---------|--------------|--------|--------|
| **Client Session Reuse** | None | Low | High - reduces latency, connection churn |
| **Context Overflow Retry** | Error parsing, regex extraction | Medium | High - enables agentic use cases with growing context |
| **Backend Detection** | `/v1/models` probe, caching | Medium | High - correct behavior with SGLang vs vLLM |

**Rationale**: These three are production prerequisites. Without them, the relay will have resource churn (connection pooling), fail on long contexts (overflow retry), or inject wrong kwargs (backend detection).

### P1: High Value, Low Complexity

| Feature | Dependencies | Effort | Impact |
|---------|--------------|--------|--------|
| **Per-Request IDs** | `os.urandom` or UUID gen | Low (2h) | High - correlation, tracing foundation |
| **Enhanced Error Handling** | Client disconnection detection | Low (2h) | Medium - better logging, user experience |

**Rationale**: High-value operational improvements with straightforward implementation. Good first tasks after P0.
**Note**: Debug dumping was REMOVED - already implemented in `buster_ripper/dump.py` with session-aware diffing (superior to cloud-relay's approach).

### P2: Nice to Have

| Feature | Dependencies | Effort | Impact |
|---------|--------------|--------|--------|
| **Image Agent Intercept** | SSE state tracking, vision model calling | High | Low-Medium - niche agentic vision workflows |
| **Keepalive Pings** | Timer-based SSE emission | Low | Low - only useful with image agent |

**Rationale**: Valuable for specific workflows but not universally needed. Defer after P0/P1.

### P3: Future Investment

| Feature | Dependencies | Effort | Impact |
|---------|--------------|--------|--------|
| **OpenTelemetry Integration** | OTel SDK, span/attribute code | High | High - observability, but foundation already log-based |
| **Full Traffic Capture** | Dump.py already exists + schemas | Medium | High - knowledge extraction, but can be incremental |

**Rationale**: Both provide significant value but require substantial implementation. Defer until operational foundation is solid.

## Relationships and Dependencies

### Dependency Graph

```
connection_pooling (P0*)
    └── reduces: latency, connection churn
    └── enables: better resource usage

context_overflow_retry (P0*)
    ├── requires: error parsing, regex extraction
    ├── relies on: backend detection (for context_limit if we auto-detect from model)
    └── enables: long-context agentic workflows

backend_detection (P0*)
    ├── requires: /v1/models probe, caching
    └── enables: correct kwargs injection (enable_thinking vs none)

per_request_ids (P1)
    ├── requires: ID generation (os.urandom or UUID)
    └── enables: OTel tracing, log correlation

image_agent_intercept (P2)
    ├── requires: SSE state tracking (similar to convert_stream)
    ├── benefits from: keepalive pings
    └── enables: multi-modal agentic stitching

opentelemetry (P3)
    ├── requires: per-request IDs
    ├── requires: span/attribute extraction
    └── enables: production observability
```

### Cross-Cutting Concerns

**Per-Request IDs**: A hidden dependency for many features:
1. OTel tracing - span correlation
2. Logging - request-level tracking
3. Error handling - context in error messages
4. Debug dumping (already implemented in `dump.py`) - uses session IDs for file correlation

**Session vs Request Scope**:
- Session IDs: Good for long-running stats, KV cache management
- Request IDs: Essential for per-request correlation, OTel, debugging
- **Both needed**: session_id existing, need to add request_id

## Surprises and Contradictions

### Surprise 1: Streaming Conversion Not Actually Needed

**Initial Assumption**: Cloud relay's `convert_openai_stream_to_anthropic()` is critical for supporting thinking tokens.

**Actual Discovery**: vLLM's `/anthropic/v1/messages` endpoint ALREADY does the internal conversion. Buster-ripper's raw passthrough is the CORRECT approach given the native endpoint. Cloud relay needs conversion because it talks to OpenAI format backends; buster-ripper talks to Anthropic format.

**Implication**: We do NOT need to port `convert_stream.py`'s conversion logic. The current implementation is sound.

### Surprise 2: Backend Detection is More Important Than Expected

**Initial Assessment**: SGLang vs vLLM detection is nice-to-have.

**Actual Discovery**: We're blindly injecting `enable_thinking` kwargs which are SGLANG-SPECIFIC. vLLM may ignore them, but this is incorrect behavior and could cause issues if vLLM starts failing on unknown kwargs.

**Implication**: Backend detection should be P0, not P1/P2.

### Contradiction in Streaming "10 Edge Cases"

**Claim**: 10 streaming edge cases identified as requirements.

**Reality Check**: Many of these are edge cases, not operational cases:
- Empty response: Rare, caught by vLLM
- Early disconnection: Client-side, client abort handling (already have)
- Context overflow: Backend behavior, not SSE-specific
- Malformed SSE: vLLM outputs valid SSE
- Buffer fragmentation: aiohttp's `iter_any()` handles
- Partial UTF-8: aiohttp handles with error handling

**True SSE-specific edge cases**:
1. Tool call deltas arriving out of order: Need state tracking
2. Thinking block not closing: Need timeout detection
3. Signature emission timing: Need to handle properly

**Revised Recommendation**: Don't build a "10 edge cases" framework. Handle the 2-3 ACTUAL edge cases that matter with targeted code, not a general-purpose SSE parser.

### Contradiction on OTel Readiness

**Claim**: vLLM provides OTel patterns we can adopt.

**Reality**: vLLM's OTel is for MODEL INFERENCE TRACING, not RELAY LAYER. The spans (`vllm_prefill`, `vllm_decode`) are inside the model engine, not in a proxy. We'd need to implement all relay spans from scratch.

**Correct View**: vLLM gives us the SDK bundles and examples, but NOT implementation. OTel is a full project, not "have them, just turn on".

## Decision Boundaries

### Essential Now (P0)

These MUST be implemented before buster-ripper can be considered production-ready:

1. **Client session reuse** - Prevent resource waste
2. **Context overflow retry** - Enable agentic use cases
3. **Backend detection** - Correct backend-specific behavior

**Boundary**: Without these, the relay is fundamentally broken for production use (connection churn, overflow failures, incorrect behavior).

### High Value Soon (P1)

Implement next for operational excellence:

1. **Per-request IDs** - Correlation foundation
2. **Enhanced error handling** - Better UX

**Boundary**: Nice operational improvements, but can survive without them temporarily.
**Note**: Debug dumping REMOVED - already implemented in `dump.py` (session-aware diffing).

### Deferred (P2/P3)

These can wait until P0/P1 are solid:

1. **Image agent intercept** - Niche use case
2. **Keepalive pings** - Only useful with image agent
3. **OpenTelemetry** - Foundation exists (logging), can add later
4. **Full traffic capture** - Foundation exists (dump.py), can add schema/analysis layer later

## Open Questions

### Q1: Context Overflow - Can We Auto-Detect Context Limit?

Currently cloud relay parses the error message to extract context_limit. Can we:
- Query vLLM's `/metrics` or `/model_info` endpoint for current context limit?
- Cache with TTL like capes-stream does?
- Or should we assume a fixed limit (e.g., from `config.MAX_MODEL_LEN`)?

**Impact**: Affects implementation of context overflow retry complexity.

**Deferred for**: /codify-review

### Q2: Image Agent Intercept API Shape

If we implement image agent intercept, what does the API look like?
- Existing `image.py` already has `strip_and_cache_images` and `get_cached_image`
- Do we need a new `image_agent_intercept()` function?
- What's the integration point in the streaming pipeline?

**Impact**: Design complexity and code placement.

**Deferred for**: /codify-review

### Q3: OTel Exporter Configuration

When we implement OTel, what's the exporter configuration?
- OTLP endpoint default? (localhost:4317 common)
- Protocol? (grpc vs http)
- Sampling rate? (100% for now?)

**Impact**: OTel implementation details, can defer to implementation.

**Deferred for**: /codify-review

### Q4: Connection Session Lifecycle

How long should we keep a shared ClientSession alive?
- Cloud relay: app lifetime (until process exit)
- Should we add periodic refresh to handle stale connections?
- What about TLS cert rotation in upstream?

**Impact**: Connection pooling robustness.

**Deferred for**: /codify-review

## Task Identification Summary (Populating Implementation Plan)

### P0 Tasks (Essential)

**T001: Client session pooling**
- Create shared ClientSession in `server.py` during startup
- Add to `app["session"]` like cloud relay does
- Update all route handlers to use `app["session"]` instead of creating new sessions
- Add cleanup in `on_shutdown` handler
- Estimated effort: 4 hours

**T002: Context overflow retry**
- Implement error regex parser for context overflow message
- Calculate new `max_completion_tokens` based on `context_limit - input_tokens - 100`
- Add retry loop with `max_retries=2`
- Integrate into `handle_streaming()` and `handle_non_streaming()`
- Estimated effort: 6 hours

**T003: Backend detection**
- Add `/v1/models` probe on startup
- Parse `owned_by` field for backend type (sglang vs vllm)
- Cache result with 30s TTL (BackendState pattern from cloud relay)
- Conditionally inject `enable_thinking` only for SGLang
- Estimated effort: 4 hours

### P1 Tasks (High Value) - REVISED

**T004: Per-request IDs** (renamed from T004)
- Add `generate_request_id()` function using `os.urandom(3).hex()` per cloud relay
- Generate at start of `handle_messages()`
- Log with request ID in all relevant log statements
- Add to session update calls (or separate tracking)
- Estimated effort: 2 hours

**T005: Enhanced error handling** (renamed from T006)
- Detect `ClientDisconnectedError` and log at WARNING, not ERROR
- Add `check_if_client_disconnected()` helper
- Clean up upstream connections on disconnect
- Add connection abort handling
- Estimated effort: 2 hours

**REMOVED: T005 (Debug request dumping)** - Already implemented in `buster_ripper/dump.py`.
The existing `dump_turn()` function provides session-aware prompt diffing:
- Turn 0: Saves full JSON to `turn_000.json`
- Turn N+: Saves unified diff to `turn_NNN.diff`
- Human-readable rendered sections ([tools], [system], [message N])
- Already wired into `routes/messages.py` at line 41
- This is SUPERIOR to cloud-relay's raw JSON approach

### P2 Tasks (Nice to Have)

**T007: Image agent intercept**
- Add state tracking for tool call deltas in stream
- Detect `analyzeImage` tool calls
- Suppress from client stream
- Call vision model with cached images
- Stream integrated response
- Add keepalive ping emission (5s intervals)
- Estimated effort: 12 hours (complex state tracking)

### P3 Tasks (Future)

**T008: OpenTelemetry integration**
- Add `opentelemetry-api` SDK dependency
- Create spans: `relay.client_request`, `relay.upstream_request`, etc.
- Extract attributes: `gen_ai.request.id`, `gen_ai.response.model`, etc.
- Create metrics: `relay_requests_inflight`, etc.
- Add OTLP exporter configuration
- Estimated effort: 16 hours (substantial)

**T009: Traffic capture enhancement**
- Add schema definition for request/response capture
- Model: request_id, session_id, timestamp, model, stream flag, token usage
- SQL views for aggregation
- Differential compression (already has dump.py foundation)
- Estimated effort: 8 hours

## Questions Asked

No questions were asked - the analysis was straightforward based on the collect findings. The architectural clarity (native Anthropic proxy vs translation proxy) resolved most ambiguities without need for user input.

## Correction Note (Learning Moment)

**CORRECTION**: Debug request dumping was incorrectly listed as "missing" in the initial analysis.

**What happened**: The analysis looked for cloud-relay's exact pattern (raw JSON to `debug/` directory) and concluded buster-ripper was missing this feature. However, buster-ripper **already has** `dump.py` which implements session-aware prompt diffing:
- Turn 0: Saves full JSON body to `turn_000.json`
- Turn N+: Saves unified diff to `turn_NNN.diff`
- Human-readable rendered sections ([tools], [system], [message N])
- Already wired into `routes/messages.py` via `dump_turn(body)` call at line 41
- Uses `DUMP_DIR` config for output directory
- Maintains session state with turn counting and index tracking

**Impact on effort estimates**:
- P1 tasks reduced from 7 hours to **4 hours** (removed 3h debug dumping task)
- Total P0+P1 reduced from 21 hours to **~18 hours**

**Key learning**: Absence of evidence is not evidence of absence. The feature exists but looks different (and is actually superior to cloud-relay's approach with differential compression and session tracking).

## Session Handoff

Next phase: `decide` agent should review these findings and select the implementation approach.

Key decisions needed from decide phase:
1. P0 scope confirmation - Are all three task clusters truly essential?
2. P1 sequencing - What order to implement P1 tasks?
3. P2/P3 deferral confirmation - OK to defer image agent and OTel?
4. Implementation sequencing - Single monolithic PR vs multiple focused PRs?
