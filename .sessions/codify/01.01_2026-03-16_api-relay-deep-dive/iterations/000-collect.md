# CODIFY Collect Phase - API Relay Deep Dive Analysis

**Session ID**: 01.01_2026-03-16_api-relay-deep-dive
**Session Directory**: /Users/alexhubitski/Projects/buster-ripper/.sessions/codify/01.01_2026-03-16_api-relay-deep-dive/
**Topic**: Expanded API Relay Migration Analysis with Cloud Relay, vLLM, and Observability
**Session Type**: New Session

## Observations Summary

### Key Finding: Cloud Relay (claude-relay) Features Not in buster-ripper

| Feature | File (claude-relay) | Status in buster-ripper | Priority | Notes |
|---------|---------------------|-------------------------|----------|-------|
| **Image Agent (analyzeImage intercept)** | `image_agent.py` | Partial (strip/cache, no intercept) | P2 | Full image agent would require intercepting `analyzeImage` tool call and stitching vision model call - complex but valuable for multi-modal agentic scenarios |
| **Context Overflow Retry** | `server.py` lines 133-200 | Missing | P1 | Auto-reduces `max_completion_tokens` on context overflow; critical for stable streaming with long history |
| **Debug Request Dumping** | `server.py` lines 89-127 | Missing | P2 | Dumps raw anthropic/openai request bodies to debug/ directory for troubleshooting |
| **Request ID Generation** | `server.py` line 70 | Partial (session_id only) | P3 | `os.urandom(3).hex()` for per-request correlation; buster-ripper uses session_id only |
| **Backend Detection (sglang vs vLLM)** | `backend.py` lines 36-79 | Missing | P1 | Caches backend type; Kwargs injection (`enable_thinking`) is sglang-specific |
| **Client Session Reuse** | `server.py` line 60 | Missing | P1 | Reuses `app["session"]` across requests instead of per-request; improves connection pooling |
| **Stream Error Handling** | `server.py` lines 244-252 | Minimal | P2 | Distinguishes "closing transport" client disconnection from actual errors |
| **Keepalive Pings** | `image_agent.py` lines 394-401 | Missing | P2 | During vision model call, emits SSE ping events every 5s to prevent client timeout |
| **KV Cache State Class** | `backend.py` lines 12-33 | Different | N/A | Cloud relay has TTL-based BackendState; buster-ripper uses session.py stats |
| **Per-request Summary Logging** | `server.py` lines 81-95 | Partial | P3 | Summarizes message types, tool names, system length; buster-ripper logs simpler info |

### vLLM Endpoint Analysis

OpenAI-Compatible Endpoint (`vllm/entrypoints/openai/chat_completions`):
- **Full streaming support with SSE**: Outputs OpenAI SSE format with `data:` chunks
- **Thinking tokens**: Support via `reasoning_content`, `reasoning`, and `chat_template_kwargs` mechanisms
- **Tool calls**: Full support with function calling, including multi-tool parallel calls
- **Tool streaming**: `parallel_tool_calls` parameter, tool result streaming
- **Usage stats**: Free-form usage info with `prompt_tokens`, `completion_tokens`
- **Reasoning effort**: `reasoning_effort` ("low"/"medium"/"high") mapping to thinking budget
- **Structured outputs**: `response_format` support
- **Logprobs**: `logprobs`, `top_logprobs` for token-level probabilities
- **Priority scheduling**: `priority` parameter for request prioritization
- **Request ID**: `request_id` for correlation

Anthropic-Compatible Endpoint (`vllm/entrypoints/anthropic/serving.py`):
- Anthropic Messages API (`/anthropic/v1/messages`)
- SSE streaming with translation from internal OpenAI format
- Full element support: `thinking`, `text`, `tool_use`, `tool_result`, `image`
- Streaming conversion logic with block state tracking (lines 466-770)
- `messages_full_converter` and `message_stream_converter`
- **Does NOT implement** SSE event conversion; converts from OpenAI Chat format

**Gap Analysis for buster-ripper**:
- buster-ripper already handles Anthropic `/v1/messages` → vLLM proxy correctly
- Cloud relay's `convert_stream.py` converts OpenAI SSE → Anthropic SSE; vLLM's Anthropic endpoint does something similar
- **Critical**: buster-ripper's `handle_streaming()` does raw SSE passthrough without conversion
- This means OpenAI SSE goes through untranslated if called via `/v1/messages`

### OpenTelemetry Blueprint

**Spans to Create**:
1. `relay.client_request` - Entry span for incoming requests
2. `relay.upstream_connect` - Time to connect to upstream
3. `relay.upstream_request` - Upstream request duration
4. `relay.stream_conversion` - Time spent in SSE conversion (if applicable)
5. `relay.normalize_request` - Request normalization time
6. `relay.session_update` - Session stats persistence time

**Attributes to Capture**:
- `gen_ai.request.id` - Request/thread ID from client
- `gen_ai.response.model` - Model name
- `gen_ai.usage.prompt_tokens` - Prompt token count
- `gen_ai.usage.completion_tokens` - Completion token count
- `gen_ai.request.max_tokens` - Max tokens requested
- `gen_ai.request.temperature` - Temperature requested
- `gen_ai.request.top_p` - Top-p requested
- `gen_ai.latency.time_to_first_token` - TTFT in milliseconds
- `gen_ai.latency.e2e` - End-to-end latency
- `relay.session.id` - Buster-ripper session ID
- `relay.normalization.changes` - List of normalizations applied
- `relay.backend.type` - "vllm" or "sglang" (if implemented)
- `relay.has_images` - Boolean flag if images present
- `relay.is_streaming` - Boolean flag
- `relay.kv_cache_usage` - Current KV cache utilization (if tracked)

**Metrics to Track**:
- `relay_requests_inflight` - Gauge of concurrent requests
- `relay_requests_total` - Counter (with labels: status, endpoint)
- `relay_requests_duration_seconds` - Histogram (buckets for p50/p95/p99)
- `relay_request_ttft_seconds` - Histogram for TTFT
- `relay_request_tokens_input` - Histogram for input tokens
- `relay_request_tokens_output` - Histogram for output tokens
- `relay_request_errors_total` - Counter (labels: error_type)
- `relay_session_turns_total` - Counter
- `relay_cache_image_hits_total` - Counter
- `relay_cache_image_misses_total` - Counter

**Integrations Needed**:
- `opentelemetry-api` / `opentelemetry-sdk` - core tracing
- `opentelemetry-exporter-otlp` - OTLP exporter
- `opentelemetry-instrumentation-aiohttp` - HTTP instrumentation (optional)
- `opentelemetry-semantic-conventions-ai` - Semantic conventions (already bundled in vLLM)
  - vLLM bundles OTel packages via its distribution

### Streaming Requirements Expansion

**Full Streaming Edge Cases**:
1. **Empty response**: Backend returns `[DONE]` with no content chunks
2. **Early disconnection**: Client closes connection mid-stream
3. **Context overflow**: Backend returns error mid-stream (handled with retry in claude-relay)
4. **Malformed SSE**: Non-standard SSE format or corrupted chunks
5. **Buffer fragmentation**: SSE event split across chunk boundaries
6. **Partial UTF-8**: Multi-byte characters split between chunks
7. **Tool call gaps**: Tool call deltas arrive non-sequentially
8. **Thinking never closes**: Thinking block starts but never gets text_delta
9. **Tool without name**: Tool call delta with null/empty name
10. **Signature emission**: Signature may emit before or after thinking block

**Connection Lifecycle Management**:
1. **Connection pooling**: Reuse aiohttp ClientSession across requests
   - Cloud relay: `app["session"]` shared across all handlers
   - buster-ripper: creates new ClientSession per request in `routes/messages.py`

2. **Client abort handling**:
   - Detect `aiohttp.ClientDisconnectedError` or ConnectionReset
   - Log at WARNING level, not ERROR (as cloud relay does)
   - Clean up upstream connections

3. **Backpressure**: Yield control during long streaming
   - Python's `async for` with `iter_any()` handles naturally
   - No explicit backpressure needed (asyncio handles)

4. **Timeout handling**:
   - Cloud relay: `ClientTimeout(total=1800)` for 30-minute timeout
   - buster-ripper: `ClientTimeout(total=None)` for streaming, 300s for non-streaming

**Error Recovery Patterns**:
1. **Context overflow retry**: Reduce `max_completion_tokens` based on error regex
2. **Backend 5xx errors**: Return `overloaded_error` with status 529
3. **Backend timeout**: Configurable timeout with admin control
4. **Invalid JSON**: Graceful handling with error response
5. **Network errors**: Propagate as 502 with error message

### Traffic Extraction Recommendations

**Immediate (P0 - capture now)**:
1. **Request metadata**:
   - Request ID
   - Model name
   - Stream vs non-stream flag
   - Presence of images
   - Presence of tools
   - Estimated input token count (before sending to backend)
   - System prompt hash

2. **Response metadata**:
   - Actual input/output token counts
   - TTFT (time to first token)
   - Total streaming duration
   - Stop reason
   - Error status if applicable

3. **Session tracking**:
   - Session ID (already implemented in session.py)
   - Turn number within session
   - Running averages (tokens per turn, latency per turn)

**Future (P1 - when ready)**:
1. **Full request/response capture** (configurable via DUMP_DIR)
2. **Differential compression** for multi-turn (diff-based, already in dump.py)
3. **Normalized body representation** (for stable diffing)
4. **Tool call telemetry** (which tools called, argument sizes)

**Data Architecture**:
- SQLite for session stats persistence (already in session.py)
- File-based turn-by-turn diffs (already in dump.py)
- Consider:.addView for aggregated traffic analysis
- Consider: Redis/memcached for production-grade caching

### File Inventory

**Cloud Relay Reference Files**:

| File | Category | Key Contents |
|------|----------|--------------|
| `claude_proxy/sse.py` | SSE Utilities | `SSEEvent`, `make_anthropic_sse()`, `parse_sse_stream()` |
| `claude_proxy/convert_stream.py` | Streaming | `convert_openai_stream_to_anthropic()` - OpenAI→Anthropic SSE conversion |
| `claude_proxy/server.py` | HTTP Server | Request handling, context overflow retry, debug dumping |
| `claude_proxy/image_agent.py` | Images | `has_images()`, `strip_and_cache_images()`, `image_agent_stream()` -拦截 analyzeImage |
| `claude_proxy/normalize.py` | Normalization | Tool sorting, billing stripping, cache control removal, date stripping |
| `claude_proxy/backend.py` | Backend | Backend detection, `send_to_backend()`, model probing |
| `claude_proxy/config.py` | Configuration | Feature flags, timeouts, cache settings |
| `claude_proxy/__main__.py` | CLI | Entry point with argparse configuration |

**vLLM Reference Files**:

| File | Category | Key Contents |
|------|----------|--------------|
| `vllm/entrypoints/openai/chat_completion/serving.py` | OpenAI Chat | Args handling, preprocessing |
| `vllm/entrypoints/openai/chat_completion/protocol.py` | OpenAI Protocol | Pydantic models, parameter validation |
| `vllm/entrypoints/openai/chat_completion/stream_harmony.py` | Streaming | `extract_harmony_streaming_delta()` for GPT_OSS models |
| `vllm/entrypoints/anthropic/serving.py` | Anthropic Protocol | Anthropic→OpenAI conversion, streaming conversion |
| `vllm/entrypoints/openai/api_server.py` | API Server | FastAPI setup, endpoint registration |
| `vllm/tracing/otel.py` | OpenTelemetry | OTel utilities, trace extraction, exporter setup |
| `vllm/tracing/utils.py` | Tracing | Semantic constants, trace headers |
| `vllm/config/observability.py` | Config | OTel endpoint config, detailed traces |

**buster-ripper Current Files**:

| File | Status | Notes |
|------|--------|-------|
| `server.py` | Complete | aiohttp app factory, routes, logging middleware |
| `main.py` | Complete | CLI with click, all config flags |
| `config.py` | Complete | Global config with constants for tunables |
| `session.py` | Complete | Session stats with SQLite persistence |
| `compaction.py` | Complete | KV cache polling and compaction nudge logic |
| `normalize.py` | Complete | Request normalizations (billing, tools, cache_control, date) |
| `convert/anthropic_to_openai.py` | Complete | Anthropic→OpenAI request conversion |
| `convert/sse.py` | Complete | SSE parsing and generation utilities |
| `routes/messages.py` | Partial | Streaming does raw passthrough, missing SSE conversion |
| `routes/chat.py` | Complete | Non-streaming eval-mode proxy with transformations |
| `routes/count_tokens.py` | Complete | Token counting endpoint |
| `routes/passthrough.py` | Complete | Catch-all passthrough |
| `image.py` | Partial | Image caching but no analyzeImage interception |
| `dump.py` | Complete | Session-aware prompt diffing

### Implementation Notes

**Context Overflow Retry (P1 priority)**:
Pattern from claude-relay (lines 132-210 of server.py):
1. Send request with `max_retries = 2` attempts
2. If response is empty, check buffered events for context overflow error
3. Parse error regex: `(\d+) tokens from the input.*?(\d+) tokens for the completion`
4. Calculate new `max_completion_tokens = max(config.min_completion_tokens, ctx_limit - input_tokens - 100)`
5. Retry with reduced token budget

**Backend Detection (P1 priority)**:
Pattern from claude-relay (lines 36-79 of backend.py):
1. Probe `/v1/models` on startup to get model info
2. Check `owned_by` for "sglang" or "vllm"
3. Try sglang-specific endpoint `/get_model_info` as fallback
4. Cache result with 30s TTL via `BackendState`

**Client Session Reuse (P1 priority)**:
Pattern from claude-relay (lines 350, 60 of server.py):
```python
# In create_app:
app["session"] = aiohttp.ClientSession()
# In on_startup cleanup:
await app["session"].close()
# Use app["session"] in handlers instead of creating new ones
```

**Image Agent Intercept (P2 priority)**:
Pattern from claude-relay/image_agent.py:
1. Stream original response
2. Detect `analyzeImage` tool calls in SSE deltas
3. Suppress those deltas from client stream
4. Call vision model with cached images
5. Generate follow-up request with tool result
6. Stream follow-up response
7. Emit keepalive pings during vision call

## Questions Asked

No questions - scope was clear from the prompt.

## Session Context

**Session Handling**: New session created with ID `01.01_2026-03-16_api-relay-deep-dive`

**Previous Progress**: This is a continuation/extension of session `01.00_2026-03-03_api-relay-migration-analysis`, which focused on SSE conversion for thinking tokens. This expanded scope adds:

1. Cloud Relay feature gap analysis
2. vLLM endpoint capability assessment
3. OpenTelemetry integration blueprint
4. Streaming requirements expansion
5. Traffic extraction architecture

**Coordination Context**: This is the initial comprehensive collection for the expanded analysis scope. Previous session completed decision phase with recommendation to adopt claude-relay's convert_stream.py pattern. This session collects broader context for the full API relay migration.

## Additional Findings

1. **Cloud relay requests are NOT converted**: Cloud relay's `server.py` sends OpenAI format to backend, receives OpenAI SSE, then converts to Anthropic SSE for client. buster-ripper's `routes/messages.py` sends Anthropic format and expects Anthropic SSE from upstream vLLM.

2. **vLLM has Anthropic endpoint**: vLLM implements `/anthropic/v1/messages` which already handles Anthropic API format. buster-ripper leverages this by proxying directly rather than converting. However, cloud relay's pattern is conversion from OpenAI backend to Anthropic client.

3. **Session ID differences**:
   - Cloud relay: per-request IDs via `os.urandom(3).hex()`
   - Buster-ripper: per-session IDs via system prompt hash (session.py)

4. **Normalization divergence**:
   - Cloud relay: `_BILLING_RE`, `_DATE_RE`, tool sorting, cache_control stripping
   - Buster-ripper: Same patterns (billing_re, date_re, tool sort, cache_control)

5. **Connection pooling gap**:
   - Cloud relay: Single `ClientSession` shared in `app["session"]`
   - Buster-ripper: New `ClientSession` created per request

6. **OpenTelemetry baseline**:
   - vLLM bundles OTel packages (documented in examples/online_serving/opentelemetry/README.md)
   - Semantic conventions defined in `vllm/tracing/utils.py`
   - OTel support is opt-in via `--otlp-traces-endpoint` flag
   - Request-level instrumentation with `gen_ai.*` attributes already defined

7. **Debug dump directory**:
   - Cloud relay: writes to `debug/` directory (DEBUG_DIR constant)
   - Buster-ripper: uses `DUMP_DIR` with session-based subdirectories

8. **Context overflow handling**:
   - Cloud retry logic is sophisticated: parses error message to calculate new limit based on context_limit - input_tokens - 100
   - Buster-ripper has compaction nudges but no auto-retry mechanism

9. **Streaming state tracking**:
   - vLLM's Anthropic endpoint has complex state machine for streaming conversion (lines 466-770 of serving.py)
   - Cloud relay's `convert_openai_stream_to_anthropic()` has similar state tracking
   - Block state includes: `content_block_index`, `block_type`, `block_index`, `block_signature`, `signature_emitted`, `tool_use_id`

10. **Metrics from vLLM**:
    - Buster-ripper currently uses `/metrics` endpoint to poll KV cache usage
    - vLLM exposes prometheus metrics extensively (see vllm/config/observability.py)
    - vLLM includes OTLP tracing support out-of-the-box

## Priority Summary

**P0 (Critical for parity)**:
- Context overflow retry mechanism
- Backend detection (sglang vs vLLM)
- Client session reuse for connection pooling

**P1 (High value)**:
- Debug request dumping
- Enhanced error handling
- Per-request IDs for better tracing

**P2 (Nice to have)**:
- Image agent full intercept (analyzeImage stitching)
- Keepalive pings during long calls
- Per-request summary logging

**P3 (Future)**:
- OpenTelemetry integration (vLLM already provides patterns)
- Full traffic capture (dump.py already provides foundation)
