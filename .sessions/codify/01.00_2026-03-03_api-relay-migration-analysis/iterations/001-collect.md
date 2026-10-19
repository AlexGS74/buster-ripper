# Collect Phase: Comprehensive Analysis

## 1. Thinking Tokens Handling (CRITICAL)

### Anthropic API Format
- Thinking blocks arrive as `{"type": "thinking", "thinking": "...", "signature": "..."}` in assistant message content
- Extended thinking config sent via `{"thinking": {"type": "enabled", "budget_tokens": N}}` in request

### Buster-Ripper Implementation
**Request Conversion** (`buster_ripper/convert/anthropic_to_openai.py`):
- `_convert_thinking()` (lines 183-199): Maps Anthropic budget_tokens to OpenAI reasoning params
  - Budget ≤5000 → effort="low"
  - Budget ≤20000 → effort="medium"
  - Budget >20000 → effort="high"
  - Injects: `reasoning`, `enable_thinking`, `chat_template_kwargs`
- `_convert_assistant_message()` (lines 119-166): Extracts thinking blocks from content array
  - Preserves thinking content and signature in `result["thinking"]` dict

**Streaming** (`buster_ripper/routes/messages.py`):
- `handle_streaming()` (lines 58-105): Raw passthrough of upstream SSE
- Uses `find_message_start()` from utils.py to detect message_start event
- Extracts `input_tokens` from usage, tracks TTFT
- **NO thinking-specific handling** - raw passthrough only

**Non-streaming response handling** (`buster_ripper/routes/chat.py`):
- `split_thinking()` in utils.py (lines 39-60): Parses `<think>...</think>` blocks from content
- Lines 84-128: Eval mode and non-eval mode handling
  - Eval mode: strips <think> blocks for lm-eval scoring
  - Non-eval: moves thinking to `reasoning_content` field

### Claude-Relay Reference (`claude_proxy/`)
**Request Conversion** (`convert_request.py`):
- Identical structure to buster-ripper's convert module
- Same `_convert_thinking()`, `_convert_assistant_message()` functions

**Streaming Conversion** (`convert_stream.py`):
- `convert_openai_stream_to_anthropic()` (lines 92-292): Full SSE event-level conversion
- Lines 163-196: Thinking/reasoning handling
  ```python
  # Extract reasoning_content from delta
  reasoning = delta.get("reasoning_content") or ""
  if not reasoning:
      thinking_obj = delta.get("thinking")
      if isinstance(thinking_obj, dict):
          reasoning = thinking_obj.get("content", "")

  if reasoning:
      # Start thinking block if not started
      if not thinking_started:
          yield _build_content_block_start(current_index, {"type": "thinking", "thinking": ""})
      yield _build_content_block_delta(current_index, {"type": "thinking_delta", "thinking": reasoning})
  ```
- Lines 188-196: Signature handling
  ```python
  thinking_obj = delta.get("thinking")
  if isinstance(thinking_obj, dict) and "signature" in thinking_obj:
      yield _build_content_block_delta(current_index, {
          "type": "signature_delta",
          "signature": sig,
      })
  ```

**Key Gap**: Buster-ripper `routes/messages.py` does NOT have streaming conversion logic. It only passes through raw upstream SSE without converting OpenAI SSE → Anthropic SSE format.

---

## 2. Current Implementation State

### Buster-Ripper Files Analyzed

| File | Purpose | Status |
|------|---------|--------|
| `/buster_ripper/server.py` | aiohttp app factory | ✅ Complete |
| `/buster_ripper/routes/messages.py` | POST /v1/messages proxy | ⚠️ Partial - no SSE conversion |
| `/buster_ripper/routes/chat.py` | POST /v1/chat/completions proxy | ✅ Complete (eval mode) |
| `/buster_ripper/convert/anthropic_to_openai.py` | Request conversion | ✅ Present |
| `/buster_ripper/convert/sse.py` | SSE parsing utilities | ✅ Present |
| `/buster_ripper/normalize.py` | Request normalization | ✅ Complete |
| `/buster_ripper/image.py` | Image handling | ✅ Present |
| `/buster_ripper/utils.py` | Helper utilities | ✅ Complete |
| `/buster_ripper/config.py` | Configuration | ✅ Complete |

### What's Working
1. **Request conversion**: Anthropic → OpenAI format in `/convert/anthropic_to_openai.py`
2. **Normalization**: KV cache stability (billing nonce, cache_control, date stripping)
3. **Image handling**: Strip/cache images with TTL
4. **Non-streaming responses**: Full handling in `messages.py`
5. **Eval mode**: Full thinking token handling in `chat.py`

### What's Missing/Partial
1. **Streaming response conversion**: `handle_streaming()` in `messages.py` does raw passthrough
   - No OpenAI SSE → Anthropic SSE conversion
   - No thinking block reconstruction
   - No tool_call conversion
2. **Backend abstraction**: No equivalent to claude-relay's `backend.py`
   - No backend detection (sglang vs vllm)
   - No automatic `chat_template_kwargs` injection for sglang
3. **Image agent**: No `image_agent_stream()` equivalent
   - Image handling is basic strip/cache only
   - No vision model integration

---

## 3. Reference Implementation Gaps

### Claude-Relay Architecture
```
claude-relay/claude_proxy/
├── server.py         # Main handler with full streaming conversion
├── backend.py        # Backend detection, model remapping, sglang injection
├── convert_request.py # Anthropic → OpenAI request conversion
├── convert_stream.py # OpenAI SSE → Anthropic SSE conversion
├── image_agent.py    # Image handling with vision model integration
├── normalize.py      # Request normalization
├── sse.py            # SSE parsing utilities
└── config.py         # Configuration
```

### Buster-Ripper Architecture
```
buster_ripper/
├── server.py         # aiohttp app factory
├── routes/
│   ├── messages.py   # /v1/messages (partial - no SSE conversion)
│   ├── chat.py       # /v1/chat/completions (eval mode)
│   └── passthrough.py# Catch-all
├── convert/
│   ├── anthropic_to_openai.py # Request conversion
│   └── sse.py        # SSE utilities
├── normalize.py       # Request normalization
├── image.py          # Basic image cache
├── utils.py          # Helpers
└── config.py         # Configuration
```

### Specific Missing Components

1. **Backend Detection & Injection** (`backend.py`)
   - Probes `/v1/models` to detect backend type
   - Remaps model name dynamically
   - Injects `chat_template_kwargs` for sglang backends
   - Adds `stream_options: {include_usage: true}` for token tracking

2. **Streaming Conversion** (`convert_stream.py`)
   - Full OpenAI SSE → Anthropic SSE event mapping
   - Thinking block handling with `thinking_started` state
   - Tool call tracking with `tool_call_blocks` dict
   - Signature delta handling
   - Usage aggregation

3. **Image Agent** (`image_agent.py`)
   - Vision model integration
   - analyzeImage tool interception
   - Follow-up request stitching
   - Keepalive pings during vision calls

---

## 4. Test Landscape

### Existing Tests (`/tests/`)
| File | Coverage |
|------|----------|
| `test_normalize.py` | Billing nonce, cache_control, tool sorting, date stripping |
| `test_compaction.py` | KV cache compaction logic |
| `test_session_stats.py` | Session statistics tracking |

### Test Gaps
1. **No tests for `convert/anthropic_to_openai.py`**:
   - Request conversion logic untested
   - Thinking conversion untested
   - Tool conversion untested
   - Image source conversion untested

2. **No tests for routes**:
   - `messages.py` handlers untested
   - `chat.py` handlers untested
   - Streaming behavior untested

3. **No integration tests**:
   - End-to-end proxy behavior
   - Backend interaction

### Testing Infrastructure
- Uses `pytest` with `mocker` fixture (aiohttp mock support)
- `conftest.py` present for shared fixtures
- Pattern: Unit tests for pure functions, mocker.Mock for HTTP

---

## 5. Integration Points

### End-to-End Flow (Current)
```
Client (Anthropic format)
    ↓
POST /v1/messages
    ↓
normalize() → strip billing, cache_control, dates
    ↓
convert_request() → OpenAI format
    ↓
aiohttp.ClientSession → upstream POST /v1/chat/completions
    ↓
Upstream SSE stream (OpenAI format)
    ↓
⚠️ RAW PASSTHROUGH (no conversion!)
    ↓
Client receives OpenAI SSE format (WRONG!)
```

### Required Flow (Target)
```
Client (Anthropic format)
    ↓
POST /v1/messages
    ↓
normalize() → strip billing, cache_control, dates
    ↓
convert_request() → OpenAI format
    ↓
aiohttp.ClientSession → upstream POST /v1/chat/completions
    ↓
Upstream SSE stream (OpenAI format)
    ↓
convert_openai_stream_to_anthropic() → Anthropic SSE format
    ↓
Client receives Anthropic SSE format (CORRECT!)
```

### Key Integration Files to Modify
1. `/buster_ripper/routes/messages.py` - Add streaming conversion
2. `/buster_ripper/convert/` - May need stream conversion module
3. Potentially create `/buster_ripper/backend.py` for backend abstraction

---

## 6. Summary of Findings

### Critical Gap
The streaming response path in `messages.py` (`handle_streaming()`, lines 58-105) performs **raw passthrough** of upstream SSE events without converting from OpenAI format to Anthropic format. This means:
- Clients expecting Anthropic SSE events will receive OpenAI SSE events
- Thinking tokens in `reasoning_content` or `delta.thinking` are NOT converted to `thinking` blocks
- Tool calls are NOT converted from OpenAI to Anthropic format

### Code Pattern from Claude-Relay
The `convert_openai_stream_to_anthropic()` function in claude-relay provides the complete streaming conversion pattern:
1. Parse OpenAI SSE events
2. Track state: `thinking_started`, `text_started`, `tool_call_blocks`
3. Yield appropriate Anthropic SSE events:
   - `message_start`
   - `content_block_start` (thinking, text, tool_use)
   - `content_block_delta` (thinking_delta, text_delta, input_json_delta)
   - `content_block_stop`
   - `message_delta`
   - `message_stop`

### Thinking Tokens Specifics
- **Request**: Convert `{"thinking": {"type": "enabled"}}` → `{"reasoning": {...}, "enable_thinking": true}`
- **Response streaming**: Convert `delta.reasoning_content` or `delta.thinking.content` → `{"type": "thinking", "thinking": "..."}`
- **Signature**: Convert `delta.thinking.signature` → `{"type": "signature_delta", "signature": "..."}`

---

## Files for Next Phase

### Must Modify
- `/Users/alexhubitski/Projects/buster-ripper/buster_ripper/routes/messages.py`

### May Create
- `/Users/alexhubitski/Projects/buster-ripper/buster_ripper/convert/stream.py` (or similar)

### Reference Files (claude-relay)
- `/Users/alexhubitski/Projects/claude-relay/claude_proxy/convert_stream.py`
- `/Users/alexhubitski/Projects/claude-relay/claude_proxy/backend.py`
- `/Users/alexhubitski/Projects/claude-relay/claude_proxy/server.py`
