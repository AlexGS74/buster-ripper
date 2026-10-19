# Decide Phase: API Relay Migration Implementation Plan

## Decision Summary

### Problem Statement

Based on the Organize phase analysis, the critical gap is:

**`buster_ripper/routes/messages.py` `handle_streaming()` performs raw passthrough of upstream SSE events without converting from OpenAI format to Anthropic format.** This means thinking tokens, tool calls, and text content are sent in the wrong format to clients expecting Anthropic SSE events.

### Assumptions Made

1. **Backend is vLLM-based**: Current deployment uses vLLM or compatible backend (not sglang-specific features needed yet)
2. **Upstream sends OpenAI SSE format**: Backend sends standard OpenAI-compatible SSE with `reasoning_content` or `thinking` fields
3. **Claude-Relay pattern is valid reference**: The reference implementation in claude-relay correctly implements the Anthropic SSE format
4. **No breaking changes to existing API**: Existing clients continue to work; only streaming conversion is added

### Decisions Dependent on Assumptions

| Assumption | If Wrong, Then... |
|------------|-------------------|
| Backend is vLLM | Backend detection (P1) becomes P0; sglang injection needed immediately |
| Upstream uses standard OpenAI SSE | Custom conversion logic needed for non-standard backends |
| Claude-Relay pattern is correct | Must verify against Anthropic API spec; may need adjustments |
| No breaking changes | Must add migration path if conversion introduces incompatibilities |

---

## Options Analysis

### Option 1: Direct Mirror of claude-relay Pattern (Recommended)

**Description**: Replicate claude-relay's `convert_stream.py` structure exactly in `buster_ripper/convert/stream.py`.

**Implementation**:
```
buster_ripper/
├── convert/
│   ├── stream.py         # NEW: OpenAI SSE → Anthropic SSE conversion
│   ├── anthropic_to_openai.py
│   └── sse.py
├── routes/
│   ├── messages.py       # MOD: Wire up convert_stream in handle_streaming()
│   └── chat.py
```

**Function signature for integration**:
```python
async def convert_openai_stream_to_anthropic(
    sse_events: AsyncIterator[SSEEvent],
    index_offset: int = 0,
    skip_message_wrapper: bool = False,
    result: StreamResult | None = None,
    req_id: str = "",
    input_tokens: int = 0,
) -> AsyncIterator[bytes]:
```

**State tracking approach**:
- `thinking_started: bool` - Track if thinking block has begun
- `text_started: bool` - Track if text block has begun
- `tool_call_blocks: dict[int, int]` - Map OpenAI tool index → Anthropic block index
- `current_index: int` - Currently open block index (-1 = none)

**Pros**:
- Proven pattern from reference implementation
- Handles all edge cases (thinking → text, thinking → tool_use, multiple tool calls)
- Clear separation of concerns (conversion vs. routing)
- Minimal changes to existing `messages.py`

**Cons**:
- Requires careful state management across async iteration
- May need adaptation for non-standard backends

**Complexity**: Medium (150-200 lines of conversion logic)

---

### Option 2: Simplified Conversion (Thinking-Only)

**Description**: Start with minimal thinking token support, defer tool call and text conversion.

**Implementation**:
```python
async def convert_openai_stream_to_anthropic(...):
    async for event in sse_events:
        # Only handle reasoning_content
        if delta.get("reasoning_content"):
            yield thinking_delta(delta["reasoning_content"])
        # Passthrough everything else
        yield original_event
```

**Pros**:
- Faster time-to-market for thinking tokens
- Lower risk (less code to maintain)
- Can be extended incrementally

**Cons**:
- Incomplete solution (tool calls break)
- Technical debt if not completed
- May require rework if initial implementation is wrong

**Complexity**: Low-Medium (50-80 lines initially)

---

### Option 3: Backend Detection First

**Description**: Implement `backend.py` detection before streaming conversion to ensure correct format handling.

**When this matters**:
- Backend is sglang (not vLLM): Requires `chat_template_kwargs` injection
- Model remapping needed: Backend model name differs from client request
- Usage tracking differs: Backends report tokens differently

**Pros**:
- Prevents wasted effort if backend requires different approach
- Enables sglang-specific optimizations
- Clean abstraction for backend differences

**Cons**:
- Adds complexity before core conversion works
- May not be needed if backend is vLLM-compatible
- Delays thinking token support

**Complexity**: Low (50-80 lines, similar to claude-relay)

---

## Trade-off Analysis

| Criterion | Option 1 (Full Mirror) | Option 2 (Simplified) | Option 3 (Backend First) |
|-----------|------------------------|------------------------|---------------------------|
| **Time to thinking tokens** | 2-3 days | 1 day | 2-3 days |
| **Completeness** | Full solution | Partial (thinking only) | Depends on backend |
| **Risk** | Low (proven pattern) | Medium (technical debt) | Low (defers risk) |
| **Maintainability** | High (clean separation) | Medium (incomplete) | High (clean abstraction) |
| **Scalability** | Supports all backends | Limited to vLLM | Supports all backends |

---

## Recommendations

### Recommended Approach: Option 1 (Direct Mirror)

**Justification**:
1. **Claude-Relay pattern is proven**: The reference implementation already handles all edge cases
2. **Thinking tokens require full conversion**: Partial implementation breaks tool calls
3. **Backend detection can be deferred**: vLLM is the common denominator; sglang support can be added later

**Implementation sequence**:

| Order | Task | Files | Priority |
|-------|------|-------|----------|
| 1 | Create `convert/stream.py` | New file | P0 |
| 2 | Modify `handle_streaming()` in `messages.py` | Existing | P0 |
| 3 | Add unit tests for `convert_stream.py` | New test file | P2 |
| 4 | Backend detection in `convert_request.py` | Existing or new | P1 |

**Why not Option 2?**
- Partial conversion creates broken UX (thinking works, tool calls don't)
- Claude-Relay already solved this; no benefit in reinventing half-solution

**Why not Option 3 first?**
- Backend detection is valuable but not blocking for thinking tokens
- Can be added as P1 after core conversion works
- vLLM compatibility is the baseline; sglang is the special case

---

## Test Strategy

### Unit Tests (P2)
**What to mock**:
- SSE event stream (use `AsyncIterator` mock)
- `make_anthropic_sse()` calls
- State tracking (verify `thinking_started` transitions)

**Test cases**:
```python
@pytest.mark.asyncio
async def test_thinking_token_conversion():
    """Thinking tokens in delta.reasoning_content are converted to thinking blocks."""
    # Mock SSE events with reasoning_content
    # Verify: thinking block_start, thinking_delta, signature_delta

@pytest.mark.asyncio
async def test_tool_call_conversion():
    """OpenAI tool_calls are converted to Anthropic tool_use blocks."""
    # Mock SSE events with tool_calls
    # Verify: tool_use block_start, input_json_delta, correct block mapping

@pytest.mark.asyncio
async def test_thinking_to_text_transition():
    """Thinking block closes when text content arrives."""
    # Mock: thinking delta, then text delta
    # Verify: thinking block_stop, then text block_start
```

### Integration Tests (P3)
**What to test**:
- End-to-end streaming from mock backend
- Correct SSE event sequence
- Usage reporting accuracy

**Mock strategy**:
```python
@pytest.mark.asyncio
async def test_streaming_integration():
    """Full streaming flow with mock backend."""
    # Create mock aiohttp response that yields SSE events
    mock_response = mocker.Mock()
    mock_response.content.iter_any = AsyncMock(
        return_value=AsyncMock(__aiter__=AsyncMock(side_effect=mock_sse_events))
    )
    # Verify: correct Anthropic SSE output
```

---

## Implementation Order (Minimum Viable Sequence)

### Phase 1: Core Conversion (P0)
1. Create `buster_ripper/convert/stream.py`
   - `convert_openai_stream_to_anthropic()` function
   - State tracking (thinking_started, text_started, tool_call_blocks)
   - SSE event builders (message_start, content_block_start/delta/stop)
2. Modify `buster_ripper/routes/messages.py`
   - Import `convert_openai_stream_to_anthropic`
   - Wire up in `handle_streaming()`
   - Parse SSE events from upstream, pass through converter

### Phase 2: Backend Detection (P1)
3. Add backend detection to `buster_ripper/convert/anthropic_to_openai.py`
   - Or create new `backend.py` if preferred
   - Probe `/v1/models` endpoint
   - Inject `chat_template_kwargs` for sglang

### Phase 3: Test Coverage (P2-P3)
4. Unit tests for `convert/stream.py`
5. Integration tests for `messages.py` streaming

---

## Success Criteria

| Criterion | Target |
|-----------|--------|
| **Thinking tokens**: Converted from `delta.reasoning_content` to `{"type": "thinking", "thinking": "..."}` | 100% accuracy |
| **Tool calls**: Converted from OpenAI format to Anthropic `tool_use` blocks | 100% accuracy |
| **Text content**: Passed through unchanged (with correct block structure) | No regression |
| **Signature handling**: `delta.thinking.signature` → `signature_delta` event | Present |
| **Usage reporting**: Input/output tokens reported in `message_delta` | Accurate |

---

## Risks and Mitigation

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Non-standard backend SSE format | Low | High | Test against actual backend early; add backend-specific adapters |
| Thinking token format mismatch | Medium | Medium | Verify against Anthropic API spec; compare with claude-relay output |
| Tool call conversion breaks | Medium | High | Test with multiple tool call scenarios; start with single tool |
| Performance regression (latency) | Low | Medium | Profile streaming path; minimize buffering |
| State tracking bugs (double-close, missed blocks) | Medium | Medium | Comprehensive unit tests; log state transitions |

---

## Time Estimates

| Priority | Task | Effort |
|----------|------|--------|
| P0 | Create `convert/stream.py` | 2-3 hours |
| P0 | Wire up in `handle_streaming()` | 1 hour |
| P1 | Backend detection | 2 hours |
| P2 | Unit tests | 2-3 hours |
| P3 | Integration tests | 2 hours |
| **Total** | | **7-10 hours** |

---

## Files for Invite Agent

The Invite agent should present these decisions to the user:

1. **Streaming conversion approach**: Full mirror of claude-relay pattern (Option 1)
2. **Backend detection timing**: Deferred to P1 (after core conversion works)
3. **Test strategy**: Unit tests for conversion logic, integration tests for end-to-end
4. **Implementation order**: `stream.py` → `messages.py` → backend detection → tests

**Questions for user**:
1. Is the proposed implementation order acceptable, or should backend detection be prioritized?
2. Are there any backend-specific requirements not covered (e.g., sglang, custom tokenizers)?
3. Should the implementation include any additional logging/telemetry beyond claude-relay?
4. Do you approve proceeding to Forward phase with P0 (streaming conversion)?
