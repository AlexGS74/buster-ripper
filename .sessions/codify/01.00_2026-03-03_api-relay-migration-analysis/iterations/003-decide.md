# Decide Phase: Final Recommendations

## Decision Summary

### Problem Statement (Restated)

The streaming response path in `/v1/messages` performs **raw passthrough** of OpenAI SSE format instead of converting to Anthropic format. This affects:
- Thinking/reasoning tokens (not converted to `thinking` blocks)
- Tool calls (not converted to `tool_use` blocks)
- Text content (wrong SSE event structure)

### Core Recommendation: Mirror Claude-Relay Pattern

**Approach**: Direct mirror of claude-relay's `convert_stream.py` pattern

**Rationale**:
1. Reference implementation already handles all edge cases
2. Proven in production (claude-relay)
3. Minimal adaptation needed for buster-ripper context

---

## Detailed Recommendations

### 1. Streaming SSE Conversion (P0)

**Recommended approach**: Full `convert_openai_stream_to_anthropic()` implementation

**Function signature** (matching claude-relay):
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

**State tracking approach** (matching claude-relay):
| State Variable | Purpose | Initial Value |
|----------------|---------|---------------|
| `thinking_started: bool` | Track if thinking block begun | `False` |
| `text_started: bool` | Track if text block begun | `False` |
| `tool_call_blocks: dict[int, int]` | OpenAI tool idx → Anthropic block idx | `{}` |
| `current_index: int` | Currently open block index | `-1` |
| `block_index: int` | Next block index to allocate | `0` |

**Integration into `handle_streaming()`**:
```python
# Current (raw passthrough - WRONG)
async with session.post(...) as resp:
    async for chunk in resp.content.iter_any():
        await response.write(chunk)

# After (with conversion - CORRECT)
sse_events = parse_sse_stream(resp)
async for chunk in convert_openai_stream_to_anthropic(
    sse_events, req_id=req_id, input_tokens=input_token_count
):
    await response.write(chunk)
```

**Why full implementation vs simplified?**
- Partial conversion creates broken UX (thinking works, tool calls break)
- No incremental benefit - claude-relay already solved this completely
- Full solution is ~200 lines; simplified would be ~100 lines (50% savings is not worth the debt)

---

### 2. Backend Detection (P1)

**Is backend detection required NOW for thinking tokens?**

**Answer: No** - thinking tokens work with standard OpenAI SSE format regardless of backend

**When backend detection becomes P0**:
- When sglang compatibility is needed
- When model remapping is required
- When backend-specific `chat_template_kwargs` injection is needed

**Recommended deferral**:
```python
# Phase 1 (P0 - vLLM baseline):
# - Send standard OpenAI SSE request
# - Backend sends standard OpenAI SSE response
# - Converter handles standard format

# Phase 2 (P1 - sglang support):
# - Add backend detection
# - Inject chat_template_kwargs for sglang
# - Handle backend-specific SSE variations
```

---

### 3. Test Strategy

**Recommended approach**: Unit tests first, then integration

**P2 - Unit tests** (`tests/test_convert_stream.py`):
```python
# Test cases (following pytest pattern from existing tests)

@mock.patch('buster_ripper.convert.stream.make_anthropic_sse')
@pytest.mark.asyncio
async def test_thinking_token_conversion(mock_sse, mocker):
    """Delta reasoning_content converted to thinking blocks."""
    # Mock SSE events with reasoning_content
    # Verify: block_start, block_delta (thinking), signature_delta

@mock.patch('buster_ripper.convert.stream.make_anthropic_sse')
@pytest.mark.asyncio
async def test_thinking_to_text_transition(mock_sse, mocker):
    """Thinking block closes when text arrives."""
    # Mock: thinking delta → text delta
    # Verify: thinking block_stop, text block_start

@mock.patch('buster_ripper.convert.stream.make_anthropic_sse')
@pytest.mark.asyncio
async def test_tool_call_conversion(mock_sse, mocker):
    """OpenAI tool_calls → Anthropic tool_use blocks."""
    # Mock: tool_calls delta
    # Verify: tool_use block_start, input_json_delta
```

**P3 - Integration tests** (`tests/test_messages_streaming.py`):
```python
# Full end-to-end with mocked aiohttp

@pytest.mark.asyncio
async def test_streaming_end_to_end(aiohttp_client, app, mocker):
    """Full streaming flow from client to upstream."""
    # Mock upstream response
    mock_upstream = mocker.Mock()
    mock_upstream.content.iter_any = AsyncMock(side_effect=mock_sse_chunks)

    # POST to /v1/messages
    # Verify: correct Anthropic SSE output
```

**Mocking SSE streams** (pattern from claude-relay):
```python
async def mock_sse_chunks():
    for chunk in chunks:
        yield chunk.encode()
```

---

### 4. Implementation Order (Minimum Viable)

| Phase | Task | Files | Priority | Effort |
|-------|------|-------|----------|--------|
| 1 | Create `convert/stream.py` | New file | P0 | 4-6h |
| 2 | Wire up in `messages.py` | Existing | P0 | 1-2h |
| 3 | Unit tests for stream.py | New tests | P2 | 2-3h |
| 4 | Backend detection | New or existing | P1 | 3-4h |
| 5 | Integration tests | New tests | P3 | 2-3h |

**Why this order?**
1. Core conversion must work before anything else
2. Wiring up is trivial once converter exists
3. Unit tests validate conversion logic
4. Backend detection is valuable but not blocking
5. Integration tests confirm end-to-end

**Total effort to thinking tokens working**: 5-8 hours (P0+P1)
**Total effort to full test coverage**: 12-15 hours (all phases)

---

## Trade-off Analysis

| Criterion | Full Mirror | Simplified | Backend First |
|-----------|-------------|------------|---------------|
| Time to thinking tokens | 5-8h | 2-4h | 8-12h |
| Time to full solution | 12-15h | 8-10h* | 12-15h |
| Risk of rework | Low | High | Medium |
| Tool call support | Yes | No | Yes |
| Backend compatibility | vLLM baseline | vLLM only | All backends |

* Simplified requires catch-up work to reach full solution

---

## Explicit Decisions for Invite

The Invite agent should present these decisions to the user:

### Decision 1: Streaming Conversion Approach
**Choice**: Full mirror of claude-relay pattern vs. simplified thinking-only
- **Recommendation**: Full mirror
- **Reasoning**: Partial solution creates broken UX; full solution is proven

### Decision 2: Backend Detection Timing
**Choice**: Defer to P1 vs. implement now
- **Recommendation**: Defer to P1 (after core conversion works)
- **Reasoning**: vLLM baseline works without detection; sglang is special case

### Decision 3: Test Priority
**Choice**: Unit tests first vs. integration tests first
- **Recommendation**: Unit tests first (P2), then integration (P3)
- **Reasoning**: Validate conversion logic before end-to-end

### Decision 4: Implementation Order
**Choice**: Follow P0 → P0 → P2 → P1 → P3 sequence
- **Recommendation**: Sequence as specified
- **Reasoning**: Minimum viable sequence to thinking tokens working

---

## Success Criteria (Measurable)

| Criterion | Target | Measurement |
|-----------|--------|-------------|
| Thinking tokens | 100% conversion | `delta.reasoning_content` → `thinking` blocks |
| Tool calls | 100% conversion | OpenAI `tool_calls` → Anthropic `tool_use` |
| Text content | No regression | Passed through with correct block structure |
| Signature handling | Present | `signature_delta` events sent |
| Usage reporting | Accurate | `input_tokens`, `output_tokens` in `message_delta` |
| Test coverage | >80% | Coverage report for `convert/stream.py` |
| Pass rate | 100% | All tests pass |

---

## Risks and Mitigation

| Risk | Likelihood | Impact | Mitigation Strategy |
|------|------------|--------|---------------------|
| Non-standard backend SSE | Low | High | Test against actual backend early; add adapters if needed |
| Thinking token format mismatch | Medium | Medium | Verify against Anthropic API spec; compare with claude-relay |
| Tool call conversion breaks | Medium | High | Test with single tool first; expand to multiple |
| State tracking bugs | Medium | Medium | Log state transitions; comprehensive unit tests |
| Performance regression | Low | Medium | Profile streaming path; minimize buffering |

---

## Deferred Questions

| Question | Status | Impact | Recommendation |
|----------|--------|--------|----------------|
| Backend type (vLLM vs sglang) | Deferred | Determines if backend detection is P0 or P1 | Assume vLLM unless specified otherwise |
| Custom SSE format requirements | Deferred | May need adapter layer | Start with standard OpenAI SSE |
| Additional logging/telemetry | Deferred | Operational visibility | Add minimal logging; expand based on need |
