# Session Context

## Session Metadata
- **Session ID**: 01.00_2026-03-03_api-relay-migration-analysis
- **Topic**: API Relay Migration Analysis
- **Question Mode**: deferred
- **Created**: 2026-03-03
- **Last Updated**: 2026-03-03

## Current Status
- **Current Phase**: decide → invite (ready for user presentation)
- **Progress**: Decision analysis complete with implementation recommendations

## User Choices Resolved
- None - decision phase completed with deferred questions for user review

## Decisions Made

**Recommended Approach**: Direct mirror of claude-relay's `convert_stream.py` pattern (Option 1)

**Implementation Sequence**:
1. **P0**: Create `buster_ripper/convert/stream.py` with `convert_openai_stream_to_anthropic()`
2. **P0**: Modify `handle_streaming()` in `routes/messages.py` to wire up converter
3. **P1**: Add backend detection (sglang vs vLLM) for `chat_template_kwargs` injection
4. **P2**: Unit tests for conversion logic
5. **P3**: Integration tests for end-to-end streaming

**Key Assumptions**:
- Backend is vLLM-based (not sglang-specific)
- Upstream sends standard OpenAI SSE format
- Claude-Relay pattern correctly implements Anthropic SSE format
- No breaking changes to existing API

**State Tracking Approach**:
- `thinking_started: bool` - Track if thinking block has begun
- `text_started: bool` - Track if text block has begun
- `tool_call_blocks: dict[int, int]` - Map OpenAI tool index to Anthropic block index
- `current_index: int` - Currently open block index (-1 = none)

## Findings Summary
- **Critical Gap**: `messages.py` `handle_streaming()` performs raw passthrough without OpenAI→Anthropic SSE conversion
- **Thinking Tokens**: Request conversion exists; response streaming conversion missing
- **Reference Implementation**: claude-relay `convert_stream.py` provides complete pattern
- **Test Coverage**: No tests for convert module or routes
- **vLLM Analysis Complete**: vLLM's Anthropic endpoint only uses `/v1/messages` and `/v1/chat/completions` - confirms our scope is correct, can ignore legacy completions API and OpenAI-only features

## Key Files
- See `/iterations/001-collect.md` for full analysis
- See `/iterations/002-decide.md` for comprehensive decision rationale
