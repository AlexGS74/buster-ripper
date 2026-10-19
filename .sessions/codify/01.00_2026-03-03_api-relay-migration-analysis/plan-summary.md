# Session Plan: API Relay Migration Analysis

## Progress Overview

| Phase | Status | Notes |
|-------|--------|-------|
| Collect | ✅ Complete | Comprehensive codebase analysis |
| Organize | ✅ Complete | Synthesis and pattern identification |
| Decide | ✅ Complete | Implementation recommendations finalized |
| Invite | ⏳ Pending | Ready for user presentation |
| Forward | ⏳ Pending | |
| Yield | ⏳ Pending | |

## Session Metadata
- **Session ID**: 01.00_2026-03-03_api-relay-migration-analysis
- **Topic**: API Relay Migration Analysis
- **Created**: 2026-03-03
- **Parent Session**: None (new parent)

## Scope
- Analyze thinking tokens handling in buster-ripper and claude-relay
- Document current implementation state
- Identify gaps vs reference implementation
- Map integration points and test coverage

## Key Investigation Areas
1. Thinking/reasoning tokens handling
2. Streaming conversion (Anthropic ↔ OpenAI)
3. Image handling (image_agent pattern)
4. Request normalization
5. Test coverage gaps

## Collect Findings Summary
- **Critical Gap**: `routes/messages.py` `handle_streaming()` performs raw passthrough without OpenAI→Anthropic SSE conversion
- **Thinking Tokens**: Request conversion exists in `convert/anthropic_to_openai.py`; streaming response conversion missing
- **Reference**: claude-relay `convert_stream.py` provides complete streaming conversion pattern
- **Test Gaps**: No tests for convert module or route handlers

## Decide Summary

**Recommended Approach**: Direct mirror of claude-relay's `convert_stream.py` pattern (Option 1)

**Implementation Sequence**:
1. **P0**: Create `buster_ripper/convert/stream.py` with `convert_openai_stream_to_anthropic()`
2. **P0**: Modify `handle_streaming()` in `routes/messages.py` to wire up converter
3. **P1**: Add backend detection (sglang vs vLLM) for `chat_template_kwargs` injection
4. **P2**: Unit tests for conversion logic
5. **P3**: Integration tests for end-to-end streaming

**Key Decisions**:
- Backend detection deferred to P1 (not blocking for thinking tokens)
- Full conversion required (not thinking-only) to avoid broken tool calls
- State tracking: `thinking_started`, `text_started`, `tool_call_blocks` dict

**See**: `iterations/003-decide.md` for comprehensive decision rationale

## Finalized Implementation Plan

| Priority | Task | File(s) | Effort | Status |
|----------|------|---------|--------|--------|
| P0 | Create `convert/stream.py` with SSE conversion logic | `buster_ripper/convert/stream.py` | 2-3 hours | Pending |
| P0 | Wire up converter in `handle_streaming()` | `buster_ripper/routes/messages.py` | 1 hour | Pending |
| P1 | Add backend detection for sglang/vLLM | `buster_ripper/convert/anthropic_to_openai.py` or `backend.py` | 2 hours | Pending |
| P2 | Unit tests for stream conversion | `tests/test_convert_stream.py` | 2-3 hours | Pending |
| P3 | Integration tests for streaming | `tests/test_messages_streaming.py` | 2 hours | Pending |

**Total Estimated Effort**: 7-10 hours (1-2 days)
