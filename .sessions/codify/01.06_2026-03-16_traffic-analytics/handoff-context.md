# Session Context

## Session Metadata
- **Session ID**: 01.06_2026-03-16_traffic-analytics
- **Topic**: Traffic Analytics - Enhanced Traffic Capture and Analysis
- **Question Mode**: deferred (default per skill defaults)
- **Created**: 2026-03-16
- **Last Updated**: 2026-03-16 (Organize phase complete)

## Current Status
- **Current Phase**: Organize complete → Ready for Decide
- **Progress**: Analyzed existing codebase, identified gaps, defined implementation plan

## Analysis Summary

### What Exists (Verified)
1. **`session.py`**: SQLite persistence with `SessionStats` dataclass, session-level aggregates
2. **`dump.py`**: Turn-by-turn prompt diffing (Turn 0 = JSON, Turn N+ = diff)
3. **`routes/messages.py`**: TTFT capture in streaming, latency capture in non-streaming

### What's Needed (T009 Gaps)
1. **Schema definition**: New `requests` table for per-request capture
2. **SQL views**: 4 analytical views for querying
3. **Capture wiring**: Extract model, tokens, latency, stop_reason from handlers
4. **Analytics layer**: Helper functions for traffic analysis

### Implementation Strategy
Follow data-flow order:
1. Schema (session.py) → 2. Capture functions → 3. Handler wiring → 4. Views → 5. Helpers

## User Choices Resolved
- **Parent Session Decision**: All P0-P3 tasks split into 5 independent sessions
  - Impact: This session (T009) is independent, can execute in parallel
  - Status: RESPECTED

## Design Decisions (Organize Phase)

### Decision 1: Function-First Architecture
**Choice**: Follow existing pattern - pure functions with module-level state
**Rationale**: Consistent with `session.py` and `dump.py` patterns
**Impact**: No new classes, additive functions only

### Decision 2: Request-Level Capture Strategy
**Choice**: Add `request_id` (UUID) for uniqueness, keep session_id from system prompt hash
**Rationale**: Current session strategy is stable but not anthropic-conversation-aware
**Impact**: Request-level granularity with session grouping

### Decision 3: Capture Points
**Choice**: Capture in handlers after upstream response
**Rationale**: Non-invasive, follows existing `update_session_stats` pattern
**Impact**: Minimal disruption to existing flow

## Assumptions Documented
| Assumption | Validation Required |
|------------|---------------------|
| vLLM returns `usage.input_tokens` and `usage.output_tokens` | Test with real response |
| SSE `message_start` contains usage data | Verify in streaming handler |
| SQLite concurrent writes safe with `check_same_thread=False` | Document if issues arise |

## Open Questions for Decide Phase
1. **Opt-in capture**: Should `--traffic-capture` CLI flag be added?
2. **Retention policy**: TTL or max rows for captured requests?
3. **Analytics exposure**: HTTP endpoint or DB-only queries?

## Success Criteria (from Organize Analysis)
- [ ] Schema defined: `requests` table with all required fields
- [ ] Capture working: Requests/responses persisted to SQLite
- [ ] SQL views created: 4 analytical views queryable
- [ ] Analytics layer: Helper functions return expected data
- [ ] No regressions: Existing session stats continue working
- [ ] Verified capture: Can query traffic by session, model, time range

## Notes
- See `iterations/001-organize.md` for detailed pattern analysis
- See `plan-summary.md` for updated implementation plan
- Files to modify: `session.py`, `routes/messages.py`, `routes/chat.py`
- Next phase: Decide (approach selection, question resolution)
