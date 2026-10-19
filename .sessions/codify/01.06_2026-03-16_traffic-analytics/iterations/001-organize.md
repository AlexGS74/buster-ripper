# Organize Phase Analysis - Traffic Analytics (T009)

## Session Overview
- **Session ID**: 01.06_2026-03-16_traffic-analytics
- **Parent Session**: 01.01_2026-03-16_api-relay-deep-dive
- **Task**: T009: Traffic Capture Enhancement (8h effort estimate)
- **Current State**: Planning phase - Organize

## Synthesis: What Exists vs What's Needed

### Existing Foundation (Verified in Code)

#### 1. `session.py` - SQLite Persistence Layer
**Location**: `/Users/alexhubitski/Projects/buster-ripper/buster_ripper/session.py`

**Current capabilities**:
- `SessionStats` dataclass tracking: `input_tokens`, `avg_latency_ms`, `request_count`
- SQLite schema: `session_stats` table with session_id as primary key
- EMA-based latency smoothing (alpha=0.3)
- Token estimation via `estimate_tokens()` (4 chars ≈ 1 token)
- Session ID derived from system prompt hash (12 chars)

**Current limitations**:
- Only tracks aggregated stats per session
- No per-request capture (timestamp, model, stream flag, stop reason, error status)
- No TTFT (Time to First Token) tracking - only averaged latency
- No SQL views for analytics queries

#### 2. `dump.py` - Session-Aware Prompt Diffing
**Location**: `/Users/alexhubitski/Projects/buster-ripper/buster_ripper/dump.py`

**Current capabilities**:
- Turn-by-turn capture: Turn 0 = full JSON, Turn N+ = unified diff
- `SessionState` tracks: `turn`, `prev_text`, `session_dir`
- File output: `turn_000.json`, `turn_001.diff`, `turn_002.diff`, etc.
- Index file: `index.txt` with summary per turn
- Human-readable rendering of tools, system prompt, messages

**Current limitations**:
- Captures request bodies only (no responses)
- No structured metadata (timestamp, model, tokens, latency per turn)
- Diffs are file-based, not queryable

#### 3. `routes/messages.py` - Request Handler
**Location**: `/Users/alexhubitski/Projects/buster-ripper/buster_ripper/routes/messages.py`

**Current capture points**:
- TTFT captured in streaming: `message_start` event triggers `update_session_stats()`
- End-to-end latency captured in non-streaming mode
- Session ID computed from body, passed to stats update

**Missing capture points**:
- Request/response bodies not persisted to DB
- Model name, stream flag, stop reason not extracted
- Error status not tracked

### What Needs to Be Added (T009 Requirements)

#### Layer 1: Schema Definition
Extend `session.py` with:
- New table: `requests` for per-request capture
- Fields per request:
  - `request_id` (TEXT PRIMARY KEY)
  - `session_id` (TEXT, indexed)
  - `timestamp` (TEXT ISO8601)
  - `model` (TEXT)
  - `stream` (BOOLEAN)
  - `input_tokens` (INTEGER)
  - `output_tokens` (INTEGER)
  - `ttft_ms` (REAL) - Time to First Token
  - `e2e_latency_ms` (REAL) - End-to-end latency
  - `stop_reason` (TEXT)
  - `error_status` (TEXT, nullable)

#### Layer 2: SQL Views for Aggregation
Create analytical views:
- `v_session_summary` - Aggregated stats per session
- `v_request_metrics` - Request-level metrics with session context
- `v_model_performance` - Latency/token stats by model
- `v_hourly_traffic` - Requests per hour, error rates

#### Layer 3: Capture Wiring
Modify handlers to capture:
- `routes/messages.py`: Extract model, stream, tokens, latency, stop_reason
- `routes/chat.py`: Same for chat completions endpoint
- Add response parsing for: `usage.input_tokens`, `usage.output_tokens`, `stop_reason`

#### Layer 4: Analytics Layer (Query Helpers)
Add functions to `session.py`:
- `get_session_stats(session_id)` - Fetch session aggregates
- `get_request_history(session_id, limit)` - Recent requests
- `get_traffic_summary(hours)` - Traffic overview
- `get_error_report(hours)` - Error analysis

## Pattern Analysis

### Data Flow Pattern
``
Request → normalize() → dump_turn() → upstream → capture response → update_session_stats()
                                      ↓
                                      New: capture_request_response() → SQLite
``

### Current Gaps vs Requirements

| Requirement | Current State | Gap |
|-------------|---------------|-----|
| Per-request capture | Session-level aggregates only | New table + capture logic |
| Timestamp tracking | Not captured | Add `datetime.utcnow().isoformat()` |
| Model name | Available in body, not captured | Extract from `body.get("model")` |
| Stream flag | Available in body, not captured | Extract from `body.get("stream")` |
| Token usage (input/output) | `input_tokens` captured, `output_tokens` missing | Parse response usage |
| TTFT | Captured in streaming, averaged only | Store per-request, add view |
| E2E latency | Captured in non-streaming, averaged only | Store per-request |
| Stop reason | Not captured | Parse from response `stop_reason` |
| Error status | Not captured | Add try/except capture |
| SQL views | None | Create 4 analytical views |

## Identified Patterns

### 1. Function-First Architecture (Existing Pattern)
The codebase uses pure functions with module-level state:
- `session.py`: `_stats: dict`, `_db_conn: Optional[Connection]`
- `dump.py`: `_sessions: dict[str, SessionState]`

**Design decision**: Follow this pattern - add functions, not classes.

### 2. Capture Timing
**Streaming path** (`handle_streaming`):
- TTFT captured at `message_start` event
- E2E would need capture at stream end

**Non-streaming path** (`handle_non_streaming`):
- E2E captured after `resp.read()`
- Already implemented in basic form

### 3. Session ID Strategy
Current: SHA256 hash of system prompt (12 chars)
- Stable across turns in same session
- Cannot track "session" in anthropic sense (conversation)

**Recommendation**: Keep current strategy for stats, add request-level UUID for uniqueness.

## Risk Assessment

### Technical Risks
| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| DB schema migration complexity | Low | Medium | Start fresh - this is new table, not migration |
| Performance impact on request path | Medium | Medium | Async writes, batch commits |
| Missing response fields in some backends | Medium | Low | Graceful defaults, nullable fields |
| Streaming response parsing complexity | High | Medium | Test with real SSE streams |

### Known Unknowns
- Response format from vLLM vs SGLang may differ
- SSE stream structure for stop_reason, usage
- Whether `usage.output_tokens` is always present

## Priority Assessment

### Critical Path (Must Complete for T009)
1. **Schema definition** - Foundation for all capture
2. **Request/response capture** - Core functionality
3. **Basic SQL views** - Enable querying

### Secondary (Should Complete)
4. **Error capture** - Production readiness
5. **Analytics helpers** - Usability

### Tertiary (Nice to Have)
6. **Advanced views** - Hourly aggregation, model performance
7. **Export functions** - CSV/JSON export for analysis

## Recommended Implementation Order

### Sequence Rationale
Follow data-flow order: Schema → Capture → Query

```
1. Define schema (session.py)
   ↓
2. Add capture functions (session.py)
   ↓
3. Wire into handlers (routes/messages.py, routes/chat.py)
   ↓
4. Create SQL views (session.py init)
   ↓
5. Add query helpers (session.py)
```

### Estimated Effort Breakdown
| Task | Hours | Notes |
|------|-------|-------|
| Schema design + migration | 1.0h | New table, indexes |
| Capture functions | 2.0h | Request + response capture |
| Handler wiring | 2.0h | messages.py + chat.py |
| SQL views | 1.5h | 4 analytical views |
| Query helpers | 1.0h | get_session_stats, etc. |
| Testing + validation | 0.5h | Verify capture works |
| **Total** | **8.0h** | Matches T009 estimate |

## Blind Spots & Uncertainties

### Identified During Analysis
1. **Response structure variance**: Different backends may return different fields
2. **SSE streaming parsing**: stop_reason may be in different event types
3. **Concurrent write handling**: SQLite with `check_same_thread=False` - need to verify thread safety under load

### Assumptions Documented
| Assumption | Validation Required |
|------------|---------------------|
| vLLM returns `usage.input_tokens` and `usage.output_tokens` | Test with real response |
| SSE `message_start` contains usage data | Verify in streaming handler |
| Non-streaming response has same structure as streaming | Test both paths |
| SQLite concurrent writes are safe with proper connection handling | Document in session.py |

## Success Criteria

### T009 Complete When:
1. [ ] Schema defined: `requests` table exists with all required fields
2. [ ] Capture working: Requests/responses persisted to SQLite
3. [ ] SQL views created: 4 analytical views queryable
4. [ ] Analytics layer: Helper functions return expected data
5. [ ] No regressions: Existing session stats continue working
6. [ ] Verified capture: Can query traffic by session, model, time range

## Questions for Next Phase

No questions asked during organize phase - analysis based on code inspection.

**Deferred for Decide Phase**:
- Should capture be opt-in via CLI flag (`--traffic-capture`)?
- What's the retention policy for captured requests (TTL, max rows)?
- Should analytics views be exposed via HTTP endpoint or DB-only?
