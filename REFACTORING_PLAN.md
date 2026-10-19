# API Relay Migration - Refactoring Plan

## Status: aiohttp migration complete — the API-conversion package was never built

**Last updated:** Phase 1 as shipped = the FastAPI/httpx → aiohttp migration and
image handling. The Anthropic→OpenAI conversion layer this plan originally scoped
(`buster_ripper/convert/`) is **not implemented**; it is listed under
§Remaining Work as future scope, not as completed work.

The migration from FastAPI/httpx to aiohttp has been completed. The codebase now
has the foundation for API relay capabilities including image handling.

## What Was Done (Phase 1)

### Migration to aiohttp
- Replaced FastAPI with aiohttp application framework
- Replaced httpx with aiohttp.ClientSession for upstream requests
- All route handlers migrated to aiohttp handlers

### New Modules Created

#### `buster_ripper/image.py`
Pure functions for image handling:
- `has_images(body)` - Detect images in user messages
- `strip_and_cache_images(body, session_id)` - Strip and cache images with TTL
- `get_cached_image(session_id, image_key)` - Retrieve cached images
- `make_data_url(data, media_type)` - Base64 to data URL conversion
- Module-level cache with configurable TTL and max size

### Configuration Added
- `STRIP_IMAGES` - Enable image stripping
- `IMAGE_CACHE_MAX_SIZE` - Max images per session (default: 100)
- `IMAGE_CACHE_TTL` - Image cache TTL in seconds (default: 300)

### CLI Options Added
- `--strip-images` - Strip images from messages
- `--image-cache-max-size` - Configure cache size
- `--image-cache-ttl` - Configure cache TTL

## Remaining Work

**Nothing in this section is implemented**, including the `buster_ripper/convert/`
package that Phases 2-4 and Phase 6 name: no `convert/` directory, no
`anthropic_to_openai.py`, no `sse.py` exists in `buster_ripper/`. These rows are
future scope, not a description of the shipped tree.

### Phase 2: Streaming Support for API Relay — not implemented
The `convert_stream.py` module from Claude Relay needs to be adapted for full streaming conversion:
- [ ] Add `convert_stream_chunk()` function to `convert/anthropic_to_openai.py`
- [ ] Implement OpenAI SSE → Anthropic SSE streaming conversion
- [ ] Wire streaming conversion into `routes/messages.py` handler

### Phase 3: Bidirectional Conversion — not implemented
No conversion exists in either direction today. Need:
- [ ] Create `convert/openai_to_anthropic.py` for reverse conversion
- [ ] Handle OpenAI chat/completions → Anthropic messages format
- [ ] Handle OpenAI SSE → Anthropic SSE streaming

### Phase 4: API Relay Integration
- [ ] Add relay configuration to `config.py`
- [ ] Add `--relay-mode` CLI option to enable conversion
- [ ] Wire conversion into message handlers conditionally
- [ ] Add relay mode logging and metrics

### Phase 5: Image Processing Features
- [ ] Implement actual image processing (resize, compress)
- [ ] Add image format normalization
- [ ] Support for URL-based images (fetch and cache)
- [ ] Integration with memory/knowledge extraction

### Phase 6: Testing
- [x] Unit tests for image module (`image.py`) — shipped as `tests/test_image.py`
- [ ] Unit tests for convert module (`convert/anthropic_to_openai.py`) — blocked on Phase 2
- [ ] Integration tests for API relay
- [ ] End-to-end streaming tests

## Architecture Notes

### Function-First Design
All modules use pure functions with module-level state (like `session.py` pattern):
- No classes except dataclasses for DTOs
- Clear data flow through functions
- Easy to test and reason about

### CODIFY Methodology
This project uses the CODIFY methodology for structured development:
- **Planning Phase**: collect → organize → decide → invite
- **Implementation Phase**: forward → yield (iterative cycles)
- **Session tracking**: `.sessions/codify/{session-id}/`
- **QA Planning**: Test cases defined in `qa-plan.md` before implementation
- **Code quality**: L→N→C→P metrics (Length ≤15, Nesting ≤1, Complexity ≤5, Parameters ≤3)

### aiohttp Patterns
- Use `web.Request` for request access
- Use `web.Response` for responses
- Use `web.StreamResponse` for streaming
- Middleware for logging/cross-cutting concerns
- Background tasks via `on_startup` hooks

### Configuration
All config is in `config.py` as module-level variables:
- Mutable, set by `main()` at startup
- Other modules import and read `config.<NAME>`
- Allows CLI to control behavior

## File Structure

```
buster_ripper/
├── config.py          # Global configuration
├── normalize.py       # Request normalization (with image detection)
├── session.py         # Session stats + SQLite persistence
├── backend.py         # sglang/vllm detection
├── compaction.py      # KV cache poller + compaction nudge
├── dump.py            # Session-aware prompt diffing
├── utils.py           # Shared HTTP helpers
├── image.py           # Image handling
├── server.py          # aiohttp application (replaces app.py)
├── main.py            # CLI entrypoint
└── routes/
    ├── messages.py    # /v1/messages handler
    ├── chat.py        # /v1/chat/completions handler
    ├── count_tokens.py
    └── passthrough.py
```

There is no `convert/` package; see §Remaining Work.

## Testing the Implementation

```bash
# Install dependencies
python3 -m venv .venv
source .venv/bin/activate
pip install -e .

# Run server
buster-ripper --upstream http://localhost:30000 --strip-images --verbose

# Test image detection
python3 -c "from buster_ripper.image import has_images; print(has_images({'messages': []}))"
```

## Related Files
- Plan originated from: `/Users/alexhubitski/.claude/projects/-Users-alexhubitski-Projects/4cca2541-ae0c-4bf0-aef3-7491497dab07.jsonl`
- Reference implementation: `~/Projects/claude-relay/claude_proxy/`

## Source Project Reference

The following files from `claude-relay` were used as reference for this migration:

| Source File | Buster Ripper Equivalent | Notes |
|-------------|-------------------------|-------|
| `~/Projects/claude-relay/claude_proxy/server.py` | `buster_ripper/server.py` | aiohttp app factory |
| `~/Projects/claude-relay/claude_proxy/image_agent.py` | `buster_ripper/image.py` | Function-based (no classes) |
| `~/Projects/claude-relay/claude_proxy/convert_request.py` | — none yet | `buster_ripper/convert/anthropic_to_openai.py` is Phase 2/3 scope, not built |
| `~/Projects/claude-relay/claude_proxy/convert_stream.py` | _pending_ | Phase 2: streaming |
| `~/Projects/claude-relay/claude_proxy/normalize.py` | `buster_ripper/normalize.py` | Extended with image detection |
| `~/Projects/claude-relay/claude_proxy/sse.py` | — none yet | `buster_ripper/convert/sse.py` is Phase 2 scope, not built; the shipped SSE helper is `utils.find_message_start` |

**Key architectural differences:**
- Claude Relay uses `ImageCache` class → Buster Ripper uses module-level state with functions
- Claude Relay has monolithic `server.py` → Buster Ripper uses `routes/` package
- Buster Ripper adds CODIFY methodology for structured development
