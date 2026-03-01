# buster-ripper

Normalizing HTTP proxy that sits between Claude Code and vLLM, ripping out
the cache busters that Claude Code injects into every request so the KV cache
prefix stays stable across turns (22% → 95%+ cache hit rate).

Also doubles as an eval-mode proxy for [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness),
with model-specific workarounds for proper response extraction.
See [docs/eval.md](docs/eval.md).

---

## Problem

vLLM V1 prefix caching hashes consecutive token blocks. Any token that changes
at position N invalidates the KV cache for all positions > N. For GLM-4.7,
tools are injected at position 0 of the formatted prompt — before the system
prompt — so a single reordered tool definition causes a full cache miss.

Observed cache busters (identified by audit, 2026-02-21):

| Cache buster | Scope | Fixed here |
|---|---|---|
| MCP tool reordering (reconnects return tools in arbitrary order) | Per-turn | ✅ sort by name |
| `currentDate` injection (`Today's date is YYYY-MM-DD.`) | Daily | ✅ `--strip-date` |
| `x-anthropic-billing-header` nonce (`cch=...`) in system prompt | Per-request | ✅ always stripped |
| `gitStatus` block (branch, recent commits) | Per-conversation | ✗ changes once per session, acceptable |
| `<system-reminder>` CLAUDE.md / skills list | Per-turn if content changes | ✗ framework-level |

---

## Install

```bash
uv tool install --editable ~/mllm/buster-ripper
```

This puts `buster-ripper` on your PATH. Because it's editable, any code
changes take effect immediately without reinstalling.

**Requirements:** Python ≥ 3.11, [uv](https://docs.astral.sh/uv/).

---

## Quick start

```bash
# Start vLLM first (port 30000), then:
bash ~/mllm/buster-ripper/serve_buster_ripper.sh

# Point Claude Code at the proxy:
ANTHROPIC_BASE_URL=http://localhost:30001 claude ...
```

Or add to `~/.bashrc`:
```bash
function glm47() {
  ANTHROPIC_BASE_URL="http://localhost:30001" \
  ANTHROPIC_API_KEY="sk-local" \
  ANTHROPIC_MODEL="glm4.7" \
  ANTHROPIC_DEFAULT_OPUS_MODEL="glm4.7" \
  ANTHROPIC_DEFAULT_SONNET_MODEL="glm4.7" \
  ANTHROPIC_DEFAULT_HAIKU_MODEL="glm4.7" \
  CLAUDE_CODE_SUBAGENT_MODEL="glm4.7" \
  /home/alex/.local/bin/claude "$@"
}
```

---

## Serve script options

```bash
# Default: listen 0.0.0.0:30001 → upstream localhost:30000
bash serve_buster_ripper.sh

# Also strip daily date injection (Today's date is YYYY-MM-DD.)
STRIP_DATE=1 bash serve_buster_ripper.sh

# Verbose — log every normalization and per-turn token/latency stats
VERBOSE=1 bash serve_buster_ripper.sh

# Enable session prompt diffing
DUMP_DIR=~/mllm/prompt-diffs bash serve_buster_ripper.sh

# Custom upstream / bind port
UPSTREAM=http://10.0.0.1:30000 PORT=30002 bash serve_buster_ripper.sh
```

Or call directly:
```bash
buster-ripper --help
buster-ripper --upstream http://localhost:30000 --port 30001 --verbose
```

---

## Normalizations applied to `/v1/messages`

1. **Strip billing nonce** (`x-anthropic-billing-header: ... cch=<nonce>`) —
   always on. Claude Code injects this at position 0 of the system prompt on
   every request. It changes every request and busts the entire KV cache
   unconditionally. Stripped silently.

2. **Sort tools by name** — always on. Makes the tool block byte-identical
   across MCP reconnects and different server orderings.

3. **Strip `cache_control`** — always on. Claude Code attaches
   `cache_control: {type: ephemeral}` markers that move forward each turn,
   changing message content hashes even when content is identical.

4. **Strip `currentDate`** — opt-in (`--strip-date`). Removes
   `Today's date is YYYY-MM-DD.` from user message content. Omit if your
   prompts rely on the model knowing the current date.

---

## Compaction nudge (`/v1/messages/count_tokens`)

Claude Code calls `POST /v1/messages/count_tokens` periodically to decide when
to auto-compact the conversation. vLLM returns 404 for this endpoint, so
without interception, compaction never fires and TTFT degrades as context grows.

buster-ripper synthesizes this response from tracked session stats. When both
the token fill ratio **and** average TTFT exceed thresholds (or when server-wide
KV cache utilization is high), it inflates the reported token count to trigger
compaction.

Tune via `--max-model-len`, `--compact-token-ratio`, `--compact-latency-ms`,
`--compact-kv-ratio`.

---

## Dump mode (`--dump-dir`)

Session-aware prompt capture and diffing — useful for identifying remaining
cache busters after normalization.

```bash
DUMP_DIR=~/mllm/prompt-diffs bash serve_buster_ripper.sh
```

**Session ID** = first 12 hex chars of SHA-256(system prompt text). Stable
within a Claude Code session; changes between sessions.

**Output layout** under `DUMP_DIR/<session_id>/`:
```
turn_000.json    # full normalized request body (turn 0)
turn_001.diff    # unified diff: turn 000 → turn 001
turn_002.diff    # unified diff: turn 001 → turn 002
index.txt        # one-line summary per turn
```

**Reading the index:**
```
turn 000  10:31:02  FULL  3 msgs  83 tools  412.3 KB
turn 001  10:32:18  +47/-3 lines  5 msgs  83 tools
turn 002  10:33:05  +892/-0 lines  7 msgs  83 tools   ← large file read
```

A non-zero `-` count on the tools section = tool list changed = cache busted
from position 0.

---

## Eval mode

See **[docs/eval.md](docs/eval.md)** for full details on using buster-ripper
as an lm-evaluation-harness proxy, model profiles, and GLM-4.7 workarounds.

---

## Project layout

```
buster_ripper/
├── config.py          globals set by CLI at startup; all modules read from here
├── normalize.py       /v1/messages request normalization
├── session.py         per-session token/latency stats + SQLite persistence
├── compaction.py      KV cache poller + compaction nudge logic
├── dump.py            session-aware prompt diffing
├── utils.py           shared HTTP helpers
├── app.py             FastAPI app + lifespan
├── main.py            CLI entrypoint (click)
└── routes/
    ├── messages.py    POST /v1/messages
    ├── chat.py        POST /v1/chat/completions  (eval-mode + model profiles)
    ├── count_tokens.py POST /v1/messages/count_tokens
    └── passthrough.py  catch-all transparent proxy
tests/
docs/
    eval.md            eval-mode guide
```

---

## Architecture

```
Claude Code  →  buster-ripper (:30001)  →  vLLM (:30000)
                      │
              /v1/messages:
                      ├─ strip billing nonce
                      ├─ strip cache_control
                      ├─ sort tools[] by name
                      ├─ drop empty tools arrays
                      └─ (optional) strip currentDate
              /v1/messages/count_tokens:
                      └─ synthesize response; nudge if thresholds exceeded
              /v1/chat/completions:
                      ├─ eval-mode: strip max_gen_toks, strip <think> blocks (OAI format)
                      ├─ eval-mode: apply model profile strategies
                      └─ non-eval: split <think> → reasoning_content (CC format)
              everything else:
                      └─ transparent passthrough
```

---

## Verify it's working

Run with `VERBOSE=1` and watch for normalization log lines:

```
10:32:01 [buster-ripper] tools reordered: [Bash, Edit, Glob, ...] → [Alpha, Beta, ...]
10:32:01 [buster-ripper] stripped billing-header from system block
10:32:01 [buster-ripper] stripped cache_control from 3 message field(s)
```

If tools are already in stable order and there are no cache-busting fields,
nothing is logged.
