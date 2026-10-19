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

# Skip upstream TLS certificate verification — for self-signed certificates.
# Verification is ON by default; SKIP_TLS_VERIFY=1 turns it off process-wide
# (--skip-tls-verify), for every outbound call the proxy makes.
SKIP_TLS_VERIFY=1 bash serve_buster_ripper.sh

# Preserved Thinking is ON by default (--preserved-thinking): forwarded
# /v1/messages requests to vLLM or an unknown backend carry
# enable_thinking=true and clear_thinking=false in chat_template_kwargs, unless
# the client already set them. SGLang v0.5.9's /v1/messages endpoint discards
# chat_template_kwargs, so nothing is injected there; vLLM's Anthropic endpoint
# supports the pair. Turn it off:
PRESERVED_THINKING=0 bash serve_buster_ripper.sh
```

Preserved Thinking is the serve script's default, not an opt-in: `PRESERVED_THINKING`
defaults to `1`, so `--preserved-thinking` reaches the binary unless you set it to `0`.
The binary itself defaults to off; the flag turns it on where the backend honors
the pair. That default-on posture is the
recorded decision **`FK-011 · preserved-thinking serve default`** (Option A — keep the
default on and document it), not an implicit side effect.

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

### Collapsed dump formats (`--dump-format`)

The per-turn layout above is the `documented` format — the default, and itself
one value of the `--dump-format` selector:

```
--dump-format [documented|change-focused|self-contained]   (default: documented)
```

`change-focused` and `self-contained` are **collapsed** formats: each turn's
record is buffered in memory and later appended to **one file per session** —
`DUMP_DIR/<session_id>/<format>.log` (e.g. `change-focused.log`) — in a single
bulk write instead of per-turn files. Records land in arrival order and carry
the same summary lines as `index.txt`. A flush fires when a session's pending
rendered text reaches `--dump-flush-bytes` (default 4 MiB), every
`--dump-flush-interval` seconds (default 10), and at shutdown.

- `change-focused` — turn 0 stores the full normalized body **and** its
  rendering; every later turn stores its unified diff.
- `self-contained` — every turn stores its complete rendering (turn 0 also the
  full normalized body).

```bash
DUMP_FORMAT=change-focused DUMP_DIR=~/mllm/prompt-diffs bash serve_buster_ripper.sh
# or directly:
buster-ripper --dump-format self-contained --dump-dir ~/mllm/prompt-diffs
```

**Reading a collapsed artifact** — spot cache busters exactly as with the
index: a non-zero `-` count in a `+A/-R lines` summary means content changed
near position 0.

```bash
cd ~/mllm/prompt-diffs/<session_id>/

# change-focused: summary lines first, then the suspect turn's inline diff
grep -E '^turn [0-9]{3}  [0-9]{2}:[0-9]{2}' change-focused.log
less change-focused.log          # jump to "--- turn NNN diff ---"

# self-contained: summary lines first, then the suspect turn's full rendering
grep -E '^turn [0-9]{3}  [0-9]{2}:[0-9]{2}' self-contained.log
less self-contained.log          # jump to "--- turn NNN rendered ---"
```

**Turns are arrival-ordered, not one linear conversation.** In a real dump,
42 of 137 transitions had *fewer* messages than the previous turn (Claude Code
sends interleaved and compacted requests) — read turns in arrival order rather
than as a single thread.

**Pre-existing session directories:** switching formats neither migrates nor
removes documented-layout files (`turn_000.json`, `turn_NNN.diff`,
`index.txt`); a collapsed `.log` is written alongside them.

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
