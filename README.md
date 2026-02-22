# buster-ripper

Normalizing HTTP proxy that sits between Claude Code and vLLM, ripping out
the cache busters that Claude Code injects into every request so the KV cache
prefix stays stable across turns (22% → 95%+ cache hit rate).

## Problem

vLLM V1 prefix caching hashes consecutive token blocks. Any token that changes
at position N invalidates the KV cache for all positions > N. For GLM-4.7,
tools are injected at position 0 of the formatted prompt — before the system
prompt — so a single reordered tool definition causes a full cache miss.

Observed cache busters (identified by audit, 2026-02-21):

| Cache buster | Scope | Fixed here |
|---|---|---|
| MCP tool reordering (reconnects return tools in arbitrary order) | Per-turn | ✅ sort by name |
| `currentDate` injection (`Today's date is YYYY-MM-DD.`) appended to MEMORY.md by the Claude Code framework | Daily | ✅ `--strip-date` |
| `gitStatus` block (branch, recent commits) | Per-conversation | ✗ changes once per session, acceptable |
| `<system-reminder>` CLAUDE.md / skills list | Per-turn if content changes | ✗ framework-level |

## Normalizations

1. **Sort tools by name** — always on. Makes the tool block byte-identical
   across MCP reconnects and different server orderings.

2. **Strip `currentDate`** — opt-in (`--strip-date`). Removes the line
   `Today's date is YYYY-MM-DD.` from user message content before forwarding.
   Omit if your prompts rely on the model knowing the current date.

## Install

No install needed — `uv run` fetches dependencies inline.

```bash
# Dependencies declared in buster_ripper.py with PEP 723 inline metadata:
# fastapi, uvicorn[standard], httpx
```

## Run

```bash
# Default: listen on 127.0.0.1:30001, forward to localhost:30000
bash buster-ripper/serve_buster_ripper.sh

# With date stripping
STRIP_DATE=1 bash buster-ripper/serve_buster_ripper.sh

# Verbose (log every normalization applied)
VERBOSE=1 bash buster-ripper/serve_buster_ripper.sh

# Enable session prompt diffing (see Dump Mode below)
DUMP_DIR=~/mllm/prompt-diffs bash buster-ripper/serve_buster_ripper.sh

# Custom upstream / port
UPSTREAM=http://localhost:30000 PORT=30001 bash buster-ripper/serve_buster_ripper.sh
```

Or directly:
```bash
uv run buster-ripper/buster_ripper.py --help
```

## Dump Mode (`--dump-dir`)

Enables session-aware prompt capture and diffing — useful for identifying
remaining cache busters after normalization.

```bash
DUMP_DIR=~/mllm/prompt-diffs bash buster-ripper/serve_buster_ripper.sh
```

**Session ID** = first 12 hex chars of SHA-256(system prompt text). Stable
within a Claude Code session; changes between sessions.

**Output layout** under `DUMP_DIR/<session_id>/`:

```
turn_000.json    # full normalized request body (turn 0)
turn_001.diff    # unified diff: turn 000 → turn 001
turn_002.diff    # unified diff: turn 001 → turn 002
...
index.txt        # one-line summary per turn (timestamp, +/- lines, msg count)
```

**What the diff shows:** each request is rendered as labelled sections before
diffing, so changes are easy to spot:

```diff
 [tools] (83 total)
 # atlassianAddComment, atlassianAddWatcher, ...

-[message 4 / user]
-<system-reminder>
-Today's date is 2026-02-20.         ← cache buster: date changed
+[message 4 / user]
+<system-reminder>
+Today's date is 2026-02-21.
```

**Reading the index:**
```
turn 000  10:31:02  FULL  3 msgs  83 tools  412.3 KB
turn 001  10:32:18  +47/-3 lines  5 msgs  83 tools
turn 002  10:33:05  +892/-0 lines  7 msgs  83 tools   ← large file read
```

A large `+` line count on a turn = new content added to context (file reads,
long tool outputs). A non-zero `-` count on the tools section = tool list
changed = cache busted from position 0.

## Configure Claude Code

Update the `glm47` function in `~/.bashrc` to point at the proxy port:

```bash
function glm47() {
  ANTHROPIC_BASE_URL="http://localhost:30001" \   # ← proxy, not vLLM directly
  ANTHROPIC_API_KEY="sk-local" \
  ANTHROPIC_MODEL="glm4.7" \
  ANTHROPIC_DEFAULT_OPUS_MODEL="glm4.7" \
  ANTHROPIC_DEFAULT_SONNET_MODEL="glm4.7" \
  ANTHROPIC_DEFAULT_HAIKU_MODEL="glm4.7" \
  CLAUDE_CODE_SUBAGENT_MODEL="glm4.7" \
  /home/alex/.local/bin/claude "$@"
}
```

## Verify it's working

Run with `VERBOSE=1` and watch for normalization log lines:

```
10:32:01 [buster-ripper] tools reordered: [Bash, Edit, Glob, ...] → [atlassianUserInfo, browser_click, ...]
```

If tools are already in a stable order, nothing is logged (no unnecessary work done).

## Architecture

```
Claude Code  →  buster-ripper (:30001)  →  vLLM (:30000)
                      │
                      ├─ strip billing nonce (cch=...)
                      ├─ strip cache_control fields
                      ├─ sort tools[] by name
                      ├─ drop empty tools arrays
                      ├─ (optional) strip currentDate
                      └─ pass through all other endpoints unchanged
```

Only `/v1/messages` POST requests are modified. All other endpoints
(`/v1/models`, `/health`, etc.) are proxied transparently.
