# T005 evidence — DUMP_FORMAT serve-script wiring + README dump-section extension
# Recorded 2026-09-28. A shim `buster-ripper` on PATH printed each exec argv token as ARG:<token>.

## claim-32 done-when: argv recording, DUMP_FORMAT set and unset

=== run 1: DUMP_FORMAT unset ===
$ env -u DUMP_FORMAT PATH="$SHIM:$PATH" bash serve_buster_ripper.sh
ARG:--upstream ARG:http://localhost:30000 ARG:--host ARG:0.0.0.0 ARG:--port ARG:30001 ARG:--dump-dir ARG:/Users/alexhubitski/mllm/prompt-diffs ARG:--preserved-thinking
→ no --dump-format flag; CLI default 'documented' applies

=== run 2: DUMP_FORMAT empty ===
$ DUMP_FORMAT= PATH="$SHIM:$PATH" bash serve_buster_ripper.sh
ARG:--upstream ARG:http://localhost:30000 ARG:--host ARG:0.0.0.0 ARG:--port ARG:30001 ARG:--dump-dir ARG:/Users/alexhubitski/mllm/prompt-diffs ARG:--preserved-thinking
→ no --dump-format flag

=== run 3: DUMP_FORMAT=change-focused ===
$ DUMP_FORMAT=change-focused PATH="$SHIM:$PATH" bash serve_buster_ripper.sh
ARG:--upstream ARG:http://localhost:30000 ARG:--host ARG:0.0.0.0 ARG:--port ARG:30001 ARG:--dump-dir ARG:/Users/alexhubitski/mllm/prompt-diffs ARG:--dump-format ARG:change-focused ARG:--preserved-thinking
→ exactly one added flag: --dump-format change-focused

=== run 4: DUMP_FORMAT=self-contained with custom DUMP_DIR peer ===
$ DUMP_FORMAT=self-contained DUMP_DIR=/tmp/peer-check PATH="$SHIM:$PATH" bash serve_buster_ripper.sh
ARG:--upstream ARG:http://localhost:30000 ARG:--host ARG:0.0.0.0 ARG:--port ARG:30001 ARG:--dump-dir ARG:/tmp/peer-check ARG:--dump-format ARG:self-contained ARG:--preserved-thinking
→ DUMP_DIR peer mapping untouched (custom value forwarded); flag rides EXTRA_FLAGS beside the peer

All four runs: every argv token is a --flag or its value — no positional arguments are forwarded.

## Failure path: invalid value rejected at parse time (closed set)

$ .venv/bin/buster-ripper --dump-format bogus
Usage: buster-ripper [OPTIONS]
Try 'buster-ripper --help' for more information.

Error: Invalid value for '--dump-format': 'bogus' is not one of 'documented', 'change-focused', 'self-contained'.
→ a bogus DUMP_FORMAT through the script terminates before the server starts

## serve_buster_ripper.sh diff (this increment)

--- /var/folders/cy/3w2sfwd55yjbqhzvrs7dw7lh0000gn/T//git-blob-A2xfxh/serve_buster_ripper.sh	2026-09-28 02:34:03
+++ serve_buster_ripper.sh	2026-09-28 02:32:51
@@ -12,6 +12,7 @@
 #   bash serve_buster_ripper.sh
 #   STRIP_DATE=1 bash serve_buster_ripper.sh   # also strip daily date injection (SGLang only)
 #   DUMP_DIR="" bash serve_buster_ripper.sh    # disable prompt diffing
+#   DUMP_FORMAT=change-focused bash serve_buster_ripper.sh   # collapsed one-file dump format
 #
 # Then point Claude Code at port 30001:
 #   ANTHROPIC_BASE_URL=http://localhost:30001 claude ...
@@ -23,12 +24,14 @@
 STRIP_DATE=${STRIP_DATE:-0}
 VERBOSE=${VERBOSE:-0}
 DUMP_DIR=${DUMP_DIR:-${HOME}/mllm/prompt-diffs}
+DUMP_FORMAT=${DUMP_FORMAT:-}
 PRESERVED_THINKING=${PRESERVED_THINKING:-1}   # 1 = Preserved Thinking (enable_thinking=true, clear_thinking=false)
 
 EXTRA_FLAGS=""
 [[ "${STRIP_DATE}"          == "1" ]] && EXTRA_FLAGS="${EXTRA_FLAGS} --strip-date"
 [[ "${VERBOSE}"             == "1" ]] && EXTRA_FLAGS="${EXTRA_FLAGS} --verbose"
 [[ -n "${DUMP_DIR}"               ]] && EXTRA_FLAGS="${EXTRA_FLAGS} --dump-dir ${DUMP_DIR}"
+[[ -n "${DUMP_FORMAT}"            ]] && EXTRA_FLAGS="${EXTRA_FLAGS} --dump-format ${DUMP_FORMAT}"
 [[ "${PRESERVED_THINKING}"  == "1" ]] && EXTRA_FLAGS="${EXTRA_FLAGS} --preserved-thinking"
 
 echo "buster-ripper → ${UPSTREAM}  listening on ${HOST}:${PORT}"

## README.md dump section after this increment (documented block lines 134-162 unchanged; new subsection appended)

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

## README workflow validated against a real artifact (actual dump code, fresh tmp DUMP_DIR)

Generated 3 turns under change-focused (turn 1 renames a tool = cache buster), flush_all(), then ran
the README's documented commands against DUMP_DIR/<sid>/change-focused.log:

$ grep -E '^turn [0-9]{3}  [0-9]{2}:[0-9]{2}' change-focused.log
turn 000  06:35:00  FULL  1 msgs  2 tools  0.4 KB
turn 001  06:35:00  +7/-2 lines  2 msgs  2 tools
turn 002  06:35:00  +5/-0 lines  3 msgs  2 tools

$ grep -n '^--- turn' change-focused.log
2:--- turn 000 body ---
22:--- turn 000 rendered ---
50:--- turn 001 diff ---
78:--- turn 002 diff ---

→ the summary grep yields index.txt-style lines only; the "--- turn NNN diff ---" / "--- turn NNN rendered ---"
  anchors the README's less workflow points at exist verbatim; the turn-001 non-zero "-2" count is the
  tool-rename cache buster, spotted exactly as the section documents.
