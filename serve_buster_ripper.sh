#!/usr/bin/env bash
set -euo pipefail

# buster-ripper — rips cache busters out of Claude Code requests before they
# reach vLLM, keeping the KV cache prefix stable across turns (22% → 95%+).
#
# Prerequisites:
#   Install the CLI globally with: uv tool install .
#   Or run directly with: uv run buster-ripper --help
#
# Usage:
#   bash serve_buster_ripper.sh
#   STRIP_DATE=1 bash serve_buster_ripper.sh   # also strip daily date injection (SGLang only)
#   DUMP_DIR="" bash serve_buster_ripper.sh    # disable prompt diffing
#   DUMP_FORMAT=change-focused bash serve_buster_ripper.sh   # collapsed one-file dump format
#   SKIP_TLS_VERIFY=1 bash serve_buster_ripper.sh  # skip upstream TLS verification (self-signed certs)
#
# Then point Claude Code at port 30001:
#   ANTHROPIC_BASE_URL=http://localhost:30001 claude ...
# or update the glm47 bash function in ~/.bashrc

UPSTREAM=${UPSTREAM:-http://localhost:30000}
HOST=${HOST:-0.0.0.0}
PORT=${PORT:-30001}
STRIP_DATE=${STRIP_DATE:-0}
VERBOSE=${VERBOSE:-0}
SKIP_TLS_VERIFY=${SKIP_TLS_VERIFY:-0}
DUMP_DIR=${DUMP_DIR-${HOME}/mllm/prompt-diffs}
DUMP_FORMAT=${DUMP_FORMAT:-}
PRESERVED_THINKING=${PRESERVED_THINKING:-1}   # 1 = Preserved Thinking (enable_thinking=true, clear_thinking=false)

# Assembled as an array so each append contributes whole words: a single string
# here word-splits every value that contains whitespace, which truncates a path
# like "/tmp/my docs" at its space and hands the remainder to the binary as an
# extra positional word (and lets a whitespace-only value swallow the next flag).
EXTRA_FLAGS=()
[[ "${STRIP_DATE}"          == "1" ]] && EXTRA_FLAGS+=(--strip-date)
[[ "${VERBOSE}"             == "1" ]] && EXTRA_FLAGS+=(--verbose)
[[ "${SKIP_TLS_VERIFY}"     == "1" ]] && EXTRA_FLAGS+=(--skip-tls-verify)
[[ -n "${DUMP_DIR}"               ]] && EXTRA_FLAGS+=(--dump-dir "${DUMP_DIR}")
[[ -n "${DUMP_FORMAT}"            ]] && EXTRA_FLAGS+=(--dump-format "${DUMP_FORMAT}")
[[ "${PRESERVED_THINKING}"  == "1" ]] && EXTRA_FLAGS+=(--preserved-thinking)

echo "buster-ripper → ${UPSTREAM}  listening on ${HOST}:${PORT}"

# The ${arr[@]+"${arr[@]}"} guard expands to zero words when every append above
# was suppressed. Bare "${arr[@]}" is not nounset-safe on bash 3.2 (this script's
# `/usr/bin/env bash` resolves there on macOS), where an empty array aborts with
# "unbound variable" — so an all-suppressed run has to use the guarded form.
exec buster-ripper \
  --upstream "${UPSTREAM}" \
  --host     "${HOST}" \
  --port     "${PORT}" \
  ${EXTRA_FLAGS[@]+"${EXTRA_FLAGS[@]}"}
