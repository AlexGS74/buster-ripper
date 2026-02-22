#!/usr/bin/env bash
set -euo pipefail

# buster-ripper — rips cache busters out of Claude Code requests before they
# reach vLLM, keeping the KV cache prefix stable across turns (22% → 95%+).
#
# Usage:
#   bash buster-ripper/serve_buster_ripper.sh
#   STRIP_DATE=1 bash buster-ripper/serve_buster_ripper.sh   # also strip daily date injection
#   DUMP_DIR="" bash buster-ripper/serve_buster_ripper.sh    # disable prompt diffing
#
# Then point Claude Code at port 30001:
#   ANTHROPIC_BASE_URL=http://localhost:30001 claude ...
# or update the glm47 bash function in ~/.bashrc

UPSTREAM=${UPSTREAM:-http://localhost:30000}
HOST=${HOST:-0.0.0.0}
PORT=${PORT:-30001}
STRIP_DATE=${STRIP_DATE:-0}
VERBOSE=${VERBOSE:-0}
DUMP_DIR=${DUMP_DIR:-${HOME}/mllm/prompt-diffs}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

EXTRA_FLAGS=""
[[ "${STRIP_DATE}" == "1" ]] && EXTRA_FLAGS="${EXTRA_FLAGS} --strip-date"
[[ "${VERBOSE}"     == "1" ]] && EXTRA_FLAGS="${EXTRA_FLAGS} --verbose"
[[ -n "${DUMP_DIR}"       ]] && EXTRA_FLAGS="${EXTRA_FLAGS} --dump-dir ${DUMP_DIR}"

echo "buster-ripper → ${UPSTREAM}  listening on ${HOST}:${PORT}"

exec uv run "${SCRIPT_DIR}/buster_ripper.py" \
  --upstream "${UPSTREAM}" \
  --host     "${HOST}" \
  --port     "${PORT}" \
  ${EXTRA_FLAGS}
