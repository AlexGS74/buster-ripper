"""Global configuration — mutable values set by main() at startup.

All other modules import this module and read config.<NAME> so that
changes made by main() are visible everywhere without re-import.
"""

import dataclasses
import re
from pathlib import Path
from typing import Optional

# ── Core proxy settings ───────────────────────────────────────────────────────
UPSTREAM: str = "http://localhost:30000"
# Upstream TLS posture: certificates are verified unless the opt-out is
# explicitly requested (--skip-tls-verify / SKIP_TLS_VERIFY=1, for
# self-signed upstreams). Never inferred from the URL scheme or host.
SKIP_TLS_VERIFY: bool = False
STRIP_DATE: bool = False
VERBOSE: bool = False
DUMP_DIR: Optional[Path] = None
BACKEND_TYPE_OVERRIDE: Optional[str] = None  # CLI override for backend detection

# ── Dump format settings ──────────────────────────────────────────────────────
# Closed set of output formats. "documented" is today's per-turn layout and the
# default; the two collapsed values buffer records in memory and flush them to
# one <format>.log artifact per session.
DUMP_FORMATS: tuple[str, ...] = ("documented", "change-focused", "self-contained")
DUMP_FORMAT: str = "documented"

# ── Dump flush settings ───────────────────────────────────────────────────────
# Flush triggers for the collapsed formats — the documented layout writes
# per-turn files and never buffers, so these have no effect on it. The timer
# fires every DUMP_FLUSH_INTERVAL seconds; the volume trigger flushes once a
# session's pending rendered-text bytes reach DUMP_FLUSH_BYTES; shutdown
# always flushes.
DUMP_FLUSH_INTERVAL: int = 10             # seconds between periodic flushes
DUMP_FLUSH_BYTES: int = 4 * 1024 * 1024   # pending rendered-text bytes before a volume flush

# ── Image handling settings ──────────────────────────────────────────────────
STRIP_IMAGES: bool = False
IMAGE_CACHE_MAX_SIZE: int = 100
IMAGE_CACHE_TTL: int = 300  # seconds

# ── Thinking / agentic settings ───────────────────────────────────────────────
# Preserved Thinking: injects {"enable_thinking": true, "clear_thinking": false}
# into /v1/messages (CC) requests so the model retains its reasoning chain across
# turns. "clear_thinking": false keeps <think>...</think> in assistant history
# instead of replacing it with </think>. Injected for vLLM and unknown backends;
# SGLang v0.5.9's /v1/messages discards chat_template_kwargs, so nothing is
# injected there. vLLM's Anthropic endpoint supports the pair.
PRESERVED_THINKING: bool = False

# ── Eval-mode settings ────────────────────────────────────────────────────────
EVAL_MODE: bool = False
EVAL_MAX_TOKENS: int = 0       # 0 = no cap (let vLLM use model default)
EVAL_THINKING: bool = False    # passed to inject_chat_template_kwargs strategy
EVAL_THINKING_BUDGET: int = 0  # 0 = no budget (unlimited thinking)

# ── Compaction policy ─────────────────────────────────────────────────────────
MAX_MODEL_LEN: int = 200_000
COMPACT_TOKEN_RATIO: float = 0.75   # nudge when tokens > this fraction of max
COMPACT_LATENCY_MS: float = 15_000  # nudge when TTFT > this (ms)
COMPACT_NUDGE_RATIO: float = 0.95   # nudged count = this × MAX_MODEL_LEN
COMPACT_KV_RATIO: float = 0.75      # nudge when server KV cache usage > this
KV_POLL_INTERVAL: int = 10          # seconds between /metrics polls
STATS_DB: Optional[Path] = None     # SQLite path; None = in-memory only

# ── Eval-mode model profiles ──────────────────────────────────────────────────
# Each profile is a named set of strategies for working around model-specific
# quirks when running lm-evaluation-harness.
#
# Strategies (all optional, compose independently):
#
#   inject_chat_template_kwargs
#       Injects chat_template_kwargs into the request body.
#       Used by models (e.g. GLM-4.7) that control thinking via this field.
#
#   strip_code_fences
#       Strips leading ```...``` wrapper from response content.
#       lm-eval's build_predictions_instruct filter truncates at the first ```
#       it finds, so a response that *starts* with ``` drops the entire body.
#       Strip here so the filter receives bare code and assembles correctly.
#
#   copy_reasoning_to_content
#       When content is null/empty and reasoning_content is present, copies
#       reasoning_content → content so lm-eval can score the answer.
#       Needed when thinking is enabled and the model separates reasoning from
#       the final answer.
#
# Model-agnostic strategies (always active in eval mode, not per-profile):
#   - strip max_gen_toks from request (lm-eval internal field, not OpenAI)
#   - strip empty Bearer auth header (lm-eval sends "Bearer " with no token)


@dataclasses.dataclass(frozen=True)
class ModelProfile:
    """Strategy flags for a specific model's eval-mode quirks."""
    inject_chat_template_kwargs: bool = False
    strip_code_fences: bool = False
    copy_reasoning_to_content: bool = False

    def describe(self) -> str:
        active = [name for name, val in dataclasses.asdict(self).items() if val]
        return ", ".join(active) if active else "none"


# Registry of named profiles. "passthrough" = no model-specific strategies.
PROFILES: dict[str, ModelProfile] = {
    "passthrough": ModelProfile(),
    "glm47": ModelProfile(
        inject_chat_template_kwargs=True,
        strip_code_fences=True,
        copy_reasoning_to_content=True,
    ),
    # Add new profiles here as new models are onboarded, e.g.:
    # "qwen": ModelProfile(strip_code_fences=True),
}

EVAL_PROFILE: ModelProfile = PROFILES["passthrough"]

# ── Context overflow retry settings ────────────────────────────────────────────
# Regex patterns for detecting SGLang context overflow errors
CONTEXT_OVERFLOW_RE = re.compile(r"Context overflow.*?(\d+)\s*>\s*(\d+)")
CTX_LIMIT_RE = re.compile(r"max_completion_tokens.*?(\d+)")
MIN_COMPLETION_TOKENS = 256
