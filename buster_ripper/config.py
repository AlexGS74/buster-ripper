"""Global configuration — mutable values set by main() at startup.

All other modules import this module and read config.<NAME> so that
changes made by main() are visible everywhere without re-import.
"""

import dataclasses
from pathlib import Path
from typing import Optional

# ── Core proxy settings ───────────────────────────────────────────────────────
UPSTREAM: str = "http://localhost:30000"
STRIP_DATE: bool = False
VERBOSE: bool = False
DUMP_DIR: Optional[Path] = None

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
