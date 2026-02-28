"""CLI entrypoint."""

import logging
from pathlib import Path

import click
import uvicorn

from . import config, session
from .app import app


@click.command()
@click.option("--upstream", default="http://localhost:30000", show_default=True,
              help="vLLM base URL.")
@click.option("--host", default="0.0.0.0", show_default=True,
              help="Bind address.")
@click.option("--port", default=30001, show_default=True, type=int,
              help="Listen port.")
@click.option("--strip-date", is_flag=True,
              help="Strip framework-injected 'Today's date is ...' from user messages.")
@click.option("--verbose", is_flag=True,
              help="Log every normalization applied and per-turn token/latency stats.")
@click.option("--dump-dir", metavar="PATH", default=None,
              help="Enable session prompt diffing under PATH/<session_id>/.")
@click.option("--max-model-len", default=200_000, show_default=True, type=int,
              help="Max context length of the served model (used by compaction policy).")
@click.option("--compact-token-ratio", default=0.75, show_default=True, type=float,
              help="Nudge compaction when tokens exceed this fraction of --max-model-len.")
@click.option("--compact-latency-ms", default=15_000.0, show_default=True, type=float,
              help="Nudge compaction when average TTFT exceeds this value in ms.")
@click.option("--compact-kv-ratio", default=0.75, show_default=True, type=float,
              help="Nudge compaction when server KV cache usage exceeds this fraction. "
                   "Set to 1.0 to disable.")
@click.option("--stats-db", metavar="PATH", default=None,
              help="SQLite file for session stats persistence across restarts.")
@click.option("--eval-mode", is_flag=True,
              help="Eval proxy mode for lm-evaluation-harness: drops max_gen_toks, strips empty "
                   "auth, applies --eval-profile strategies.")
@click.option("--eval-profile",
              type=click.Choice(list(config.PROFILES.keys())), default="passthrough",
              show_default=True,
              help=f"Model-specific eval strategy profile. Available: {', '.join(config.PROFILES)}.")
@click.option("--eval-max-tokens", default=0, show_default=True, type=int,
              help="max_tokens cap injected into requests when --eval-mode is active. 0 = no cap.")
@click.option("--eval-thinking", is_flag=True,
              help="Pass enable_thinking=true via chat_template_kwargs (requires a profile with "
                   "inject_chat_template_kwargs, e.g. --eval-profile glm47).")
@click.option("--eval-thinking-budget", default=0, show_default=True, type=int,
              help="thinking_budget cap for --eval-thinking. 0 = unlimited.")
def main(
    upstream, host, port, strip_date, verbose, dump_dir,
    max_model_len, compact_token_ratio, compact_latency_ms, compact_kv_ratio,
    stats_db, eval_mode, eval_profile, eval_max_tokens, eval_thinking, eval_thinking_budget,
):
    """buster-ripper — Anthropic /v1/messages normalizing proxy for vLLM prefix-cache stability."""
    config.UPSTREAM = upstream
    config.STRIP_DATE = strip_date
    config.VERBOSE = verbose
    config.MAX_MODEL_LEN = max_model_len
    config.COMPACT_TOKEN_RATIO = compact_token_ratio
    config.COMPACT_LATENCY_MS = compact_latency_ms
    config.COMPACT_KV_RATIO = compact_kv_ratio
    config.EVAL_MODE = eval_mode
    config.EVAL_PROFILE = config.PROFILES[eval_profile]
    config.EVAL_MAX_TOKENS = eval_max_tokens
    config.EVAL_THINKING = eval_thinking
    config.EVAL_THINKING_BUDGET = eval_thinking_budget

    if dump_dir:
        config.DUMP_DIR = Path(dump_dir).expanduser().resolve()
        config.DUMP_DIR.mkdir(parents=True, exist_ok=True)

    if stats_db:
        db_path = Path(stats_db).expanduser().resolve()
        session._db_conn = session.db_init(db_path)
        session._stats.update(session.db_load(session._db_conn))
        logging.getLogger("buster-ripper").info(
            "stats db: %s (%d sessions loaded)", db_path, len(session._stats)
        )

    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s [buster-ripper] %(message)s",
        datefmt="%H:%M:%S",
    )
    log = logging.getLogger("buster-ripper")
    log.info("Listening on %s:%d → upstream %s", host, port, upstream)
    log.info(
        "Tool sorting: enabled | Date stripping: %s | Dump dir: %s | Eval mode: %s",
        "enabled" if strip_date else "disabled",
        config.DUMP_DIR or "disabled",
        f"enabled (profile={eval_profile}, max_tokens={eval_max_tokens}, "
        f"thinking={'on' if eval_thinking else 'off'})" if eval_mode else "disabled",
    )
    log.info(
        "Compaction policy: max_len=%d token_ratio=%.2f latency_ms=%.0f "
        "nudge_ratio=%.2f kv_ratio=%.2f",
        max_model_len, compact_token_ratio, compact_latency_ms,
        config.COMPACT_NUDGE_RATIO, compact_kv_ratio,
    )

    uvicorn.run(app, host=host, port=port, log_level="warning")


if __name__ == "__main__":
    main()
