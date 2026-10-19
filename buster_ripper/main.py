"""CLI entrypoint."""

import logging
from pathlib import Path

import click

from . import config, session
from .server import create_app


def _set_dump_format(ctx, param, value):
    """Assign config.DUMP_FORMAT at parse time; main() takes no new parameter."""
    config.DUMP_FORMAT = value
    return value


def _set_dump_flush_interval(ctx, param, value):
    """Assign config.DUMP_FLUSH_INTERVAL at parse time; main() takes no new parameter."""
    config.DUMP_FLUSH_INTERVAL = value
    return value


def _set_dump_flush_bytes(ctx, param, value):
    """Assign config.DUMP_FLUSH_BYTES at parse time; main() takes no new parameter."""
    config.DUMP_FLUSH_BYTES = value
    return value


def _set_skip_tls_verify(ctx, param, value):
    """Assign config.SKIP_TLS_VERIFY at parse time; main() takes no new parameter."""
    config.SKIP_TLS_VERIFY = value
    return value


def _apply_cli_config(**cli):
    """Assign every CLI-passed option onto its config global.

    Pure assignments only; the side-effectful dump-dir and stats-db blocks
    stay in main(). Keys are the CLI parameter names.
    """
    config.UPSTREAM = cli["upstream"]
    config.STRIP_DATE = cli["strip_date"]
    config.STRIP_IMAGES = cli["strip_images"]
    config.VERBOSE = cli["verbose"]
    config.PRESERVED_THINKING = cli["preserved_thinking"]
    config.MAX_MODEL_LEN = cli["max_model_len"]
    config.COMPACT_TOKEN_RATIO = cli["compact_token_ratio"]
    config.COMPACT_LATENCY_MS = cli["compact_latency_ms"]
    config.COMPACT_KV_RATIO = cli["compact_kv_ratio"]
    config.EVAL_MODE = cli["eval_mode"]
    config.EVAL_PROFILE = config.PROFILES[cli["eval_profile"]]
    config.EVAL_MAX_TOKENS = cli["eval_max_tokens"]
    config.EVAL_THINKING = cli["eval_thinking"]
    config.EVAL_THINKING_BUDGET = cli["eval_thinking_budget"]
    config.IMAGE_CACHE_MAX_SIZE = cli["image_cache_max_size"]
    config.IMAGE_CACHE_TTL = cli["image_cache_ttl"]
    config.BACKEND_TYPE_OVERRIDE = cli["backend_type"]


@click.command()
@click.option("--upstream", default=config.UPSTREAM, show_default=True,
              help="vLLM base URL.")
@click.option("--skip-tls-verify", is_flag=True, expose_value=False,
              callback=_set_skip_tls_verify,
              help="Skip upstream TLS certificate verification (for self-signed "
                   "certificates). Verification is on by default.")
@click.option("--host", default="0.0.0.0", show_default=True,
              help="Bind address.")
@click.option("--port", default=30001, show_default=True, type=int,
              help="Listen port.")
@click.option("--strip-date", is_flag=True,
              help="Strip framework-injected 'Today's date is ...' from user messages.")
@click.option("--strip-images", is_flag=True,
              help="Strip images from user messages and cache them for later retrieval.")
@click.option("--verbose", is_flag=True,
              help="Log every normalization applied and per-turn token/latency stats.")
@click.option("--dump-dir", metavar="PATH", default=None,
              help="Enable session prompt diffing under PATH/<session_id>/.")
@click.option("--dump-format", type=click.Choice(list(config.DUMP_FORMATS)),
              default="documented", show_default=True, expose_value=False,
              callback=_set_dump_format,
              help="Dump output format: documented = per-turn files; "
                   "change-focused/self-contained = one collapsed <format>.log "
                   "per session, bulk-flushed.")
@click.option("--dump-flush-interval", type=click.IntRange(min=1), expose_value=False,
              default=config.DUMP_FLUSH_INTERVAL, show_default=True,
              callback=_set_dump_flush_interval,
              help="Seconds between periodic flushes of buffered dump records "
                   "(collapsed formats only).")
@click.option("--dump-flush-bytes", type=click.IntRange(min=1), expose_value=False,
              default=config.DUMP_FLUSH_BYTES, show_default=True,
              callback=_set_dump_flush_bytes,
              help="Flush buffered dump records once a session's pending "
                   "rendered-text bytes reach this (collapsed formats only).")
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
@click.option("--preserved-thinking", is_flag=True,
              help="Inject enable_thinking=true + clear_thinking=false into /v1/messages "
                   "requests for vLLM and unknown backends. Keeps the model's reasoning "
                   "chain visible across turns (Preserved Thinking). SGLang v0.5.9's "
                   "/v1/messages discards chat_template_kwargs, so nothing is injected "
                   "there; vLLM's Anthropic endpoint supports the pair.")
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
@click.option("--image-cache-max-size", default=100, show_default=True, type=int,
              help="Maximum number of images to cache per session.")
@click.option("--image-cache-ttl", default=300, show_default=True, type=int,
              help="Image cache TTL in seconds.")
@click.option("--backend-type", type=click.Choice(["sglang", "vllm"]),
              help="Override backend type (default: auto-detect)")
def main(
    upstream, host, port, strip_date, strip_images, verbose, dump_dir,
    max_model_len, compact_token_ratio, compact_latency_ms, compact_kv_ratio,
    stats_db, preserved_thinking,
    eval_mode, eval_profile, eval_max_tokens, eval_thinking, eval_thinking_budget,
    image_cache_max_size, image_cache_ttl, backend_type,
):
    """buster-ripper — Anthropic /v1/messages normalizing proxy for vLLM prefix-cache stability."""
    _apply_cli_config(
        upstream=upstream, strip_date=strip_date, strip_images=strip_images,
        verbose=verbose, preserved_thinking=preserved_thinking,
        max_model_len=max_model_len, compact_token_ratio=compact_token_ratio,
        compact_latency_ms=compact_latency_ms, compact_kv_ratio=compact_kv_ratio,
        eval_mode=eval_mode, eval_profile=eval_profile,
        eval_max_tokens=eval_max_tokens, eval_thinking=eval_thinking,
        eval_thinking_budget=eval_thinking_budget,
        image_cache_max_size=image_cache_max_size,
        image_cache_ttl=image_cache_ttl, backend_type=backend_type,
    )

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
        "Tool sorting: enabled | Date stripping: %s | Image stripping: %s | Preserved thinking: %s | Dump dir: %s | Eval mode: %s",
        "enabled" if strip_date else "disabled",
        "enabled" if strip_images else "disabled",
        "enabled (enable_thinking=true, clear_thinking=false)" if preserved_thinking else "disabled",
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

    # Create and run aiohttp app
    from aiohttp import web
    app = create_app()
    web.run_app(app, host=host, port=port, print=None)


if __name__ == "__main__":
    main()
