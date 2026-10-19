"""POST /v1/messages/count_tokens — synthetic response for Claude Code compaction."""

import json
import logging

from aiohttp import web

from .. import compaction, config, session as session_mod
from ..compaction import CompactionConfig
from ..normalize import strip_billing_nonce
from ..session import estimate_tokens, session_id


log = logging.getLogger("buster-ripper")


async def _read_body(request: web.Request) -> dict:
    """Parse the body; a syntax error falls back, anything but an object is a 400."""
    try:
        body = await request.json()
    except json.JSONDecodeError:
        return {}
    except (UnicodeDecodeError, LookupError):
        body = None
    if not isinstance(body, dict):
        raise web.HTTPBadRequest(
            text=json.dumps({"error": "Invalid JSON"}), content_type="application/json",
        )
    return body


def _nudge_trigger_name(real_tokens: int, avg_latency: float) -> str:
    """Which nudge condition fired for this probe, for the log line."""
    kv_trigger = compaction._kv_cache_usage >= config.COMPACT_KV_RATIO
    session_trigger = (
        real_tokens >= int(config.MAX_MODEL_LEN * config.COMPACT_TOKEN_RATIO)
        and avg_latency >= config.COMPACT_LATENCY_MS
    )
    if kv_trigger and not session_trigger:
        return "kv_pressure"
    if session_trigger and not kv_trigger:
        return "session"
    return "both"


def _nudge_tokens(sid: str, real_tokens: int, avg_latency: float, cfg: CompactionConfig) -> int:
    """The reported count: inflated to the compaction target when the nudge fires.

    The nudge fires when the KV cache is over its ratio, or when token fill AND
    average TTFT both exceed their thresholds (`should_nudge_compact`).
    """
    if not compaction.should_nudge_compact(
        real_tokens,
        avg_latency,
        compaction._kv_cache_usage,
        cfg,
    ):
        return real_tokens
    reported = compaction.compact_token_count(config.MAX_MODEL_LEN, config.COMPACT_NUDGE_RATIO)
    log.info(
        "session %s: nudging compaction tokens=%d→%d avg_latency=%.0f ms kv=%.1f%% trigger=%s",
        sid, real_tokens, reported, avg_latency, compaction._kv_cache_usage * 100,
        _nudge_trigger_name(real_tokens, avg_latency),
    )
    return reported


async def handle_count_tokens(request: web.Request) -> web.Response:
    """Respond to Claude Code's context-size probe.

    Claude Code calls POST /v1/messages/count_tokens periodically to decide
    when to auto-compact the conversation context. vLLM returns 404, so without
    this endpoint compaction never fires and the context grows until TTFT
    becomes unbearable.

    Strategy:
      - Return the input_tokens from the last response for this session.
      - If both token fill AND average TTFT exceed thresholds, inflate the
        reported count to trigger compaction.
      - If no stats yet, fall back to a character-count estimate.
    """
    body = await _read_body(request)

    # Strip the billing nonce before keying, as /v1/messages does when it records the
    # stats this lookup reads: the nonce changes every request, so the un-stripped digest
    # matches nothing. Only that step — session_id() hashes system alone
    # (session.py:93-98), and the tools/messages steps assume every member is a mapping,
    # which is CR4-F-01: a body base answers an estimate for would answer a 500.
    # tests/test_count_tokens_parity.py keeps the narrowed key equal to the full one.
    strip_billing_nonce(body)
    sid = session_id(body)
    stats = session_mod._stats.get(sid)

    if stats and stats.input_tokens > 0:
        real_tokens = stats.input_tokens
    else:
        real_tokens = estimate_tokens(body)
        log.debug("session %s: count_tokens no-stats estimate=%d", sid, real_tokens)

    avg_latency = stats.avg_latency_ms if stats else 0.0
    cfg = CompactionConfig(
        max_model_len=config.MAX_MODEL_LEN,
        token_ratio=config.COMPACT_TOKEN_RATIO,
        latency_threshold_ms=config.COMPACT_LATENCY_MS,
        kv_ratio=config.COMPACT_KV_RATIO,
    )
    reported = _nudge_tokens(sid, real_tokens, avg_latency, cfg)

    return web.json_response({"input_tokens": reported})
