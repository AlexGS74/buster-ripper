"""POST /v1/messages/count_tokens — synthetic response for Claude Code compaction."""

import json
import logging

from fastapi import APIRouter, Request
from fastapi.responses import Response

from .. import compaction, config, session as session_mod
from ..session import estimate_tokens, session_id

router = APIRouter()
log = logging.getLogger("buster-ripper")


@router.post("/v1/messages/count_tokens")
async def count_tokens(request: Request) -> Response:
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
    raw_body = await request.body()
    try:
        body = json.loads(raw_body)
    except Exception:
        body = {}

    sid = session_id(body)
    stats = session_mod._stats.get(sid)

    if stats and stats.input_tokens > 0:
        real_tokens = stats.input_tokens
    else:
        real_tokens = estimate_tokens(body)
        log.debug("session %s: count_tokens no-stats estimate=%d", sid, real_tokens)

    avg_latency = stats.avg_latency_ms if stats else 0.0
    if compaction.should_nudge_compact(
        real_tokens,
        avg_latency,
        config.MAX_MODEL_LEN,
        config.COMPACT_TOKEN_RATIO,
        config.COMPACT_LATENCY_MS,
        kv_cache_usage=compaction._kv_cache_usage,
        kv_ratio=config.COMPACT_KV_RATIO,
    ):
        reported = compaction.compact_token_count(config.MAX_MODEL_LEN, config.COMPACT_NUDGE_RATIO)
        kv_trigger = compaction._kv_cache_usage >= config.COMPACT_KV_RATIO
        session_trigger = (
            real_tokens >= int(config.MAX_MODEL_LEN * config.COMPACT_TOKEN_RATIO)
            and avg_latency >= config.COMPACT_LATENCY_MS
        )
        trigger = (
            "kv_pressure" if kv_trigger and not session_trigger
            else "session" if session_trigger and not kv_trigger
            else "both"
        )
        log.info(
            "session %s: nudging compaction tokens=%d→%d avg_latency=%.0f ms kv=%.1f%% trigger=%s",
            sid, real_tokens, reported, avg_latency, compaction._kv_cache_usage * 100, trigger,
        )
    else:
        reported = real_tokens

    return Response(
        content=json.dumps({"input_tokens": reported}).encode(),
        status_code=200,
        media_type="application/json",
    )
