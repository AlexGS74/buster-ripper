"""KV cache utilization polling and compaction nudge logic."""

import asyncio
import logging
import re

import httpx

from . import config

log = logging.getLogger("buster-ripper")

# Current server-wide KV cache utilization (EMA-smoothed, 0.0–1.0).
# Updated by _poll_kv_cache() background task.
_kv_cache_usage: float = 0.0

# Matches vLLM Prometheus metric lines, e.g.:
#   vllm:kv_cache_usage_perc{engine="0",...} 0.714
_KV_METRIC_RE = re.compile(r'^vllm:kv_cache_usage_perc\b[^\n]*\s+([\d.]+)', re.MULTILINE)

# EMA alpha for KV cache smoothing. vLLM reports per-interval averages that can
# bounce (75% → 0% → 23%) when the idle interval resets the counter. α=0.3
# damps the noise while still reacting within a few poll cycles to real pressure.
_KV_EMA_ALPHA: float = 0.3


async def poll_kv_cache() -> None:
    """Background task: poll vLLM /metrics every KV_POLL_INTERVAL seconds."""
    global _kv_cache_usage
    first = True
    while True:
        try:
            async with httpx.AsyncClient(timeout=5) as client:
                resp = await client.get(f"{config.UPSTREAM}/metrics")
            if resp.status_code == 200:
                m = _KV_METRIC_RE.search(resp.text)
                if m:
                    raw = float(m.group(1))
                    if first:
                        _kv_cache_usage = raw
                        first = False
                    else:
                        _kv_cache_usage = (1 - _KV_EMA_ALPHA) * _kv_cache_usage + _KV_EMA_ALPHA * raw
                    log.debug("kv_cache_usage=%.1f%% (raw=%.1f%%)", _kv_cache_usage * 100, raw * 100)
        except Exception:
            pass  # upstream may be starting up; ignore silently
        await asyncio.sleep(config.KV_POLL_INTERVAL)


def should_nudge_compact(
    input_tokens: int,
    avg_latency_ms: float,
    max_model_len: int,
    token_ratio: float,
    latency_threshold_ms: float,
    kv_cache_usage: float = 0.0,
    kv_ratio: float = 1.0,
) -> bool:
    """Return True when compaction should be nudged.

    Two independent triggers — either is sufficient:
      1. Per-session: token fill AND latency both exceed thresholds.
      2. Server-wide: KV cache utilization exceeds kv_ratio.
    """
    token_full = input_tokens >= int(max_model_len * token_ratio)
    latency_high = avg_latency_ms >= latency_threshold_ms
    per_session = token_full and latency_high
    kv_pressure = kv_cache_usage >= kv_ratio
    return per_session or kv_pressure


def compact_token_count(max_model_len: int, nudge_ratio: float) -> int:
    """Return the inflated token count that triggers Claude Code auto-compaction."""
    return int(max_model_len * nudge_ratio)
