"""Backend detection and state management.

Detects backend type (SGLang vs vLLM) on startup and caches result.
Detection priority: CLI flag > auto-detect > fallback.
A probe without a definitive answer stays uncached, so a retry can classify the
backend once the upstream answers.
"""

import asyncio
import logging
from typing import Optional

import aiohttp

log = logging.getLogger("buster-ripper")

# Module-level state
_backend_type: Optional[str] = None
_backend_state_cached: bool = False

# Seconds between re-probes while the upstream has not answered definitively.
_RETRY_INTERVAL: float = 5.0


async def detect_backend(
    session: aiohttp.ClientSession,
    backend_url: str,
    cli_override: Optional[str] = None,
) -> str:
    """Detect backend type with CLI override, auto-detect, or fallback.

    Priority chain:
    1. CLI flag (--backend-type) - highest priority
    2. Auto-detect via /v1/models probe
    3. Fallback to "unknown" - lowest priority (probe failure, unrecognized backend)

    A failed or empty probe returns "unknown" without caching it: the caller may
    keep that value for now, and a later probe can still classify the backend.

    Args:
        session: Shared aiohttp ClientSession
        backend_url: Backend URL (e.g., "http://localhost:30000")
        cli_override: CLI flag override if provided

    Returns:
        Backend type: "sglang", "vllm", or "unknown"
    """
    global _backend_type, _backend_state_cached

    # Priority 1: CLI override takes precedence
    if cli_override:
        _backend_type = cli_override
        _backend_state_cached = True
        log.info("Backend type set via CLI override: %s", _backend_type)
        return _backend_type

    # Priority 2: Return cached result if available
    if _backend_state_cached:
        return _backend_type

    # Priority 3: Auto-detect via /v1/models probe
    try:
        async with session.get(
            f"{backend_url}/v1/models", timeout=aiohttp.ClientTimeout(total=5)
        ) as resp:
            resp.raise_for_status()
            data = await resp.json()
        detected = _classify_models(data)
    except Exception as e:
        # Priority 4: Fallback to unknown on probe failure
        log.warning("Backend probe failed: %s, defaulting to 'unknown'", e)
        detected = None

    _backend_type = detected or "unknown"
    if detected is None:
        return _backend_type

    _backend_state_cached = True
    log.info("Backend type detected: %s", _backend_type)
    return _backend_type


def _classify_models(data: dict) -> Optional[str]:
    """Classify the first model; None when the answer is not definitive."""
    models = data.get("data", [])
    if not models:
        log.warning("No models returned from /v1/models, retrying later")
        return None
    owned_by = models[0].get("owned_by", "").lower()
    if "sglang" in owned_by:
        return "sglang"
    if "vllm" in owned_by:
        return "vllm"
    log.warning("Unknown backend detected (owned_by=%s), treating as 'unknown'", owned_by)
    return "unknown"


def is_backend_state_cached() -> bool:
    """Whether a definitive backend type has been cached."""
    return _backend_state_cached


async def _retry_detect_backend(
    session: aiohttp.ClientSession, backend_url: str
) -> None:
    """Background task: re-probe until the upstream answers definitively."""
    while not _backend_state_cached:
        await asyncio.sleep(_RETRY_INTERVAL)
        await detect_backend(session, backend_url)


def get_backend_state() -> Optional[str]:
    """Get cached backend type.

    Returns:
        Backend type if detected, None if not yet detected
    """
    return _backend_type
