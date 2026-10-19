"""Image handling for API relay with Base64 passthrough and caching.

Pure functions for image detection, caching, and retrieval, plus the periodic
sweep that expires cached images while the process is idle.
Module-level state for image cache (like session.py pattern).
"""

import asyncio
import hashlib
import json
import logging
import time
from typing import Optional

from . import config

log = logging.getLogger("buster-ripper")


# Retention stamps come from a non-decreasing source, injected through this name.
# A wall clock that steps backwards leaves the entry stamped before the step holding
# the largest stamp: it sorts last in every eviction slice, so nothing removes it,
# and `now - first_seen` goes negative, so the TTL comparator can never expire it
# either — one entry immune to both advertised bounds. `time.monotonic` is what the
# package already times with (routes/messages.py).
_now = time.monotonic


# Module-level image cache:
# session_id -> {image_key -> (source_dict, first_seen, last_touched)}
# `first_seen` is what the TTL ages against (absolute since first cached);
# `last_touched` is what eviction orders on (least recently touched).
_image_cache: dict[str, dict[str, tuple[dict, float, float]]] = {}


def _has_image_in_blocks(content: list) -> bool:
    """Check if content list contains an image block."""
    for block in content:
        if isinstance(block, dict) and block.get("type") == "image":
            return True
    return False


def _user_content(msg) -> Optional[list]:
    """The list content of a user message, or None for anything else."""
    if not isinstance(msg, dict) or msg.get("role") != "user":
        return None

    content = msg.get("content")
    return content if isinstance(content, list) else None


def _scoped_contents(messages) -> list[tuple[dict, list]]:
    """The (message, content) pairs this flag acts on — one scope, gate and writer.

    A `messages` value that is not a list holds no user messages. This flag
    transforms image blocks, it does not validate a request body: a body shaped
    like that gets forwarded as it arrived, and answers its own 4xx upstream, which
    is the status class the flag-off path already produces for it.
    """
    if not isinstance(messages, list):
        return []
    return [(msg, content) for msg in messages if (content := _user_content(msg)) is not None]


def has_images(body: dict) -> bool:
    """Detect images in any user message content.

    Scans every user message, because that is the scope `strip_and_cache_images`
    mutates. A detector that looked at only one message would gate the writer on
    what a later turn happens to contain, and a historical image would be forwarded
    as base64 whenever the latest user message is text-only.
    """
    return any(_has_image_in_blocks(content) for _, content in _scoped_contents(body.get("messages")))


def _cleanup_expired_images(session_id: str) -> None:
    """Remove expired images from a session's cache.

    Expiry only, and it never touches the session's own key in `_image_cache`: the
    read path calls this and then indexes into that mapping.
    """
    if session_id not in _image_cache:
        return

    cache = _image_cache[session_id]
    now = _now()
    expired_keys = [
        key for key, (_, first_seen, _) in cache.items()
        if now - first_seen > config.IMAGE_CACHE_TTL
    ]
    for key in expired_keys:
        del cache[key]


def _evict_by_lru(session_id: str) -> None:
    """Evict least-recently-touched entries when cache exceeds max size."""
    if session_id not in _image_cache:
        return

    cache = _image_cache[session_id]
    if len(cache) <= config.IMAGE_CACHE_MAX_SIZE:
        return

    # Slot [2] is `last_touched`. Sorting the age slot instead keeps every size
    # assertion green while turning eviction into FIFO.
    sorted_items = sorted(cache.items(), key=lambda x: x[1][2])
    excess = len(cache) - config.IMAGE_CACHE_MAX_SIZE
    for key, _ in sorted_items[:excess]:
        del cache[key]


def _image_key(source: dict) -> str:
    """The cache key of one image source: content-derived and order-independent.

    `source` is the whole input — not the image block, whose `cache_control` field
    one call site has already stripped and the other has not, which would make a
    block-level key differ per call site for one unchanged image. The canonical
    serialization is what makes two equal sources built in different key order agree,
    and the slice comes off the digest, never off the serialization: realistic base64
    sources share their first characters, so a truncated serialization would hand
    every image in a session the same key.
    """
    canonical = json.dumps(source, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return f"img_{digest[:12]}"


def _cache_source(session_id: str, image_key: str, source: dict) -> None:
    """Store one source under its key, refreshing recency but never age."""
    cache = _image_cache[session_id]
    now = _now()
    existing = cache.get(image_key)
    # A write onto an entry that is present but already expired carries its
    # `first_seen` forward: `IMAGE_CACHE_TTL` bounds absolute age, so re-sending an
    # image must not buy its payload another TTL. This call's own tail sweep is what
    # drops it, and the key is content-derived, so the placeholder text is unchanged.
    first_seen = existing[1] if existing is not None else now
    cache[image_key] = (source, first_seen, now)


def _replace_image_with_placeholder(block: dict, session_id: str) -> dict:
    """Replace image block with its content-addressed placeholder and cache the source."""
    source = block.get("source", {})
    image_key = _image_key(source)

    _cache_source(session_id, image_key, source)

    # Evict oldest entries if over limit
    _evict_by_lru(session_id)

    # Return placeholder block
    return {
        "type": "text",
        "text": f"[image:{image_key}]",
    }


def _process_content_block(block: dict, session_id: str) -> dict:
    """Process a single content block, replacing images with placeholders."""
    if isinstance(block, dict) and block.get("type") == "image":
        return _replace_image_with_placeholder(block, session_id)
    return block


def _process_user_content(content: list, session_id: str) -> list:
    """Process user message content, replacing images with placeholders."""
    return [_process_content_block(block, session_id) for block in content]


def strip_and_cache_images(body: dict, session_id: str) -> dict:
    """Strip images from user message content and cache them.

    Each image block becomes a text block holding `[image:<key>]`, where the key is
    derived from that block's `source`, so one source keeps the same marker text on
    every request that carries it. That byte-for-byte stability in the forwarded
    body is the marker's whole contract: the cached source has no reader on the
    request path, and the placeholder is not a reference that restores anything.

    Images are stored in module-level cache with TTL-based expiration and LRU
    eviction, swept across every session once this request's writes are done.

    Args:
        body: Request body dict with messages
        session_id: Session identifier for cache isolation

    Returns:
        Modified body dict with images replaced by placeholders
    """
    if session_id not in _image_cache:
        _image_cache[session_id] = {}

    messages = body.get("messages")
    if not isinstance(messages, list) or not messages:
        return body

    for msg, content in _scoped_contents(messages):
        msg["content"] = _process_user_content(content, session_id)

    # Swept after the writes, never between the ensure above and them: no guarded
    # call site has a try to absorb a deleted session, so a delete placed there
    # would answer an otherwise-forwardable request with an HTTP 500.
    cleanup_expired_images()

    return body


def get_cached_image(session_id: str, image_key: str) -> Optional[dict]:
    """Retrieve a cached image by session and key.

    Args:
        session_id: Session identifier
        image_key: Image cache key

    Returns:
        Image source dict if found and not expired, None otherwise
    """
    if session_id not in _image_cache:
        return None

    # Cleanup expired entries first
    _cleanup_expired_images(session_id)

    if image_key not in _image_cache[session_id]:
        return None

    source, _, _ = _image_cache[session_id][image_key]
    return source


def make_data_url(data: str, media_type: str) -> str:
    """Convert base64 data to a data: URL.

    Handles pre-prefixed data URLs by stripping the prefix.
    """
    if "base64" in data:
        data = data.split("base64")[-1]
        if data.startswith(","):
            data = data[1:]
    return f"data:{media_type};base64,{data}"


def cleanup_expired_images() -> None:
    """Prune expired images from every session and reclaim the sessions that emptied.

    This is the only place a session's own key leaves `_image_cache`. Those outer
    keys are derived from client-sent content, so without this drop a proxy kept
    receiving new conversations accumulates one permanent entry per session for the
    life of the process, holding that request's own parsed `source` mappings.
    """
    for session_id in list(_image_cache.keys()):
        _cleanup_expired_images(session_id)
        if not _image_cache[session_id]:
            del _image_cache[session_id]


def clear_session_cache(session_id: str) -> None:
    """Clear all cached images for a session."""
    if session_id in _image_cache:
        del _image_cache[session_id]


async def poll_expiry() -> None:
    """Expire cached images every IMAGE_CACHE_TTL seconds.

    The write path only sweeps while image-bearing requests keep arriving, so an
    idle proxy would hold a source past its advertised bound. Failures are logged
    and the loop continues; cancellation (from on_cleanup) lands at the sleep
    await, whose interval floors at one second.
    """
    while True:
        try:
            cleanup_expired_images()
        except Exception:
            log.exception("image: periodic expiry failed")
        await asyncio.sleep(max(config.IMAGE_CACHE_TTL, 1))
