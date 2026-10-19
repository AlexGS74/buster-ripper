"""Request normalization for KV cache stability."""

import re
from . import config
from . import image as image_handler

# Matches the framework-injected currentDate line, e.g.:
#   Today's date is 2026-02-21.\n
_DATE_RE = re.compile(r"Today's date is \d{4}-\d{2}-\d{2}\.\n?")

# Matches the per-request billing nonce block injected by Claude Code, e.g.:
#   x-anthropic-billing-header: cc_version=2.1.50.f15; cc_entrypoint=cli; cch=27acd;
# This changes on every request and lives at position 0 of the system prompt,
# busting the entire KV cache. Meaningless to a local model — always stripped.
_BILLING_RE = re.compile(r"x-anthropic-billing-header:[^\n]*\n?")


def _strip_billing_from_string(system: str) -> tuple[str, list[str]]:
    """Strip billing header from string system prompt."""
    if _BILLING_RE.search(system):
        return _BILLING_RE.sub("", system).strip(), ["stripped billing-header from system string"]
    return system, []


def _strip_billing_from_blocks(system: list) -> tuple[list, list[str]]:
    """Strip billing header from list of system blocks."""
    filtered = []
    changes = []

    for block in system:
        # Skip non-text blocks
        if not (isinstance(block, dict) and isinstance(block.get("text"), str)):
            filtered.append(block)
            continue

        cleaned = _BILLING_RE.sub("", block["text"]).strip()

        # Block became empty after stripping
        if not cleaned:
            changes.append("stripped billing-header system block")
            continue

        if cleaned != block["text"].strip():
            changes.append("stripped billing-header from system block")
        filtered.append({**block, "text": cleaned})

    return filtered, changes


def _normalize_system(body: dict) -> list[str]:
    """Strip billing nonce from system prompt."""
    system = body.get("system")

    if isinstance(system, list):
        body["system"], changes = _strip_billing_from_blocks(system)
        return changes
    elif isinstance(system, str):
        body["system"], changes = _strip_billing_from_string(system)
        return changes

    return []


def strip_billing_nonce(body: dict) -> list[str]:
    """Remove the per-request billing nonce from ``system``, and touch nothing else.

    Public because one caller needs exactly this step and no other: the stats lookup on
    ``/v1/messages/count_tokens`` keys on ``session_id()``, which hashes ``system`` alone
    (``session.py:93-98``), and this is the only step below that writes its text. Running
    the whole of ``normalize()`` there buys nothing and costs ``CR4-F-01`` — the ``tools``
    and ``messages`` steps assume every element is a mapping, so a body base answers an
    estimate for would answer a 500 instead.
    """
    return _normalize_system(body)


def _normalize_tools(body: dict) -> list[str]:
    """Sort tools by name and drop empty tools array."""
    tools = body.get("tools")

    if not isinstance(tools, list):
        return []

    if not tools:
        del body["tools"]
        return ["dropped empty tools array"]

    original_names = [t.get("name", "") for t in tools]
    body["tools"] = sorted(tools, key=lambda t: t.get("name", ""))
    sorted_names = [t.get("name", "") for t in body["tools"]]

    if original_names == sorted_names:
        return []

    orig_preview = ', '.join(original_names[:4])
    sorted_preview = ', '.join(sorted_names[:4])
    orig_suffix = '...' if len(original_names) > 4 else ''
    sorted_suffix = '...' if len(sorted_names) > 4 else ''

    return [f"tools reordered: [{orig_preview}{orig_suffix}] → [{sorted_preview}{sorted_suffix}]"]


def _strip_cache_control_from_system(system: list) -> int:
    """Strip cache_control fields from system blocks."""
    stripped = 0
    for block in system:
        if isinstance(block, dict) and "cache_control" in block:
            del block["cache_control"]
            stripped += 1
    return stripped


def _strip_cache_control_from_content(content) -> int:
    """Strip cache_control from content blocks. Returns count stripped."""
    if not isinstance(content, list):
        return 0

    stripped = 0
    for block in content:
        if isinstance(block, dict) and "cache_control" in block:
            del block["cache_control"]
            stripped += 1
    return stripped


def _strip_cache_control_from_messages(messages: list) -> int:
    """Strip cache_control fields from messages."""
    stripped = 0
    for msg in messages:
        stripped += _strip_cache_control_from_content(msg.get("content"))
        if "cache_control" in msg:
            del msg["cache_control"]
            stripped += 1
    return stripped


def _normalize_cache_control(body: dict) -> list[str]:
    """Strip cache_control fields from messages and system blocks."""
    changes = []

    if isinstance(body.get("system"), list):
        stripped = _strip_cache_control_from_system(body["system"])
        if stripped:
            changes.append(f"stripped cache_control from {stripped} system block(s)")

    if isinstance(body.get("messages"), list):
        stripped = _strip_cache_control_from_messages(body["messages"])
        if stripped:
            changes.append(f"stripped cache_control from {stripped} message field(s)")

    return changes


def _strip_date_from_content(content) -> tuple[str | list, list[str]]:
    """Strip currentDate injection from content. Preserves original type."""
    if isinstance(content, str):
        if _DATE_RE.search(content):
            return _DATE_RE.sub("", content), ["stripped currentDate from user message"]
        return content, []

    if not isinstance(content, list):
        return content, []

    new_blocks = []
    changes = []
    for block in content:
        # Skip non-text blocks
        if not (isinstance(block, dict) and isinstance(block.get("text"), str)):
            new_blocks.append(block)
            continue

        if _DATE_RE.search(block["text"]):
            new_blocks.append({**block, "text": _DATE_RE.sub("", block["text"])})
            changes.append("stripped currentDate from user content block")
        else:
            new_blocks.append(block)

    return new_blocks, changes


def _normalize_date(body: dict) -> list[str]:
    """Strip currentDate injection from user messages.

    Only applies when:
    1. STRIP_DATE config is enabled, AND
    2. Backend is detected as SGLang

    Unknown backends get no smarts applied.
    """
    if not config.STRIP_DATE:
        return []

    # Only apply date stripping for SGLang backends
    from . import backend
    backend_type = backend.get_backend_state()
    if backend_type != "sglang":
        return []

    messages = body.get("messages")
    if not isinstance(messages, list):
        return []

    changes = []
    for msg in messages:
        if msg.get("role") != "user":
            continue

        new_content, date_changes = _strip_date_from_content(msg.get("content", ""))
        msg["content"] = new_content
        changes.extend(date_changes)

    return changes


def normalize(body: dict, session_id: str = "") -> tuple[dict, list[str]]:
    """Apply cache-stabilizing normalizations. Returns (body, list of changes).

    If session_id is provided and STRIP_IMAGES is enabled, images will be
    stripped and cached. Otherwise, images are left untouched for later handling.
    """
    changes: list[str] = []

    # 0. Image handling - strip and cache images if enabled AND session_id provided
    # (session_id is needed for cache isolation; empty string skips image handling)
    if config.STRIP_IMAGES and session_id and image_handler.has_images(body):
        image_handler.strip_and_cache_images(body, session_id)
        changes.append("stripped and cached images")

    # 1. Strip per-request billing nonce from system prompt
    changes.extend(_normalize_system(body))

    # 2. Sort tools by name; drop empty tools array
    changes.extend(_normalize_tools(body))

    # 3. Strip cache_control fields from messages and system blocks
    changes.extend(_normalize_cache_control(body))

    # 4. Strip currentDate injection from user message content
    changes.extend(_normalize_date(body))

    return body, changes
