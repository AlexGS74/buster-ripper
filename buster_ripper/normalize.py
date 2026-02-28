"""Request normalization for KV cache stability."""

import re
from . import config

# Matches the framework-injected currentDate line, e.g.:
#   Today's date is 2026-02-21.\n
_DATE_RE = re.compile(r"Today's date is \d{4}-\d{2}-\d{2}\.\n?")

# Matches the per-request billing nonce block injected by Claude Code, e.g.:
#   x-anthropic-billing-header: cc_version=2.1.50.f15; cc_entrypoint=cli; cch=27acd;
# This changes on every request and lives at position 0 of the system prompt,
# busting the entire KV cache. Meaningless to a local model — always stripped.
_BILLING_RE = re.compile(r"x-anthropic-billing-header:[^\n]*\n?")


def normalize(body: dict) -> tuple[dict, list[str]]:
    """Apply cache-stabilizing normalizations. Returns (body, list of changes)."""
    changes: list[str] = []

    # 0. Strip per-request billing nonce from system prompt.
    system = body.get("system")
    if isinstance(system, list):
        filtered = []
        for block in system:
            if isinstance(block, dict) and isinstance(block.get("text"), str):
                cleaned = _BILLING_RE.sub("", block["text"]).strip()
                if cleaned:
                    if cleaned != block["text"].strip():
                        changes.append("stripped billing-header from system block")
                    block = {**block, "text": cleaned}
                    filtered.append(block)
                else:
                    changes.append("stripped billing-header system block")
            else:
                filtered.append(block)
        body["system"] = filtered
    elif isinstance(system, str) and _BILLING_RE.search(system):
        body["system"] = _BILLING_RE.sub("", system).strip()
        changes.append("stripped billing-header from system string")

    # 1. Sort tools by name; drop empty tools array.
    if isinstance(body.get("tools"), list):
        if not body["tools"]:
            del body["tools"]
            changes.append("dropped empty tools array")
        else:
            original_names = [t.get("name", "") for t in body["tools"]]
            body["tools"] = sorted(body["tools"], key=lambda t: t.get("name", ""))
            sorted_names = [t.get("name", "") for t in body["tools"]]
            if original_names != sorted_names:
                changes.append(
                    f"tools reordered: [{', '.join(original_names[:4])}{'...' if len(original_names) > 4 else ''}]"
                    f" → [{', '.join(sorted_names[:4])}{'...' if len(sorted_names) > 4 else ''}]"
                )

    # 2. Strip cache_control fields from messages and system blocks.
    if isinstance(body.get("system"), list):
        stripped = 0
        for block in body["system"]:
            if isinstance(block, dict) and "cache_control" in block:
                del block["cache_control"]
                stripped += 1
        if stripped:
            changes.append(f"stripped cache_control from {stripped} system block(s)")

    if isinstance(body.get("messages"), list):
        stripped = 0
        for msg in body["messages"]:
            content = msg.get("content")
            if isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and "cache_control" in block:
                        del block["cache_control"]
                        stripped += 1
            if "cache_control" in msg:
                del msg["cache_control"]
                stripped += 1
        if stripped:
            changes.append(f"stripped cache_control from {stripped} message field(s)")

    # 3. Strip currentDate injection from user message content.
    if config.STRIP_DATE and isinstance(body.get("messages"), list):
        for msg in body["messages"]:
            if msg.get("role") != "user":
                continue
            content = msg.get("content", "")
            if isinstance(content, str) and _DATE_RE.search(content):
                msg["content"] = _DATE_RE.sub("", content)
                changes.append("stripped currentDate from user message")
            elif isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and isinstance(block.get("text"), str):
                        if _DATE_RE.search(block["text"]):
                            block["text"] = _DATE_RE.sub("", block["text"])
                            changes.append("stripped currentDate from user content block")

    return body, changes
