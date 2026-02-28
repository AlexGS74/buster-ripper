"""HTTP helper utilities shared across route handlers."""

import json
from typing import Optional

import httpx
from fastapi import Request

_HOP_BY_HOP = frozenset([
    "connection", "keep-alive", "proxy-authenticate", "proxy-authorization",
    "te", "trailers", "transfer-encoding", "upgrade",
])


def forward_headers(request: Request) -> dict[str, str]:
    """Build headers to send upstream, dropping hop-by-hop and host."""
    return {
        k: v
        for k, v in request.headers.items()
        if k.lower() not in ("host", "content-length", *_HOP_BY_HOP)
    }


def response_headers(upstream_headers: httpx.Headers) -> dict[str, str]:
    """Strip hop-by-hop headers from upstream response."""
    return {
        k: v
        for k, v in upstream_headers.items()
        if k.lower() not in _HOP_BY_HOP
    }


def extract_input_tokens(response_body: bytes) -> Optional[int]:
    """Extract input_tokens from a non-streaming JSON response body."""
    try:
        return json.loads(response_body).get("usage", {}).get("input_tokens")
    except Exception:
        return None


def find_message_start(chunk: bytes) -> Optional[dict]:
    """Scan an SSE chunk for a 'message_start' event. Returns parsed dict or None.

    The message_start event carries usage.input_tokens and is always the first
    SSE event in an Anthropic streaming response — it arrives in the first
    network chunk, making it a reliable TTFT signal.
    """
    text = chunk.decode(errors="replace")
    for line in text.splitlines():
        if not line.startswith("data: "):
            continue
        payload = line[6:].strip()
        if payload == "[DONE]":
            continue
        try:
            evt = json.loads(payload)
            if evt.get("type") == "message_start":
                return evt
        except Exception:
            pass
    return None
