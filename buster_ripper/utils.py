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


def split_thinking(content: str) -> tuple[str, str]:
    """Parse a <think>...</think> block out of raw content.

    Some models (e.g. GLM-4.7 with enable_thinking=true) return the full
    generation in the content field:
        <think>reasoning text</think>actual answer
    vLLM may or may not split this into reasoning_content + content depending
    on the version and model. This function normalizes it so callers always get
    a clean (thinking, answer) pair regardless of what vLLM did.

    Returns (thinking, answer):
      - thinking: text inside the first <think>...</think> block, or ""
      - answer:   everything after </think>, stripped of leading whitespace,
                  or the full content unchanged if no <think> block is found
    """
    open_tag = content.find("<think>")
    close_tag = content.find("</think>")
    if open_tag == -1 or close_tag == -1 or close_tag < open_tag:
        return "", content
    thinking = content[open_tag + len("<think>"):close_tag]
    answer = content[close_tag + len("</think>"):].lstrip("\n")
    return thinking, answer


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
