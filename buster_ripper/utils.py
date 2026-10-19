"""HTTP helper utilities shared across route handlers."""

import json
import logging
from collections.abc import Mapping
from typing import Optional

from multidict import CIMultiDict

from . import config

log = logging.getLogger("buster-ripper")


_HOP_BY_HOP = frozenset([
    "connection", "keep-alive", "proxy-authenticate", "proxy-authorization",
    "te", "trailers", "transfer-encoding", "upgrade",
])


def forward_headers(headers: dict) -> dict:
    """Build headers to send upstream, dropping hop-by-hop and host."""
    return {
        k: v
        for k, v in headers.items()
        if k.lower() not in ("host", "content-length", "connection", "transfer-encoding", *_HOP_BY_HOP)
    }


def response_headers(upstream_headers: Mapping[str, str]) -> CIMultiDict:
    """Build headers to send downstream, dropping hop-by-hop and content-length.

    One pair per *occurrence*: the client hands the upstream's head over as a
    per-occurrence mapping and ``web.Response`` writes one field-line per entry it is
    given, so a name the upstream sent twice (`Set-Cookie: a=1` then `b=2`, two `Vary`
    values) reaches the client twice. Folding into a plain ``dict`` here kept the last
    occurrence and destroyed the rest — and that was not base parity either, because
    base's client merged occurrences into one comma-joined value, so no value was ever
    lost on the way through.

    content-length is dropped alongside the hop-by-hop names because aiohttp
    re-derives it from the body we actually send: an upstream that answers with
    both Transfer-Encoding: chunked and a Content-Length would otherwise have its
    (lying) length relayed, since Response._start only derives a length when the
    header is absent.

    Kept names include ``Content-Encoding``, and this filter does not reconcile it with
    the body it travels beside: the package's client defaults to ``auto_decompress=True``
    and never overrides it, so the upstream's encoding label is relayed over bytes this
    proxy already decoded. That posture is base parity and stays as it is until the
    design pass rules on the alternative shapes — see ``roadmap.md`` §RM-008.
    """
    return CIMultiDict(
        (k, v)
        for k, v in upstream_headers.items()
        if k.lower() not in ("content-length", *_HOP_BY_HOP)
    )


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
        thinking reasoning text</think>actual answer
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


class ContextOverflowError(Exception):
    """Raised when SGLang reports context overflow error."""

    def __init__(self, message: str, input_tokens: int, ctx_limit: int):
        super().__init__(message)
        self.input_tokens = input_tokens
        self.ctx_limit = ctx_limit


def parse_context_overflow_error(error_text: str) -> Optional[ContextOverflowError]:
    """Parse SGLang context overflow error and extract token counts.

    Returns ContextOverflowError if pattern matches, None otherwise.
    """
    match = config.CONTEXT_OVERFLOW_RE.search(error_text)
    if not match:
        return None

    input_tokens = int(match.group(1))
    ctx_limit = int(match.group(2))

    # Also try to extract from CTX_LIMIT_RE as fallback
    ctx_match = config.CTX_LIMIT_RE.search(error_text)
    if ctx_match:
        ctx_limit = int(ctx_match.group(1))

    return ContextOverflowError(error_text, input_tokens, ctx_limit)


def calculate_new_max_tokens(ctx_limit: int, input_tokens: int) -> int:
    """Calculate reduced max_completion_tokens after context overflow.

    Formula: max(MIN_COMPLETION_TOKENS, ctx_limit - input_tokens - 100)
    The -100 provides a safety margin.
    """
    return max(config.MIN_COMPLETION_TOKENS, ctx_limit - input_tokens - 100)
