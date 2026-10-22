"""The post-commit SSE relay for /v1/messages."""

import time

from .. import config
from ..session import update_session_stats
from ..utils import find_message_start


async def _relay_after_commit(request, resp, response, state) -> None:
    """Relay the upstream body into a response whose head is already committed.

    A failure cannot be answered with another response — a second head would land in
    the first body — so it logs and returns; catches Exception, not ClientError, while
    CancelledError (a BaseException) still propagates."""
    try:
        async for chunk in resp.content.iter_any():
            _note_message_start(request, chunk, state)
            await response.write(chunk)
        await response.write_eof()
    except Exception as exc:
        # Position, never cause: one arm covers upstream, timeout and downstream loss.
        request.app["log"].error(
            "Post-commit relay error (%s): %s", type(exc).__name__, exc
        )


def _note_message_start(request, chunk: bytes, state: dict) -> None:
    """Record usage and time-to-first-token from the first message_start event."""
    if state["done"]:
        return
    evt = find_message_start(chunk)
    if evt is None:
        return
    message = evt.get("message")
    usage = message.get("usage") if isinstance(message, dict) else None
    input_tokens = usage.get("input_tokens", 0) if isinstance(usage, dict) else 0
    state["done"] = True
    ttft_ms = (time.monotonic() - state["started"]) * 1000
    update_session_stats(state["sid"], input_tokens, ttft_ms)
    if config.VERBOSE:
        request.app["log"].info(
            "session %s: tokens=%d ttft=%.0f ms",
            state["sid"], input_tokens, ttft_ms,
        )
