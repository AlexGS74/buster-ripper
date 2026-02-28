"""POST /v1/messages — Anthropic messages API proxy with normalization."""

import time
from typing import AsyncIterator

import httpx
from fastapi import APIRouter, Request
from fastapi.responses import Response, StreamingResponse

from .. import config
from ..dump import dump_turn
from ..normalize import normalize
from ..session import session_id, update_session_stats
from ..utils import extract_input_tokens, find_message_start, forward_headers, response_headers

router = APIRouter()


@router.post("/v1/messages")
async def proxy_messages(request: Request) -> Response:
    body = await request.json()
    body, changes = normalize(body)

    if changes and config.VERBOSE:
        import logging
        logging.getLogger("buster-ripper").info("normalized: %s", "; ".join(changes))

    if config.DUMP_DIR is not None:
        try:
            dump_turn(body)
        except Exception:
            import logging
            logging.getLogger("buster-ripper").exception("dump failed")

    sid = session_id(body)
    headers = forward_headers(request)
    stream = body.get("stream", False)

    if stream:
        t_start = time.monotonic()
        capture: dict = {"done": False}

        async def generate() -> AsyncIterator[bytes]:
            async with httpx.AsyncClient(timeout=None) as client:
                async with client.stream(
                    "POST",
                    f"{config.UPSTREAM}/v1/messages",
                    json=body,
                    headers=headers,
                ) as resp:
                    async for chunk in resp.aiter_bytes():
                        if not capture["done"]:
                            evt = find_message_start(chunk)
                            if evt is not None:
                                ttft_ms = (time.monotonic() - t_start) * 1000
                                input_tokens = (
                                    evt.get("message", {})
                                    .get("usage", {})
                                    .get("input_tokens", 0)
                                )
                                capture["done"] = True
                                update_session_stats(sid, input_tokens, ttft_ms)
                                if config.VERBOSE:
                                    import logging
                                    logging.getLogger("buster-ripper").info(
                                        "session %s: tokens=%d ttft=%.0f ms",
                                        sid, input_tokens, ttft_ms,
                                    )
                        yield chunk

        return StreamingResponse(
            generate(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )
    else:
        t_start = time.monotonic()
        async with httpx.AsyncClient(timeout=300) as client:
            resp = await client.post(
                f"{config.UPSTREAM}/v1/messages",
                json=body,
                headers=headers,
            )
        elapsed_ms = (time.monotonic() - t_start) * 1000
        input_tokens = extract_input_tokens(resp.content)
        if input_tokens is not None:
            update_session_stats(sid, input_tokens, elapsed_ms)
            if config.VERBOSE:
                import logging
                logging.getLogger("buster-ripper").info(
                    "session %s: tokens=%d latency=%.0f ms",
                    sid, input_tokens, elapsed_ms,
                )
        return Response(
            content=resp.content,
            status_code=resp.status_code,
            headers=response_headers(resp.headers),
        )
