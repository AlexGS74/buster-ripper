"""POST /v1/messages — Anthropic messages API proxy with normalization."""

import json
import time

import aiohttp
from aiohttp import web

from .. import backend
from .. import config
from .. import image as image_handler
from ..dump import dump_turn
from ..normalize import normalize
from ..session import session_id, update_session_stats
from ..utils import (
    ContextOverflowError,
    calculate_new_max_tokens,
    extract_input_tokens,
    forward_headers,
    parse_context_overflow_error,
    response_headers,
)
from .stream_relay import _relay_after_commit


async def handle_messages(request: web.Request) -> web.StreamResponse:
    """Proxy /v1/messages to upstream with normalization."""
    try:
        body = await request.json()
    except (json.JSONDecodeError, UnicodeDecodeError, LookupError):
        # LookupError, not UnicodeDecodeError, is a client naming a codec Python
        # lacks (web_request.py:689 hands the charset to bytes.decode()).
        return web.json_response({"error": "Invalid JSON"}, status=400)

    sid = _prepare_body(request, body)

    if config.DUMP_DIR is not None:
        try:
            dump_turn(body)
        except Exception:
            request.app["log"].exception("dump failed")

    # Build headers, removing hop-by-hop (case-insensitive)
    headers = forward_headers(request.headers)

    stream = body.get("stream", False)

    if stream:
        return await handle_streaming(request, body, headers, sid)
    else:
        return await handle_non_streaming(request, body, headers, sid)


def _inject_preserved_thinking(body: dict) -> None:
    """Put the flag's kwargs pair on the body; a client-sent non-mapping is replaced, not skipped."""
    kwargs = body.get("chat_template_kwargs")
    if not isinstance(kwargs, dict):
        kwargs = {}
        body["chat_template_kwargs"] = kwargs
    kwargs.setdefault("enable_thinking", True)
    kwargs.setdefault("clear_thinking", False)


def _prepare_body(request: web.Request, body: dict) -> str:
    """Normalize the body in place and return its session id."""
    # Normalize FIRST to strip billing nonce, THEN compute session_id: the nonce
    # changes every request, so the id is only stable after the strip.
    body, changes = normalize(body, session_id="")
    sid = session_id(body)

    # Handle images AFTER computing session_id (for proper cache isolation)
    if config.STRIP_IMAGES and image_handler.has_images(body):
        image_handler.strip_and_cache_images(body, sid)
        changes.append("stripped and cached images")

    # SGLang v0.5.9's /v1/messages discards chat_template_kwargs, so injecting
    # the pair there would advertise an effect that never happens.
    if config.PRESERVED_THINKING and backend.get_backend_state() != "sglang":
        _inject_preserved_thinking(body)
        if config.VERBOSE:
            request.app["log"].info(
                "preserved-thinking: injected enable_thinking=true clear_thinking=false"
            )

    if changes and config.VERBOSE:
        request.app["log"].info("normalized: %s", "; ".join(changes))

    return sid


async def handle_streaming(
    request: web.Request, body: dict, headers: dict, sid: str
) -> web.StreamResponse:
    """Handle streaming for /v1/messages with context overflow retry.

    Retries on SGLang only, detecting overflow from the actual response."""
    backend_type = backend.get_backend_state()

    # Non-SGLang backends: no retry
    if backend_type != "sglang":
        return await _stream_single_request(request, body, headers, sid)

    # SGLang: retry on context overflow, detect from actual response
    max_retries = 2
    current_body = dict(body)

    for attempt in range(max_retries + 1):
        result = await _stream_single_request(request, current_body, headers, sid)

        # Check if response indicates context overflow (error response)
        overflow_err = _stream_overflow(result)

        if overflow_err and attempt < max_retries:
            new_max = calculate_new_max_tokens(overflow_err.ctx_limit, overflow_err.input_tokens)
            current_body["max_tokens"] = new_max
            request.app["log"].info(
                "Context overflow retry %d/%d: reducing max_tokens to %d",
                attempt + 1, max_retries, new_max,
            )
            continue

        # Max retries reached or non-overflow error
        if overflow_err:
            return web.json_response(
                {"error": f"Context overflow after {max_retries} retries"},
                status=400,
            )

        return result

    # Unreachable but needed for type checker
    return await _stream_single_request(request, current_body, headers, sid)


def _stream_overflow(result) -> ContextOverflowError | None:
    """The overflow a relayed error response carries, or None."""
    if not (isinstance(result, web.Response) and result.status >= 400):
        return None
    # The probe reads the bytes this arm relays: `.text` would decode under the
    # upstream's charset, which may name a codec Python lacks.
    error_text = result.body.decode(errors="replace")
    return parse_context_overflow_error(error_text)


async def _stream_single_request(
    request: web.Request, body: dict, headers: dict, sid: str
) -> web.StreamResponse:
    """Execute a single streaming request without retry logic."""
    state = {"done": False, "sid": sid, "started": time.monotonic()}
    session = request.app["session"]

    try:
        # No deadline on the relay: base's stream path ran without one
        # (buster_ripper_legacy.py:664), and a request-local ClientTimeout
        # replaces the session's rather than merging into it.
        async with session.post(
            f"{config.UPSTREAM}/v1/messages", headers=headers,
            data=_request_payload(body, await request.read(), headers),
            timeout=aiohttp.ClientTimeout(total=None), allow_redirects=False,
        ) as resp:
            # A redirect or an error is the upstream's answer, not a stream to relay.
            if resp.status >= 300:
                return await _upstream_answer(resp)

            # Only prepare response after we know upstream succeeded
            response = web.StreamResponse(
                status=200,
                headers={
                    "Content-Type": "text/event-stream",
                    "Cache-Control": "no-cache",
                    "X-Accel-Buffering": "no",
                    "Connection": "keep-alive",
                },
            )
            await response.prepare(request)
            await _relay_after_commit(request, resp, response, state)
            return response

    except (aiohttp.ClientError, TimeoutError) as e:
        request.app["log"].error("Upstream error: %s", e)
        return web.json_response({"error": "Upstream request failed"}, status=502)


async def _upstream_answer(resp: aiohttp.ClientResponse) -> web.Response:
    """Relay the upstream's own answer, with its bytes, status and headers."""
    return web.Response(
        body=await resp.read(),
        status=resp.status,
        headers=response_headers(resp.headers),
    )


def _request_payload(body: dict, raw: bytes, headers: dict) -> bytes:
    """The client's own bytes unless normalization changed the body; a changed body is
    re-encoded and relabelled, because aiohttp takes a payload's own Content-Type only
    when the header is absent."""
    try:
        if json.loads(raw) == body:
            return raw
    except (json.JSONDecodeError, UnicodeDecodeError, LookupError):
        pass
    headers["Content-Type"] = "application/json"
    return json.dumps(body).encode()


async def handle_non_streaming(
    request: web.Request, body: dict, headers: dict, sid: str
) -> web.Response:
    """Handle non-streaming response for /v1/messages with context overflow retry.

    Only retries on SGLang backends. vLLM and unknown backends skip retry logic.
    """
    backend_type = backend.get_backend_state()

    # Non-SGLang backends: no retry
    if backend_type != "sglang":
        return await _non_stream_single_request(request, body, headers, sid)

    # SGLang: retry on context overflow
    max_retries = 2
    current_body = dict(body)

    for attempt in range(max_retries + 1):
        try:
            return await _non_stream_single_request(request, current_body, headers, sid)
        except ContextOverflowError as e:
            if attempt == max_retries:
                # Return error response instead of raising
                return web.json_response(
                    {"error": f"Context overflow after {max_retries} retries: {e}"},
                    status=400,
                )

            new_max = calculate_new_max_tokens(e.ctx_limit, e.input_tokens)
            current_body["max_tokens"] = new_max
            request.app["log"].info(
                "Context overflow retry %d/%d: reducing max_tokens to %d",
                attempt + 1, max_retries, new_max,
            )

    # Unreachable but needed for type checker
    return await _non_stream_single_request(request, current_body, headers, sid)


async def _non_stream_single_request(
    request: web.Request, body: dict, headers: dict, sid: str
) -> web.Response:
    """Execute a single non-streaming request without retry logic."""
    t_start = time.monotonic()
    session = request.app["session"]

    try:
        async with session.post(
            f"{config.UPSTREAM}/v1/messages",
            data=_request_payload(body, await request.read(), headers),
            headers=headers,
            allow_redirects=False,
        ) as resp:
            content = await resp.read()
            elapsed_ms = (time.monotonic() - t_start) * 1000
            input_tokens = extract_input_tokens(content)

            # Check for context overflow error
            if resp.status >= 400:
                _raise_context_overflow(content, input_tokens)

            _note_non_stream_usage(request, sid, input_tokens, elapsed_ms)

            resp_headers = response_headers(resp.headers)
            return web.Response(
                body=content,
                status=resp.status,
                headers=resp_headers,
            )
    except (aiohttp.ClientError, TimeoutError) as e:
        request.app["log"].error("Upstream error: %s", e)
        return web.json_response({"error": "Upstream request failed"}, status=502)


def _raise_context_overflow(content: bytes, input_tokens: int | None) -> None:
    """Raise the overflow the relayed error bytes carry, if they carry one."""
    # Lenient: the body is relayed as bytes regardless, and the ASCII
    # overflow text survives replacement.
    overflow_err = parse_context_overflow_error(content.decode(errors="replace"))
    if overflow_err:
        # Use parsed tokens if available, otherwise use extracted
        if overflow_err.input_tokens is None:
            overflow_err.input_tokens = input_tokens or 0
        raise overflow_err


def _note_non_stream_usage(request, sid, input_tokens, elapsed_ms) -> None:
    """Record the request's tokens and latency; the line is verbose-only."""
    if input_tokens is not None:
        update_session_stats(sid, input_tokens, elapsed_ms)
        if config.VERBOSE:
            request.app["log"].info(
                "session %s: tokens=%d latency=%.0f ms",
                sid, input_tokens, elapsed_ms,
            )
