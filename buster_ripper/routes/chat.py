"""POST /v1/chat/completions — eval-mode proxy with model profile strategies."""

import json
import logging

import aiohttp
from aiohttp import web

from .. import backend
from .. import config
from ..utils import (
    ContextOverflowError,
    calculate_new_max_tokens,
    forward_headers,
    parse_context_overflow_error,
    response_headers,
    split_thinking,
)


log = logging.getLogger("buster-ripper")


async def _read_chat_body(request: web.Request) -> tuple:
    """Parse the client's body; a non-object or unparseable root means answer 400."""
    try:
        body = await request.json()
    except (json.JSONDecodeError, UnicodeDecodeError, LookupError):
        return None, b""
    return (body if isinstance(body, dict) else None), await request.read()


async def handle_chat(request: web.Request) -> web.Response:
    """Forward chat completions with eval-mode transforms; SGLang overflow retry."""
    body, _ = await _read_chat_body(request)
    if body is None:
        return web.json_response({"error": "Invalid JSON"}, status=400)

    profile = config.EVAL_PROFILE
    body = _apply_request_transforms(body, profile)

    headers = _prepare_headers(request.headers)
    return await _chat_with_overflow_retry(request, body, headers, profile)


async def _chat_with_overflow_retry(request, body, headers, profile) -> web.Response:
    backend_type = backend.get_backend_state()

    if backend_type != "sglang":
        return await _make_chat_request(request, body, headers, profile)

    max_retries = 2
    current_body = dict(body)

    for attempt in range(max_retries + 1):
        try:
            return await _make_chat_request(request, current_body, headers, profile)
        except ContextOverflowError as e:
            if attempt == max_retries:
                return web.json_response(
                    {"error": f"Context overflow after {max_retries} retries: {e}"},
                    status=400,
                )

            new_max = calculate_new_max_tokens(e.ctx_limit, e.input_tokens)
            current_body["max_completion_tokens"] = new_max
            log.info("Context overflow retry %d/%d: reducing max_completion_tokens to %d",
                     attempt + 1, max_retries, new_max)

    # Unreachable but needed for type checker
    return await _make_chat_request(request, current_body, headers, profile)


def _chat_template_kwargs(body: dict) -> dict:
    """A client-sent non-mapping is replaced, not skipped (the messages route's rule)."""
    kwargs = body.get("chat_template_kwargs")
    if not isinstance(kwargs, dict):
        body["chat_template_kwargs"] = kwargs = {}
    return kwargs


def _apply_request_transforms(body: dict, profile) -> dict:
    if not config.EVAL_MODE or not body:
        return body

    try:
        body.pop("max_gen_toks", None)

        if config.EVAL_MAX_TOKENS > 0:
            body.setdefault("max_tokens", config.EVAL_MAX_TOKENS)

        if profile.inject_chat_template_kwargs:
            _inject_eval_kwargs(body)
    except Exception:
        pass

    return body


def _inject_eval_kwargs(body: dict) -> None:
    kwargs = _chat_template_kwargs(body)
    kwargs.setdefault("enable_thinking", config.EVAL_THINKING)
    if config.EVAL_THINKING_BUDGET > 0:
        kwargs.setdefault("thinking_budget", config.EVAL_THINKING_BUDGET)
    if config.VERBOSE:
        log.info(
            "eval-mode: injected chat_template_kwargs enable_thinking=%s thinking_budget=%s",
            config.EVAL_THINKING,
            config.EVAL_THINKING_BUDGET if config.EVAL_THINKING_BUDGET > 0 else "unlimited",
        )


def _prepare_headers(request_headers) -> dict:
    """Upstream headers: shared hop-by-hop policy plus the chat-local empty-Bearer rule."""
    headers = forward_headers(request_headers)

    # lm-eval's empty "Bearer " carries no token: read the raw mapping, not the plain dict.
    values = request_headers.getall("authorization", ())
    keep = next((v for v in values if _carries_token(v)), None)
    if keep is None:
        # Value-less ``Authorization:`` survives as base forwarded it; ``None`` is not ``""``.
        keep = next((v for v in values if not v.strip()), None)

    for name in [k for k in headers if k.lower() == "authorization"]:
        del headers[name]
    if keep is not None:
        headers["Authorization"] = keep
    return headers


def _carries_token(value: str) -> bool:
    """True when a line names a credential; ruling only the bearer form out lets a blank line delete the real one."""
    stripped = value.strip()
    return bool(stripped) and stripped != "Bearer"


def _chat_payload(body: dict, raw: bytes, headers: dict) -> tuple:
    """Relay the client's own bytes unless a transform changed the body."""
    try:
        if json.loads(raw) == body:
            return raw, headers
    except (json.JSONDecodeError, UnicodeDecodeError, LookupError):
        pass
    headers["Content-Type"] = "application/json"
    return json.dumps(body).encode(), headers


async def _make_chat_request(
    request: web.Request, body: dict, headers: dict, profile,
) -> web.Response:
    timeout = aiohttp.ClientTimeout(total=None if body.get("stream") else 300)
    payload, headers = _chat_payload(body, await request.read(), headers)
    session = request.app["session"]

    try:
        async with session.post(
            f"{config.UPSTREAM}/v1/chat/completions",
            data=payload,
            headers=headers, allow_redirects=False,
            timeout=timeout,
        ) as resp:
            if body.get("stream"):
                return await _relay_chat_stream(request, resp, profile)
            return _chat_answer(resp, await resp.read(), profile)
    except (aiohttp.ClientError, TimeoutError) as e:
        log.error("Upstream error: %s", e)
        return web.json_response({"error": "Upstream request failed"}, status=502)


def _chat_answer(resp, content: bytes, profile) -> web.Response:
    """Relay a finished answer: transform the parsed form, else the upstream's bytes."""
    try:
        resp_data = json.loads(content)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return _chat_response(resp, content)
    _raise_chat_overflow(resp_data, resp.status)
    if _apply_response_transforms(resp_data, profile):
        content = json.dumps(resp_data).encode()
    return _chat_response(resp, content)


def _chat_response(resp, body: bytes) -> web.Response:
    """Answer with the upstream's status and its non-hop-by-hop headers, as base did."""
    return web.Response(body=body, status=resp.status, headers=response_headers(resp.headers))


def _raise_chat_overflow(resp_data, status: int) -> None:
    if status < 400 or not isinstance(resp_data, dict):
        return
    overflow_err = parse_context_overflow_error(json.dumps(resp_data))
    if overflow_err:
        usage = resp_data.get("usage")
        if overflow_err.input_tokens is None:
            overflow_err.input_tokens = usage.get("prompt_tokens", 0) if isinstance(usage, dict) else 0
        raise overflow_err


async def _relay_chat_stream(request: web.Request, resp, profile) -> web.StreamResponse:
    """Relay SSE chunks without decoding or applying JSON-only transforms."""
    if resp.status >= 300:
        return _chat_answer(resp, await resp.read(), profile)
    response = web.StreamResponse(status=resp.status, headers={
        "Content-Type": "text/event-stream", "Cache-Control": "no-cache",
        "X-Accel-Buffering": "no", "Connection": "keep-alive",
    })
    await response.prepare(request)
    try:
        async for chunk in resp.content.iter_any():
            await response.write(chunk)
        await response.write_eof()
    except Exception as exc:
        log.error("Post-commit relay error (%s): %s", type(exc).__name__, exc)
    return response


def _apply_response_transforms(resp_data: dict, profile) -> bool:
    """Rewrite the parsed answer in place; True when a transform changed it."""
    try:
        fixed = False

        for choice in resp_data.get("choices", []):
            msg = choice.get("message", {})
            content = msg.get("content") or ""

            if config.EVAL_MODE:
                content, fixed = _transform_eval_response(msg, content, profile, fixed)
            else:
                content, fixed = _transform_cc_response(msg, content, fixed)
    except Exception:
        return False

    return fixed


def _strip_think_block(msg: dict, content: str) -> tuple[str, bool]:
    if "<think" not in content:
        return content, False

    _, answer = split_thinking(content)
    if config.VERBOSE:
        log.info("eval-mode: stripped <think block from content (%d -> %d chars)",
                 len(content), len(answer))
    msg["content"] = answer
    return answer, True


def _copy_reasoning_to_content(msg: dict, content: str) -> tuple[str, bool]:
    if msg.get("content"):
        return content, False

    thinking = msg.get("reasoning_content") or msg.get("reasoning") or ""
    if not thinking:
        return content, False

    msg["content"] = thinking
    if config.VERBOSE:
        log.info("eval-mode: copied reasoning_content -> content")
    return thinking, True


def _strip_code_fence(msg: dict, content: str) -> tuple[str, bool]:
    if not content.startswith("```"):
        return content, False

    after_open = content[content.index("\n") + 1:] if "\n" in content else ""
    close_idx = after_open.rfind("\n```")
    bare = after_open[:close_idx] if close_idx != -1 else after_open
    msg["content"] = bare
    if config.VERBOSE:
        log.info("eval-mode: stripped code fence (%d -> %d chars)", len(content), len(bare))
    return bare, True


def _transform_eval_response(msg: dict, content: str, profile, fixed: bool) -> tuple:
    steps = [_strip_think_block]
    if profile.copy_reasoning_to_content:
        steps.append(_copy_reasoning_to_content)
    if profile.strip_code_fences:
        steps.append(_strip_code_fence)

    for step in steps:
        content, changed = step(msg, content)
        fixed = fixed or changed
    return content, fixed


def _transform_cc_response(msg: dict, content: str, fixed: bool) -> tuple:
    if "<think" in content:
        thinking, answer = split_thinking(content)
        msg["content"] = answer
        if not msg.get("reasoning_content"):
            msg["reasoning_content"] = thinking
        fixed = True
        if config.VERBOSE:
            log.info("split <think block from content (%d thinking, %d answer chars)",
                     len(thinking), len(answer))

    return content, fixed
