"""POST /v1/chat/completions — eval-mode proxy with model profile strategies."""

import json
import logging

import httpx
from fastapi import APIRouter, Request
from fastapi.responses import Response

from .. import config
from ..utils import forward_headers, response_headers, split_thinking

router = APIRouter()
log = logging.getLogger("buster-ripper")


@router.post("/v1/chat/completions")
async def chat_completions(request: Request) -> Response:
    """Forward chat completions to vLLM with optional eval-mode transformations.

    Model-agnostic (always applied in eval mode):
      - Drops max_gen_toks (lm-eval internal field; vLLM ignores it with a warning
        and lm-eval also uses it to truncate responses internally).
      - Strips empty Bearer auth header (lm-eval sends "Bearer " with no token).
      - Applies max_tokens cap if --eval-max-tokens was specified.

    Model-specific (controlled by --eval-profile):
      - inject_chat_template_kwargs: injects enable_thinking into the request.
      - copy_reasoning_to_content:   copies reasoning_content → content when null.
      - strip_code_fences:           extracts bare code from ```...``` wrappers so
                                     lm-eval's build_predictions_instruct filter
                                     can assemble doc["prompt"] + body correctly.
    """
    body = await request.body()
    headers = forward_headers(request)
    profile = config.EVAL_PROFILE

    # ── Request transforms ────────────────────────────────────────────────────
    if config.EVAL_MODE and body:
        try:
            data = json.loads(body)

            # Model-agnostic: strip lm-eval internal field
            data.pop("max_gen_toks", None)

            # Model-agnostic: cap max_tokens
            if config.EVAL_MAX_TOKENS > 0:
                data.setdefault("max_tokens", config.EVAL_MAX_TOKENS)

            # Profile strategy: inject chat_template_kwargs
            if profile.inject_chat_template_kwargs:
                kwargs = data.setdefault("chat_template_kwargs", {})
                kwargs.setdefault("enable_thinking", config.EVAL_THINKING)
                if config.EVAL_THINKING_BUDGET > 0:
                    kwargs.setdefault("thinking_budget", config.EVAL_THINKING_BUDGET)
                if config.VERBOSE:
                    log.info(
                        "eval-mode: injected chat_template_kwargs enable_thinking=%s thinking_budget=%s",
                        config.EVAL_THINKING,
                        config.EVAL_THINKING_BUDGET if config.EVAL_THINKING_BUDGET > 0 else "unlimited",
                    )

            body = json.dumps(data).encode()
            headers["content-length"] = str(len(body))
        except Exception:
            pass  # leave body untouched on parse error

    # Model-agnostic: strip empty/invalid Authorization header
    auth = headers.get("authorization", "")
    if auth.strip() in ("Bearer", "Bearer "):
        headers.pop("authorization", None)

    async with httpx.AsyncClient(timeout=300) as client:
        resp = await client.post(
            f"{config.UPSTREAM}/v1/chat/completions",
            content=body,
            headers=headers,
        )

    # ── Response transforms ───────────────────────────────────────────────────
    try:
        resp_data = resp.json()
        fixed = False

        for choice in resp_data.get("choices", []):
            msg = choice.get("message", {})
            content = msg.get("content") or ""

            if config.EVAL_MODE:
                # Eval path: produce clean OAI content for lm-eval.
                # Do NOT add non-standard fields (reasoning_content) that lm-eval
                # doesn't understand and that can confuse downstream scoring.

                # Strip embedded <think>...</think> block from content.
                # vLLM may return the full generation (thinking + answer) in the
                # content field; keep only the answer part.
                if "<think>" in content:
                    _, answer = split_thinking(content)
                    if config.VERBOSE:
                        log.info("eval-mode: stripped <think> block from content (%d→%d chars)",
                                 len(content), len(answer))
                    msg["content"] = answer
                    content = answer
                    fixed = True

                # Profile strategy: copy reasoning_content → content when null.
                if profile.copy_reasoning_to_content and not msg.get("content"):
                    thinking = msg.get("reasoning_content") or msg.get("reasoning") or ""
                    if thinking:
                        msg["content"] = thinking
                        content = thinking
                        fixed = True
                        if config.VERBOSE:
                            log.info("eval-mode: copied reasoning_content → content")

                # Profile strategy: strip code fences so lm-eval filter assembles
                # the full function correctly (doc["prompt"] + bare_body).
                # Only fires when response *starts* with a fence — using find("```")
                # was causing false positives on responses that contain ``` inside
                # docstrings but have no outer fence wrapping.
                if profile.strip_code_fences:
                    content = msg.get("content") or ""
                    if content.startswith("```"):
                        after_open = content[content.index("\n") + 1:] if "\n" in content else ""
                        close_idx = after_open.rfind("\n```")
                        bare = after_open[:close_idx] if close_idx != -1 else after_open
                        msg["content"] = bare
                        fixed = True
                        if config.VERBOSE:
                            log.info("eval-mode: stripped code fence (%d→%d chars)", len(content), len(bare))

            else:
                # Non-eval (CC) path: split <think>...</think> out of content and
                # populate reasoning_content so CC clients see a clean separation.
                if "<think>" in content:
                    thinking, answer = split_thinking(content)
                    msg["content"] = answer
                    if not msg.get("reasoning_content"):
                        msg["reasoning_content"] = thinking
                    fixed = True
                    if config.VERBOSE:
                        log.info("split <think> block from content (%d thinking, %d answer chars)",
                                 len(thinking), len(answer))

        if fixed:
            resp_body = json.dumps(resp_data).encode()
            resp_headers = dict(response_headers(resp.headers))
            resp_headers["content-length"] = str(len(resp_body))
            return Response(content=resp_body, status_code=resp.status_code, headers=resp_headers)
    except Exception:
        pass  # fall through to normal response on parse error

    return Response(
        content=resp.content,
        status_code=resp.status_code,
        headers=response_headers(resp.headers),
    )
