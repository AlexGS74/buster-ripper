"""POST /v1/chat/completions — eval-mode proxy with model profile strategies."""

import json
import logging

import httpx
from fastapi import APIRouter, Request
from fastapi.responses import Response

from .. import config
from ..utils import forward_headers, response_headers

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
                if config.VERBOSE:
                    log.info("eval-mode: injected chat_template_kwargs enable_thinking=%s", config.EVAL_THINKING)

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
    if config.EVAL_MODE:
        try:
            resp_data = resp.json()
            fixed = False

            for choice in resp_data.get("choices", []):
                msg = choice.get("message", {})

                # Profile strategy: copy reasoning_content → content when null
                if profile.copy_reasoning_to_content and not msg.get("content"):
                    thinking = msg.get("reasoning_content") or msg.get("reasoning") or ""
                    if thinking:
                        msg["content"] = thinking
                        fixed = True
                        if config.VERBOSE:
                            log.info("eval-mode: copied reasoning_content → content")

                # Profile strategy: strip code fences so lm-eval filter assembles
                # the full function correctly (doc["prompt"] + bare_body)
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
