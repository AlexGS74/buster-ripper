"""Catch-all passthrough for all other endpoints (health, /v1/models, etc.)."""

import httpx
from fastapi import APIRouter, Request
from fastapi.responses import Response

from .. import config
from ..utils import forward_headers, response_headers

router = APIRouter()


@router.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "HEAD"])
async def passthrough(request: Request, path: str) -> Response:
    body = await request.body()
    headers = forward_headers(request)
    async with httpx.AsyncClient(timeout=60) as client:
        resp = await client.request(
            method=request.method,
            url=f"{config.UPSTREAM}/{path}",
            content=body,
            headers=headers,
            params=dict(request.query_params),
        )
    return Response(
        content=resp.content,
        status_code=resp.status_code,
        headers=response_headers(resp.headers),
    )
