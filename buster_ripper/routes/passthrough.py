"""Catch-all passthrough for all other endpoints (health, /v1/models, etc.)."""

import aiohttp
from aiohttp import web

from ..utils import forward_headers, response_headers


async def handle_passthrough(request: web.Request) -> web.Response:
    """Passthrough request to upstream server."""
    path = request.match_info.get("path", "")

    # Read request body if present
    body = await request.read()

    # Build headers, removing hop-by-hop (case-insensitive)
    headers = forward_headers(request.headers)

    # Build upstream URL
    upstream_url = f"{config.UPSTREAM}/{path}"

    # Get query parameters
    params = dict(request.query)

    # Get shared session from app
    session = request.app["session"]
    timeout = aiohttp.ClientTimeout(total=60)

    try:
        async with session.request(
            method=request.method,
            url=upstream_url,
            data=body if body else None,
            headers=headers,
            params=params,
            timeout=timeout,
            allow_redirects=False,
        ) as resp:
            content = await resp.read()

            # Build response headers
            resp_headers = response_headers(resp.headers)

            return web.Response(
                body=content,
                status=resp.status,
                headers=resp_headers,
            )
    except (aiohttp.ClientError, TimeoutError) as e:
        # Same rule as the other boundaries: the exception is the operator's, the
        # body is the client's, and neither the upstream host nor its port is.
        request.app["log"].error("Upstream error: %s", e)
        return web.json_response({"error": "Upstream request failed"}, status=502)


# Import config for the module
from .. import config  # noqa: E402
