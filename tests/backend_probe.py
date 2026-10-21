"""Mock /v1/models peers and a backend-state reset for the backend tests."""

import asyncio

import pytest
from aiohttp import web

from buster_ripper import backend


@pytest.fixture
def reset_backend_state():
    """Reset backend module state before each test."""
    backend._backend_type = None
    backend._backend_state_cached = False
    yield
    backend._backend_type = None
    backend._backend_state_cached = False


@pytest.fixture
async def mock_sglang_server(aiohttp_server):
    """Create mock SGLang server."""
    app = web.Application()
    app.router.add_get(
        "/v1/models",
        lambda req: web.json_response(
            {"data": [{"id": "test-model", "owned_by": "SGLang"}]}
        ),
    )
    return await aiohttp_server(app)


@pytest.fixture
async def mock_vllm_server(aiohttp_server):
    """Create mock vLLM server."""
    app = web.Application()
    app.router.add_get(
        "/v1/models",
        lambda req: web.json_response(
            {"data": [{"id": "test-model", "owned_by": "vLLM"}]}
        ),
    )
    return await aiohttp_server(app)


@pytest.fixture
async def mock_unknown_server(aiohttp_server):
    """Create mock server with unknown backend."""
    app = web.Application()
    app.router.add_get(
        "/v1/models",
        lambda req: web.json_response(
            {"data": [{"id": "test-model", "owned_by": "unknown"}]}
        ),
    )
    return await aiohttp_server(app)


@pytest.fixture
async def mock_empty_server(aiohttp_server):
    """Create mock server with empty models list."""
    app = web.Application()
    app.router.add_get(
        "/v1/models", lambda req: web.json_response({"data": []})
    )
    return await aiohttp_server(app)


@pytest.fixture
async def mock_failing_server(aiohttp_server):
    """Create mock server that returns error."""
    app = web.Application()
    app.router.add_get(
        "/v1/models", lambda req: web.json_response({"error": "not found"}, status=404)
    )
    return await aiohttp_server(app)


@pytest.fixture
async def scripted_probe_server(aiohttp_server):
    """A /v1/models peer answering from a script; the last answer repeats."""
    script = []
    calls = []

    async def models(request):
        status, payload = script[min(len(calls), len(script) - 1)]
        calls.append(status)
        return web.json_response(payload, status=status)

    app = web.Application()
    app.router.add_get("/v1/models", models)
    return await aiohttp_server(app), script, calls


@pytest.fixture
async def stalled_backend_server(aiohttp_server):
    """A probe peer that accepts GET and never sends a response."""
    started = asyncio.Event()
    never_released = asyncio.Event()

    async def stall(request):
        started.set()
        await never_released.wait()
        return web.json_response({"data": []})

    app = web.Application()
    app.router.add_get("/v1/models", stall)
    return await aiohttp_server(app), started
