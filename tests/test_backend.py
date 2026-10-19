"""Tests for backend detection module.

Tests cover:
- CLI override priority
- Auto-detection via /v1/models probe
- Fallback behavior on probe failure
- Caching behavior
"""

import asyncio

import pytest
from aiohttp import web
import aiohttp

from buster_ripper import backend, config


class TestBackendDetection:
    """Test backend detection logic."""

    @pytest.fixture
    def reset_backend_state(self):
        """Reset backend module state before each test."""
        backend._backend_type = None
        backend._backend_state_cached = False
        yield
        backend._backend_type = None
        backend._backend_state_cached = False

    @pytest.fixture
    async def mock_sglang_server(self, aiohttp_server):
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
    async def mock_vllm_server(self, aiohttp_server):
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
    async def mock_unknown_server(self, aiohttp_server):
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
    async def mock_empty_server(self, aiohttp_server):
        """Create mock server with empty models list."""
        app = web.Application()
        app.router.add_get(
            "/v1/models", lambda req: web.json_response({"data": []})
        )
        return await aiohttp_server(app)

    @pytest.fixture
    async def mock_failing_server(self, aiohttp_server):
        """Create mock server that returns error."""
        app = web.Application()
        app.router.add_get(
            "/v1/models", lambda req: web.json_response({"error": "not found"}, status=404)
        )
        return await aiohttp_server(app)

    @pytest.fixture
    async def scripted_probe_server(self, aiohttp_server):
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
    async def stalled_backend_server(self, aiohttp_server):
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

    @pytest.mark.asyncio
    async def test_cli_override_takes_priority(self, reset_backend_state):
        """CLI override should take priority over auto-detection."""
        async with aiohttp.ClientSession() as session:
            result = await backend.detect_backend(
                session, "http://localhost:30000", cli_override="sglang"
            )

        assert result == "sglang"
        assert backend.get_backend_state() == "sglang"

    @pytest.mark.asyncio
    async def test_cli_override_vllm(self, reset_backend_state):
        """CLI override for vLLM backend."""
        async with aiohttp.ClientSession() as session:
            result = await backend.detect_backend(
                session, "http://localhost:30000", cli_override="vllm"
            )

        assert result == "vllm"
        assert backend.get_backend_state() == "vllm"

    @pytest.mark.asyncio
    async def test_auto_detect_sglang(self, reset_backend_state, mock_sglang_server):
        """Auto-detect SGLang backend from /v1/models response."""
        async with aiohttp.ClientSession() as session:
            result = await backend.detect_backend(
                session, f"http://localhost:{mock_sglang_server.port}"
            )

        assert result == "sglang"
        assert backend.get_backend_state() == "sglang"

    @pytest.mark.asyncio
    async def test_auto_detect_vllm(self, reset_backend_state, mock_vllm_server):
        """Auto-detect vLLM backend from /v1/models response."""
        async with aiohttp.ClientSession() as session:
            result = await backend.detect_backend(
                session, f"http://localhost:{mock_vllm_server.port}"
            )

        assert result == "vllm"
        assert backend.get_backend_state() == "vllm"

    @pytest.mark.asyncio
    async def test_auto_detect_unknown_backend(self, reset_backend_state, mock_unknown_server):
        """Unknown backend should return 'unknown'."""
        async with aiohttp.ClientSession() as session:
            result = await backend.detect_backend(
                session, f"http://localhost:{mock_unknown_server.port}"
            )

        assert result == "unknown"
        assert backend.get_backend_state() == "unknown"

    @pytest.mark.asyncio
    async def test_empty_models_returns_unknown(self, reset_backend_state, mock_empty_server):
        """Empty models list should return 'unknown'."""
        async with aiohttp.ClientSession() as session:
            result = await backend.detect_backend(
                session, f"http://localhost:{mock_empty_server.port}"
            )

        assert result == "unknown"
        assert backend.get_backend_state() == "unknown"

    @pytest.mark.asyncio
    async def test_probe_failure_returns_unknown(self, reset_backend_state, mock_failing_server):
        """Probe failure should return 'unknown'."""
        async with aiohttp.ClientSession() as session:
            result = await backend.detect_backend(
                session, f"http://localhost:{mock_failing_server.port}"
            )

        assert result == "unknown"
        assert backend.get_backend_state() == "unknown"

    @pytest.mark.asyncio
    async def test_connection_failure_returns_unknown(self, reset_backend_state):
        """Connection failure should return 'unknown'."""
        async with aiohttp.ClientSession() as session:
            result = await backend.detect_backend(
                session, "http://localhost:99999"  # Non-existent server
            )

        assert result == "unknown"
        assert backend.get_backend_state() == "unknown"

    @pytest.mark.asyncio
    async def test_stalled_probe_uses_five_second_request_deadline(
        self, reset_backend_state, stalled_backend_server
    ):
        """An uncached probe must not inherit the shared client's 600s timeout."""
        server, started = stalled_backend_server
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=600)) as session:
            loop = asyncio.get_running_loop()
            before = loop.time()
            result = await backend.detect_backend(session, f"http://localhost:{server.port}")
            elapsed = loop.time() - before

        assert started.is_set()
        assert result == "unknown"
        assert backend.get_backend_state() == "unknown"
        assert 4.5 <= elapsed < 8

    @pytest.mark.asyncio
    @pytest.mark.parametrize("case", [
        (False, "sglang", "sglang"),
        (True, None, "vllm"),
    ])
    async def test_short_circuits_do_not_probe(
        self, reset_backend_state, mocker, case
    ):
        """Both higher-priority paths avoid the request entirely."""
        cached, override, expected = case
        backend._backend_state_cached = cached
        backend._backend_type = "vllm" if cached else None
        session = mocker.Mock()

        result = await backend.detect_backend(session, "http://unused", override)

        assert result == expected
        session.get.assert_not_called()

    @pytest.mark.asyncio
    async def test_cancellation_does_not_cache_unknown(
        self, reset_backend_state, stalled_backend_server
    ):
        """Cancelling an in-flight GET propagates instead of taking fallback."""
        server, started = stalled_backend_server
        async with aiohttp.ClientSession() as session:
            task = asyncio.create_task(
                backend.detect_backend(session, f"http://localhost:{server.port}")
            )
            await asyncio.wait_for(started.wait(), timeout=2)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task

        assert backend.get_backend_state() is None
        assert not backend._backend_state_cached

    @pytest.mark.asyncio
    async def test_retry_classifies_after_transient_startup_failure(
        self, reset_backend_state, scripted_probe_server, mocker
    ):
        """A refusal at startup is retried until the upstream answers SGLang."""
        server, script, calls = scripted_probe_server
        script.extend([
            (503, {"error": "starting"}),
            (200, {"data": [{"id": "test-model", "owned_by": "SGLang"}]}),
        ])
        mocker.patch.object(backend, "_RETRY_INTERVAL", 0.01)
        async with aiohttp.ClientSession() as session:
            url = f"http://localhost:{server.port}"
            assert await backend.detect_backend(session, url) == "unknown"
            assert not backend.is_backend_state_cached()
            await asyncio.wait_for(backend._retry_detect_backend(session, url), timeout=2)

        assert backend.get_backend_state() == "sglang"
        assert backend.is_backend_state_cached()
        assert calls == [503, 200]

    @pytest.mark.asyncio
    async def test_definitive_answer_stops_retries(
        self, reset_backend_state, scripted_probe_server, mocker
    ):
        """A definitive vLLM answer caches once and the retry issues no probe."""
        server, script, calls = scripted_probe_server
        script.append((200, {"data": [{"id": "test-model", "owned_by": "vLLM"}]}))
        mocker.patch.object(backend, "_RETRY_INTERVAL", 0)
        async with aiohttp.ClientSession() as session:
            url = f"http://localhost:{server.port}"
            assert await backend.detect_backend(session, url) == "vllm"
            await asyncio.wait_for(backend._retry_detect_backend(session, url), timeout=1)

        assert backend.is_backend_state_cached()
        assert calls == [200]

    @pytest.mark.asyncio
    async def test_empty_models_answer_is_retried_not_cached(
        self, reset_backend_state, scripted_probe_server, mocker
    ):
        """An empty data list is not definitive; the retry classifies later."""
        server, script, calls = scripted_probe_server
        script.extend([
            (200, {"data": []}),
            (200, {"data": [{"id": "test-model", "owned_by": "vLLM"}]}),
        ])
        mocker.patch.object(backend, "_RETRY_INTERVAL", 0.01)
        async with aiohttp.ClientSession() as session:
            url = f"http://localhost:{server.port}"
            assert await backend.detect_backend(session, url) == "unknown"
            assert not backend.is_backend_state_cached()
            await asyncio.wait_for(backend._retry_detect_backend(session, url), timeout=2)

        assert backend.get_backend_state() == "vllm"
        assert calls == [200, 200]

    @pytest.mark.asyncio
    async def test_cli_override_issues_no_probe_or_retry(
        self, reset_backend_state, mocker
    ):
        """The CLI override wins and leaves nothing for a retry to probe."""
        session = mocker.Mock()

        result = await backend.detect_backend(session, "http://unused", "sglang")
        await backend._retry_detect_backend(session, "http://unused")

        assert result == "sglang"
        assert backend.is_backend_state_cached()
        session.get.assert_not_called()

    @pytest.mark.asyncio
    async def test_stalled_retry_never_blocks_a_state_read(
        self, reset_backend_state, stalled_backend_server, mocker
    ):
        """A hanging re-probe is background-only; cancelling it caches nothing."""
        server, started = stalled_backend_server
        mocker.patch.object(backend, "_RETRY_INTERVAL", 0)
        async with aiohttp.ClientSession() as session:
            task = asyncio.create_task(
                backend._retry_detect_backend(session, f"http://localhost:{server.port}")
            )
            await asyncio.wait_for(started.wait(), timeout=2)
            loop = asyncio.get_running_loop()
            before = loop.time()
            state = backend.get_backend_state()
            elapsed = loop.time() - before
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task

        assert state is None
        assert elapsed < 0.5
        assert backend.get_backend_state() is None
        assert not backend.is_backend_state_cached()

    @pytest.mark.asyncio
    async def test_caching_prevents_duplicate_probes(self, reset_backend_state, mock_sglang_server):
        """Cached result should prevent duplicate probes."""
        async with aiohttp.ClientSession() as session:
            url = f"http://localhost:{mock_sglang_server.port}"

            # First call should probe
            result1 = await backend.detect_backend(session, url)
            assert result1 == "sglang"

            # Second call should use cache
            result2 = await backend.detect_backend(session, url)
            assert result2 == "sglang"

    @pytest.mark.asyncio
    async def test_cli_override_overrides_cached_value(self, reset_backend_state, mock_vllm_server):
        """CLI override should override cached value."""
        async with aiohttp.ClientSession() as session:
            url = f"http://localhost:{mock_vllm_server.port}"

            # First call auto-detects vLLM
            result1 = await backend.detect_backend(session, url)
            assert result1 == "vllm"

            # CLI override should take priority
            result2 = await backend.detect_backend(
                session, url, cli_override="sglang"
            )
            assert result2 == "sglang"

    def test_get_backend_state_returns_none_before_detection(self, reset_backend_state):
        """get_backend_state should return None before detection."""
        assert backend.get_backend_state() is None

    @pytest.mark.asyncio
    async def test_case_insensitive_detection(self, reset_backend_state, aiohttp_server):
        """Backend detection should be case-insensitive."""
        app = web.Application()
        app.router.add_get(
            "/v1/models",
            lambda req: web.json_response(
                {"data": [{"id": "test-model", "owned_by": "SGLANG"}]}
            ),
        )
        server = await aiohttp_server(app)

        async with aiohttp.ClientSession() as session:
            result = await backend.detect_backend(
                session, f"http://localhost:{server.port}"
            )

        assert result == "sglang"

    @pytest.mark.asyncio
    async def test_owned_by_with_extra_text(self, reset_backend_state, aiohttp_server):
        """Detection should handle owned_by with additional text."""
        app = web.Application()
        app.router.add_get(
            "/v1/models",
            lambda req: web.json_response(
                {"data": [{"id": "test-model", "owned_by": "SGLang v0.2.0"}]}
            ),
        )
        server = await aiohttp_server(app)

        async with aiohttp.ClientSession() as session:
            result = await backend.detect_backend(
                session, f"http://localhost:{server.port}"
            )

        assert result == "sglang"

    @pytest.mark.asyncio
    async def test_integration_with_config(self, reset_backend_state, mock_sglang_server):
        """Test integration with config.BACKEND_TYPE_OVERRIDE."""
        config.BACKEND_TYPE_OVERRIDE = "vllm"

        async with aiohttp.ClientSession() as session:
            result = await backend.detect_backend(
                session,
                f"http://localhost:{mock_sglang_server.port}",
                cli_override=config.BACKEND_TYPE_OVERRIDE,
            )

        assert result == "vllm"

        # Cleanup
        config.BACKEND_TYPE_OVERRIDE = None
