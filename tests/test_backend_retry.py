"""Tests for the backend retry loop that re-probes until an answer caches the backend."""

import asyncio

import pytest
import aiohttp

from buster_ripper import backend
from tests.backend_probe import (
    reset_backend_state,
    scripted_probe_server,
    stalled_backend_server,
)


class TestBackendRetry:
    """The retry loop re-probes until a definitive answer caches the backend."""

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
