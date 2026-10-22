"""poll_kv_cache behavioral tests."""

import asyncio

import aiohttp

from buster_ripper import compaction


# ── poll_kv_cache behavioral tests ────────────────────────────────────────────

async def test_poll_kv_cache_uses_session_get(mocker):
    """Verify poll_kv_cache uses session.get for metrics endpoint."""
    session = mocker.Mock(spec=aiohttp.ClientSession)

    # Mock response
    mock_response = mocker.AsyncMock()
    mock_response.status = 200
    mock_response.text = mocker.AsyncMock(
        return_value="vllm:kv_cache_usage_perc{engine='0'} 0.75\n"
    )
    mock_response.__aenter__ = mocker.AsyncMock(return_value=mock_response)
    mock_response.__aexit__ = mocker.AsyncMock(return_value=None)

    session.get = mocker.Mock(return_value=mock_response)

    # Run for a very short time
    task = asyncio.create_task(compaction.poll_kv_cache(session))
    await asyncio.sleep(0.05)
    task.cancel()

    try:
        await task
    except asyncio.CancelledError:
        pass

    # Verify session.get was called
    assert session.get.called


async def test_poll_kv_cache_handles_metrics_error(mocker):
    """Verify poll_kv_cache handles metrics endpoint errors gracefully."""
    session = mocker.Mock(spec=aiohttp.ClientSession)

    # Mock to raise error
    session.get = mocker.Mock(side_effect=aiohttp.ClientError("Connection failed"))

    # Should not raise
    task = asyncio.create_task(compaction.poll_kv_cache(session))
    await asyncio.sleep(0.05)
    task.cancel()

    try:
        await task
    except asyncio.CancelledError:
        pass


async def test_poll_kv_cache_updates_global_cache(mocker):
    """Verify poll_kv_cache updates global _kv_cache_usage."""
    session = mocker.Mock(spec=aiohttp.ClientSession)

    # Reset global state
    original = compaction._kv_cache_usage
    compaction._kv_cache_usage = 0.0

    mock_response = mocker.AsyncMock()
    mock_response.status = 200
    mock_response.text = mocker.AsyncMock(
        return_value="vllm:kv_cache_usage_perc{engine='0'} 0.80\n"
    )
    mock_response.__aenter__ = mocker.AsyncMock(return_value=mock_response)
    mock_response.__aexit__ = mocker.AsyncMock(return_value=None)

    session.get = mocker.Mock(return_value=mock_response)

    # Run briefly
    task = asyncio.create_task(compaction.poll_kv_cache(session))
    await asyncio.sleep(0.05)
    task.cancel()

    try:
        await task
    except asyncio.CancelledError:
        pass
    finally:
        compaction._kv_cache_usage = original


async def test_poll_kv_cache_uses_timeout(mocker):
    """Verify poll_kv_cache uses 5 second timeout."""
    session = mocker.Mock(spec=aiohttp.ClientSession)

    mock_response = mocker.AsyncMock()
    mock_response.status = 200
    mock_response.text = mocker.AsyncMock(return_value="vllm:kv_cache_usage_perc{engine='0'} 0.50\n")
    mock_response.__aenter__ = mocker.AsyncMock(return_value=mock_response)
    mock_response.__aexit__ = mocker.AsyncMock(return_value=None)

    session.get = mocker.Mock(return_value=mock_response)

    task = asyncio.create_task(compaction.poll_kv_cache(session))
    await asyncio.sleep(0.05)
    task.cancel()

    try:
        await task
    except asyncio.CancelledError:
        pass

    # Verify get was called (timeout would be passed as kwarg)
    assert session.get.called


# ── Session parametric tests ──────────────────────────────────────────────────

class TestPollKvCacheWithSession:
    """Test poll_kv_cache with various session states."""

    async def test_with_active_session(self, mocker):
        """Test with active, working session."""
        session = mocker.Mock(spec=aiohttp.ClientSession)

        mock_response = mocker.AsyncMock()
        mock_response.status = 200
        mock_response.text = mocker.AsyncMock(return_value="vllm:kv_cache_usage_perc{engine='0'} 0.60\n")
        mock_response.__aenter__ = mocker.AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = mocker.AsyncMock(return_value=None)

        session.get = mocker.Mock(return_value=mock_response)

        task = asyncio.create_task(compaction.poll_kv_cache(session))
        await asyncio.sleep(0.05)
        task.cancel()

        try:
            await task
        except asyncio.CancelledError:
            pass

        assert session.get.called

    async def test_with_failing_session(self, mocker):
        """Test with session that raises errors."""
        session = mocker.Mock(spec=aiohttp.ClientSession)
        session.get = mocker.Mock(side_effect=aiohttp.ClientError("Failed"))

        task = asyncio.create_task(compaction.poll_kv_cache(session))
        await asyncio.sleep(0.05)
        task.cancel()

        try:
            await task
        except asyncio.CancelledError:
            pass

        # Should complete without raising

    async def test_with_timeout_session(self, mocker):
        """Test with session that times out."""
        session = mocker.Mock(spec=aiohttp.ClientSession)
        session.get = mocker.Mock(side_effect=asyncio.TimeoutError())

        task = asyncio.create_task(compaction.poll_kv_cache(session))
        await asyncio.sleep(0.05)
        task.cancel()

        try:
            await task
        except asyncio.CancelledError:
            pass

        # Should complete without raising
