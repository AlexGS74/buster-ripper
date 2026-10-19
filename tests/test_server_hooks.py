"""Tests for server.py startup and lifecycle hooks."""

import asyncio
from contextlib import suppress

import pytest

from buster_ripper.server import (
    on_cleanup,
    on_startup,
    spawn_dump_flush,
    spawn_image_expiry,
    spawn_kv_cache_monitor,
    start_kv_cache_monitoring,
)
from buster_ripper import backend, config, dump
from buster_ripper.session import session_id


async def _never_ends():
    """A stand-in background task that only a cancel can end."""
    await asyncio.sleep(3600)


async def _fails():
    """A task that has already stored an ordinary error by cleanup time."""
    raise RuntimeError("background failed")


@pytest.fixture()
def pending_dump(tmp_path, mocker):
    """A collapsed record waiting for shutdown rather than a timer."""
    config.DUMP_DIR = tmp_path
    config.DUMP_FORMAT = "change-focused"
    dump._sessions.clear()
    body = {"system": "shutdown", "messages": [{"role": "user", "content": "pending"}]}
    dump.dump_turn(body)
    yield tmp_path, session_id(body), mocker.AsyncMock()
    dump._sessions.clear()
    config.DUMP_DIR = None
    config.DUMP_FORMAT = "documented"


class TestSpawnKvCacheMonitor:
    """Tests for spawn_kv_cache_monitor function."""

    def test_returns_none_immediately(self, mocker):
        """Verify spawn function returns immediately without awaiting."""
        mock_create_task = mocker.patch("asyncio.create_task")

        app = {}

        result = spawn_kv_cache_monitor(app)

        assert result is None
        mock_create_task.assert_called_once()

    def test_stores_task_in_app_state(self, mocker):
        """Verify task is stored in app state for cleanup."""
        mock_task = mocker.Mock()
        mocker.patch("asyncio.create_task", return_value=mock_task)

        app = {}

        spawn_kv_cache_monitor(app)

        assert "kv_monitor_task" in app
        assert app["kv_monitor_task"] is mock_task

    def test_creates_task_with_correct_coroutine(self, mocker):
        """Verify spawn creates task with start_kv_cache_monitoring coroutine."""
        mock_create_task = mocker.patch("asyncio.create_task")

        session = mocker.Mock()
        app = {"session": session}

        spawn_kv_cache_monitor(app)

        # Verify create_task was called
        assert mock_create_task.called
        # The call should be with a coroutine object
        call_arg = mock_create_task.call_args[0][0]
        # coroutine objects have a cr_frame attribute
        assert hasattr(call_arg, "cr_frame") or hasattr(call_arg, "cr_code")


class TestStartKvCacheMonitoring:
    """Tests for start_kv_cache_monitoring function."""

    @pytest.mark.asyncio
    async def test_delegates_to_compaction_module(self, mocker):
        """Verify the monitoring function calls compaction.poll_kv_cache."""
        mock_poll = mocker.patch("buster_ripper.compaction.poll_kv_cache")

        session = mocker.Mock()
        app = {"session": session}

        await start_kv_cache_monitoring(app)

        mock_poll.assert_called_once_with(session)

    @pytest.mark.asyncio
    async def test_passes_session_from_app_context(self, mocker):
        """Verify session is correctly extracted from app and passed along."""
        mock_poll = mocker.patch("buster_ripper.compaction.poll_kv_cache")

        expected_session = mocker.Mock(name="expected_session")
        app = {"session": expected_session}

        await start_kv_cache_monitoring(app)

        mock_poll.assert_called_once_with(expected_session)


class TestSpawnDumpFlush:
    """Tests for spawn_dump_flush function (T003)."""

    def test_stores_task_in_app_state(self, mocker):
        """Verify the flush task is stored in app state for cleanup."""
        mock_task = mocker.Mock()
        mocker.patch("buster_ripper.dump.poll_flush")
        mocker.patch("asyncio.create_task", return_value=mock_task)

        app = {}

        spawn_dump_flush(app)

        assert app["dump_flush_task"] is mock_task

    def test_spawns_dump_poll_flush_coroutine(self, mocker):
        """Verify the spawned task runs dump.poll_flush."""
        mock_poll = mocker.patch("buster_ripper.dump.poll_flush")
        mocker.patch("asyncio.create_task")

        spawn_dump_flush({})

        mock_poll.assert_called_once_with()


class TestSpawnImageExpiry:
    """The periodic image sweep is spawned and torn down like the other pollers."""

    def test_stores_task_in_app_state(self, mocker):
        """Verify the sweep task is stored in app state for cleanup."""
        mock_task = mocker.Mock()
        mocker.patch("buster_ripper.image.poll_expiry", new=mocker.Mock())
        mocker.patch("asyncio.create_task", return_value=mock_task)

        app = {}

        spawn_image_expiry(app)

        assert app["image_expiry_task"] is mock_task

    def test_spawns_image_poll_expiry_coroutine(self, mocker):
        """Verify the spawned task runs image.poll_expiry."""
        mock_poll = mocker.patch("buster_ripper.image.poll_expiry", new=mocker.Mock())
        mocker.patch("asyncio.create_task")

        spawn_image_expiry({})

        mock_poll.assert_called_once_with()


class TestOnStartupBackendRetry:
    """A retry task is spawned only while the backend is still unknown."""

    @pytest.fixture()
    def startup_app(self, mocker):
        """Run on_startup with the other background spawns stubbed."""
        mocker.patch("buster_ripper.server.spawn_kv_cache_monitor")
        mocker.patch("buster_ripper.server.spawn_dump_flush")
        yield mocker.patch("buster_ripper.server.spawn_image_expiry")
        backend._backend_type = None
        backend._backend_state_cached = False

    async def test_image_expiry_is_spawned(self, startup_app, mocker):
        """Startup spawns the image sweep beside the other periodic tasks."""
        mocker.patch.object(config, "BACKEND_TYPE_OVERRIDE", "sglang")
        app = {}
        await on_startup(app)
        try:
            startup_app.assert_called_once_with(app)
        finally:
            await app["session"].close()

    async def test_cli_override_spawns_no_retry(self, startup_app, mocker):
        """The override classifies the backend, so no retry task is needed."""
        mocker.patch.object(config, "BACKEND_TYPE_OVERRIDE", "sglang")
        app = {}
        await on_startup(app)
        try:
            assert backend.get_backend_state() == "sglang"
            assert "backend_retry_task" not in app
        finally:
            await app["session"].close()

    async def test_unknown_detection_spawns_a_retry(self, startup_app, mocker):
        """An unknown, uncached detection leaves a live retry task behind."""
        mocker.patch("buster_ripper.server.backend.detect_backend", return_value="unknown")
        app = {}
        await on_startup(app)
        try:
            assert not app["backend_retry_task"].done()
        finally:
            app["backend_retry_task"].cancel()
            with suppress(asyncio.CancelledError):
                await app["backend_retry_task"]
            await app["session"].close()


class TestOnCleanupBackendRetry:
    """on_cleanup ends the retry task before the shared session closes."""

    async def test_cancels_retry_before_closing_session(self, mocker):
        """The retry task is cancelled and awaited ahead of session close."""
        retry = asyncio.create_task(_never_ends())
        session = mocker.AsyncMock()
        at_close = []
        session.close.side_effect = lambda: at_close.append(retry.cancelled())

        await on_cleanup({"backend_retry_task": retry, "session": session})

        assert at_close == [True]
        session.close.assert_awaited_once()


class TestOnCleanupImageExpiry:
    """on_cleanup ends the image sweep before the shared session closes."""

    async def test_cancels_sweep_before_closing_session(self, mocker):
        """The sweep task is cancelled and awaited ahead of session close."""
        sweep = asyncio.create_task(_never_ends())
        session = mocker.AsyncMock()
        at_close = []
        session.close.side_effect = lambda: at_close.append(sweep.cancelled())

        await on_cleanup({"image_expiry_task": sweep, "session": session})

        assert at_close == [True]
        session.close.assert_awaited_once()


class TestOnCleanupDumpFlush:
    """T003: on_cleanup cancels the flush task, flushes, then closes the session."""

    async def test_teardown_order_cancel_then_flush_then_close(self, mocker):
        """Both tasks end cancelled before the flush; the flush precedes close."""
        kv_task = asyncio.create_task(_never_ends())
        flush_task = asyncio.create_task(_never_ends())
        flush_mock = mocker.patch("buster_ripper.dump.flush_all")
        at_flush = []
        flush_mock.side_effect = lambda: at_flush.append(
            (kv_task.cancelled(), flush_task.cancelled())
        )
        session = mocker.AsyncMock()
        at_close = []
        session.close.side_effect = lambda: at_close.append(flush_mock.call_count)

        await on_cleanup({
            "kv_monitor_task": kv_task,
            "dump_flush_task": flush_task,
            "session": session,
        })

        assert at_flush == [(True, True)]
        assert at_close == [1]
        session.close.assert_called_once()

    async def test_handles_missing_dump_flush_task(self, mocker):
        """on_cleanup works when no flush task exists; the flush still runs."""
        flush_mock = mocker.patch("buster_ripper.dump.flush_all")
        session = mocker.AsyncMock()

        await on_cleanup({"session": session})

        flush_mock.assert_called_once_with()
        session.close.assert_called_once()

    async def test_shutdown_flush_failure_never_raises(self, mocker):
        """A raising shutdown flush is logged; cleanup still closes the session."""
        mocker.patch("buster_ripper.dump.flush_all", side_effect=OSError("disk full"))
        session = mocker.AsyncMock()

        await on_cleanup({"session": session})

        session.close.assert_called_once()

    @pytest.mark.parametrize("failed_key", ["kv_monitor_task", "dump_flush_task"])
    async def test_done_failed_task_does_not_skip_final_flush(
        self, pending_dump, failed_key
    ):
        """Ordinary task failure does not prevent the other cancel, flush, close."""
        root, sid, session = pending_dump
        failed = asyncio.create_task(_fails())
        await asyncio.sleep(0)
        other = asyncio.create_task(_never_ends())
        app = {failed_key: failed, "session": session}
        other_key = ({"kv_monitor_task", "dump_flush_task"} - {failed_key}).pop()
        app[other_key] = other
        session.close.side_effect = lambda: _assert_flushed(root, sid, other)

        await on_cleanup(app)

        session.close.assert_awaited_once()
        assert other.cancelled()
        assert dump._sessions[sid].pending == []

    async def test_outer_cleanup_cancellation_propagates(self, mocker):
        """A cancelled cleanup coroutine is not mistaken for a cancelled timer."""
        waiting = asyncio.Event()
        release = asyncio.Event()

        async def slow_cancel():
            try:
                await asyncio.Future()
            except asyncio.CancelledError:
                waiting.set()
                await release.wait()

        background = asyncio.create_task(slow_cancel())
        flush = mocker.patch("buster_ripper.dump.flush_all")
        session = mocker.AsyncMock()
        cleanup = asyncio.create_task(on_cleanup({"kv_monitor_task": background, "session": session}))
        await waiting.wait()
        cleanup.cancel()
        with pytest.raises(asyncio.CancelledError):
            await cleanup
        release.set()
        assert background.cancelled()
        flush.assert_not_called()
        session.close.assert_not_awaited()

    async def test_runner_cancel_count_does_not_skip_pending_dump(self, pending_dump):
        """aiohttp enters cleanup in its already-cancelled runner task."""
        root, sid, session = pending_dump
        kv = asyncio.create_task(_never_ends())
        timer = asyncio.create_task(_never_ends())

        async def runner():
            try:
                await asyncio.Future()
            except asyncio.CancelledError:
                assert asyncio.current_task().cancelling() == 1
                session.close.side_effect = lambda: _assert_flushed(root, sid, timer)
                await on_cleanup({"kv_monitor_task": kv, "dump_flush_task": timer, "session": session})

        task = asyncio.create_task(runner())
        await asyncio.sleep(0)
        task.cancel()
        await task
        assert kv.cancelled() and timer.cancelled()
        session.close.assert_awaited_once()
        assert dump._sessions[sid].pending == []

    async def test_new_cancel_during_runner_cleanup_propagates(self, mocker):
        """A second cancellation while awaiting a child is not child cancellation."""
        waiting = asyncio.Event()
        release = asyncio.Event()

        async def slow_cancel():
            try:
                await asyncio.Future()
            except asyncio.CancelledError:
                waiting.set()
                await release.wait()

        child = asyncio.create_task(slow_cancel())
        flush = mocker.patch("buster_ripper.dump.flush_all")
        session = mocker.AsyncMock()

        async def runner():
            try:
                await asyncio.Future()
            except asyncio.CancelledError:
                await on_cleanup({"kv_monitor_task": child, "session": session})

        task = asyncio.create_task(runner())
        await asyncio.sleep(0)
        task.cancel()
        await waiting.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        release.set()
        assert child.cancelled()
        flush.assert_not_called()
        session.close.assert_not_awaited()


def _assert_flushed(root, sid, other):
    """Observe a cancelled peer and durable payload before close."""
    assert other.cancelled()
    assert "pending" in (root / sid / "change-focused.log").read_text()


class TestOnCleanup:
    """Tests for on_cleanup function."""

    @pytest.mark.asyncio
    async def test_cancels_kv_monitor_task(self, mocker):
        """Verify on_cleanup cancels the KV monitor task."""
        # Create an async function that raises CancelledError
        async def raise_cancelled():
            raise asyncio.CancelledError()

        # Create a real asyncio task that will be cancelled
        loop = asyncio.get_event_loop()
        coro = raise_cancelled()
        task = loop.create_task(coro)

        mock_session = mocker.AsyncMock()
        app = {"kv_monitor_task": task, "session": mock_session}

        # Cancel before running on_cleanup to simulate the expected state
        task.cancel()

        await on_cleanup(app)

        mock_session.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_closes_session(self, mocker):
        """Verify on_cleanup closes the shared session."""
        mock_session = mocker.AsyncMock()
        app = {"session": mock_session}

        await on_cleanup(app)

        mock_session.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_handles_missing_kv_monitor_task(self, mocker):
        """Verify on_cleanup works when no KV monitor task exists."""
        mock_session = mocker.AsyncMock()
        app = {"session": mock_session}

        # Should not raise
        await on_cleanup(app)

        mock_session.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_handles_cancelled_error_gracefully(self, mocker):
        """Verify CancelledError is caught and doesn't propagate."""
        # Create an async function that raises CancelledError
        async def raise_cancelled():
            raise asyncio.CancelledError()

        loop = asyncio.get_event_loop()
        coro = raise_cancelled()
        task = loop.create_task(coro)

        mock_session = mocker.AsyncMock()
        app = {"kv_monitor_task": task, "session": mock_session}

        # Cancel the task
        task.cancel()

        # Should not raise CancelledError
        await on_cleanup(app)

        mock_session.close.assert_called_once()
