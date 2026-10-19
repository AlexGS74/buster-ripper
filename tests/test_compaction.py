"""Tests for compaction module with behavioral session testing."""

import asyncio
import contextlib

import aiohttp
import pytest

from buster_ripper import compaction, config
from buster_ripper.compaction import compact_token_count, should_nudge_compact, CompactionConfig


# ── should_nudge_compact ──────────────────────────────────────────────────────

def test_nudge_when_both_thresholds_exceeded():
    """Nudge when both token and latency thresholds exceeded."""
    cfg = CompactionConfig(
        max_model_len=200_000,
        token_ratio=0.75,
        latency_threshold_ms=15_000,
    )
    assert should_nudge_compact(
        input_tokens=160_000,
        avg_latency_ms=20_000,
        kv_cache_usage=0.0,
        cfg=cfg,
    )


def test_no_nudge_when_only_tokens_high():
    """No nudge when only token threshold is high."""
    cfg = CompactionConfig(
        max_model_len=200_000,
        token_ratio=0.75,
        latency_threshold_ms=15_000,
    )
    assert not should_nudge_compact(
        input_tokens=160_000,
        avg_latency_ms=5_000,
        kv_cache_usage=0.0,
        cfg=cfg,
    )


def test_no_nudge_when_only_latency_high():
    """No nudge when only latency threshold is high."""
    cfg = CompactionConfig(
        max_model_len=200_000,
        token_ratio=0.75,
        latency_threshold_ms=15_000,
    )
    assert not should_nudge_compact(
        input_tokens=50_000,
        avg_latency_ms=20_000,
        kv_cache_usage=0.0,
        cfg=cfg,
    )


def test_no_nudge_when_both_below():
    """No nudge when both below thresholds."""
    cfg = CompactionConfig(
        max_model_len=200_000,
        token_ratio=0.75,
        latency_threshold_ms=15_000,
    )
    assert not should_nudge_compact(
        input_tokens=50_000,
        avg_latency_ms=5_000,
        kv_cache_usage=0.0,
        cfg=cfg,
    )


def test_nudge_at_exact_token_threshold():
    """Nudge at exact token threshold."""
    cfg = CompactionConfig(
        max_model_len=200_000,
        token_ratio=0.75,
        latency_threshold_ms=15_000,
    )
    assert should_nudge_compact(
        input_tokens=150_000,
        avg_latency_ms=20_000,
        kv_cache_usage=0.0,
        cfg=cfg,
    )


def test_no_nudge_one_token_below_threshold():
    """No nudge one token below threshold."""
    cfg = CompactionConfig(
        max_model_len=200_000,
        token_ratio=0.75,
        latency_threshold_ms=15_000,
    )
    assert not should_nudge_compact(
        input_tokens=149_999,
        avg_latency_ms=20_000,
        kv_cache_usage=0.0,
        cfg=cfg,
    )


def test_nudge_on_kv_pressure_alone():
    """Nudge on KV cache pressure alone."""
    cfg = CompactionConfig(
        max_model_len=200_000,
        token_ratio=0.75,
        latency_threshold_ms=15_000,
        kv_ratio=0.75,
    )
    assert should_nudge_compact(
        input_tokens=10_000,
        avg_latency_ms=100,
        kv_cache_usage=0.80,
        cfg=cfg,
    )


def test_no_nudge_kv_below_threshold():
    """No nudge when KV below threshold."""
    cfg = CompactionConfig(
        max_model_len=200_000,
        token_ratio=0.75,
        latency_threshold_ms=15_000,
        kv_ratio=0.75,
    )
    assert not should_nudge_compact(
        input_tokens=10_000,
        avg_latency_ms=100,
        kv_cache_usage=0.74,
        cfg=cfg,
    )


def test_nudge_kv_at_exact_threshold():
    """Nudge when KV at exact threshold."""
    cfg = CompactionConfig(
        max_model_len=200_000,
        token_ratio=0.75,
        latency_threshold_ms=15_000,
        kv_ratio=0.75,
    )
    assert should_nudge_compact(
        input_tokens=10_000,
        avg_latency_ms=100,
        kv_cache_usage=0.75,
        cfg=cfg,
    )


def test_kv_ratio_1_0_disables_kv_trigger():
    """KV ratio 1.0 disables KV trigger."""
    cfg = CompactionConfig(
        max_model_len=200_000,
        token_ratio=0.75,
        latency_threshold_ms=15_000,
        kv_ratio=1.0,
    )
    assert not should_nudge_compact(
        input_tokens=10_000,
        avg_latency_ms=100,
        kv_cache_usage=0.99,
        cfg=cfg,
    )


# ── compact_token_count ───────────────────────────────────────────────────────

def test_compact_token_count_value():
    """Verify compact_token_count returns correct value."""
    assert compact_token_count(200_000, 0.95) == 190_000


def test_compact_token_count_is_int():
    """Verify compact_token_count returns int."""
    assert isinstance(compact_token_count(200_000, 0.95), int)


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


# ── EMA smoothing tests ───────────────────────────────────────────────────────

class _TickSeam:
    """``session.get`` stand-in that hands out one ``/metrics`` answer per tick.

    The value the loop computed is read at the tick boundary, not after the run: the
    call happens even when ``async with`` bound a discarded child (``claim-409``), so a
    witness built on this seam fails on a *value* rather than staying green on a call
    count. One answer object per tick is what makes the awaited ``text()`` countable.
    """

    def __init__(self, answers):
        self.answers = list(answers)
        self.observed = []
        self.task = None

    def get(self, url, **kwargs):
        self.observed.append(compaction._kv_cache_usage)
        if not self.answers:
            self.task.cancel()
            raise aiohttp.ClientError("no further tick to drive")
        return self.answers.pop(0)


def _kv_answer(mocker, raw: float):
    """A 200 ``/metrics`` body, enterable with ``async with``, reporting one usage reading.

    This module's own sibling idiom, and the only permitted shape here: an ``AsyncMock``
    response with ``__aenter__``/``__aexit__`` bound back to itself, returned by a plain
    ``Mock`` on a spec'd session. A bare ``Mock()``/``NonCallableMock()`` cannot be
    entered, a spec'd response double prunes the dunders, and an unconfigured
    ``MagicMock`` silently discards everything configured on it — each of those fails
    inside ``compaction.py:55-56``'s swallow, leaving a witness green with ``:49`` never
    executed.
    """
    resp = mocker.AsyncMock()
    resp.status = 200
    resp.text = mocker.AsyncMock(
        return_value=f'vllm:kv_cache_usage_perc{{engine="0"}} {raw}\n')
    resp.__aenter__ = mocker.AsyncMock(return_value=resp)
    resp.__aexit__ = mocker.AsyncMock(return_value=False)
    return resp


async def _settle(task) -> None:
    """Await the driven loop; the seam's cancel on the extra tick is how the run ends."""
    with contextlib.suppress(asyncio.CancelledError):
        await asyncio.wait_for(task, timeout=2.0)


async def _drive_ticks(mocker, readings):
    """Run production's ``poll_kv_cache`` for one tick per reading.

    Returns the value after each tick and the per-tick answers. The interval the loop
    sleeps on is its own (``config.KV_POLL_INTERVAL``, read per iteration at ``:57``):
    it is set to 0 for the task's duration and restored with the global the loop writes,
    because ``asyncio.sleep(0)`` only guarantees another *ready* task runs — a witness is
    entitled to a second iteration by shortening the interval, never by treating a yield
    as a clock. A timeout here means the loop never reached its next tick at all.
    """
    answers = [_kv_answer(mocker, raw) for raw in readings]
    seam = _TickSeam(answers)
    untouched, interval = compaction._kv_cache_usage, config.KV_POLL_INTERVAL
    session = mocker.Mock(spec=aiohttp.ClientSession)
    session.get = mocker.Mock(side_effect=seam.get)
    compaction._kv_cache_usage = 0.0
    config.KV_POLL_INTERVAL = 0
    task = asyncio.create_task(compaction.poll_kv_cache(session))
    seam.task = task
    try:
        await _settle(task)
    finally:
        compaction._kv_cache_usage, config.KV_POLL_INTERVAL = untouched, interval
    return seam.observed[1:], answers


async def test_ema_first_reading_is_raw(mocker):
    """The name's promise, through the loop: the first reading lands raw.

    Before this pair ran ``poll_kv_cache`` at all — they computed the formula inline and
    asserted on their own arithmetic, so ``compaction.py:49`` stayed a missing line and
    the module globals never moved.
    """
    values, answers = await _drive_ticks(mocker, [0.85])

    assert values == [0.85], f"tick 1 did not produce the raw reading: {values!r}"
    assert answers[0].text.await_count == 1, "the awaited channel never delivered a body"


async def test_ema_smoothing_reduces_bounces(mocker):
    """Three distinct raw readings, and the blend is production's, not the test's.

    The expectations are literals for α = 0.3: an assertion that re-runs
    ``(1 - α) * prev + α * raw`` in the source's operand order cannot fail when the two
    weights are swapped, which is the mutation this witness exists to catch. The bouncy
    sequence is the one the docstring at ``compaction.py:25-27`` describes.
    """
    readings = [0.75, 0.0, 0.23]
    values, answers = await _drive_ticks(mocker, readings)

    assert compaction._KV_EMA_ALPHA == 0.3, "the literals below are this α's"
    assert values == [0.75, pytest.approx(0.525), pytest.approx(0.4365)], (
        f"the blend is not (1-α)·prev + α·raw over the three ticks: {values!r}")
    assert [a.text.await_count for a in answers] == [1, 1, 1]
    assert abs(values[2] - values[1]) < abs(readings[2] - readings[1]), "nothing smoothed"
