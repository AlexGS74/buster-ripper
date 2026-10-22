"""EMA smoothing witnesses for poll_kv_cache."""

import asyncio
import contextlib

import aiohttp
import pytest

from buster_ripper import compaction, config


# ── EMA smoothing tests ───────────────────────────────────────────────────────

class _TickSeam:
    """``session.get`` stand-in handing out one ``/metrics`` answer per tick.

    The value is read at the tick boundary, not after the run, so a witness fails on
    a *value*; one answer per tick makes the awaited ``text()`` countable."""

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
    """A 200 ``/metrics`` body, enterable with ``async with``, reporting one reading.

    Only this shape works: an ``AsyncMock`` whose ``__aenter__``/``__aexit__`` bind
    back to itself, from a plain ``Mock`` on a spec'd session — a bare ``Mock`` cannot
    be entered, a spec'd response prunes the dunders, a ``MagicMock`` discards them."""
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

    Returns the value after each tick and the per-tick answers; the loop's interval is
    set to 0 for the task's duration and restored afterwards, because
    ``asyncio.sleep(0)`` only guarantees another *ready* task runs."""
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
    """The name's promise, through the loop: the first reading lands raw, not inline.

    Before this pair ran ``poll_kv_cache`` at all, so ``compaction.py:49`` stayed missing."""
    values, answers = await _drive_ticks(mocker, [0.85])

    assert values == [0.85], f"tick 1 did not produce the raw reading: {values!r}"
    assert answers[0].text.await_count == 1, "the awaited channel never delivered a body"


async def test_ema_smoothing_reduces_bounces(mocker):
    """Three distinct raw readings, and the blend is production's, not the test's.

    The expectations are literals for α = 0.3: re-running the formula in the source's
    operand order cannot fail when the two weights are swapped — the mutation this
    witness exists to catch."""
    readings = [0.75, 0.0, 0.23]
    values, answers = await _drive_ticks(mocker, readings)

    assert compaction._KV_EMA_ALPHA == 0.3, "the literals below are this α's"
    assert values == [0.75, pytest.approx(0.525), pytest.approx(0.4365)], (
        f"the blend is not (1-α)·prev + α·raw over the three ticks: {values!r}")
    assert [a.text.await_count for a in answers] == [1, 1, 1]
    assert abs(values[2] - values[1]) < abs(readings[2] - readings[1]), "nothing smoothed"
