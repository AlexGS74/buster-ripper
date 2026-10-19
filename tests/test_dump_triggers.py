"""Tests for dump.py's flush triggers (T003): volume, timer, resilience.

The volume check lives at the collapsed append — after it, reading the same
pending_bytes field the append just wrote — so it can never fire on the
documented path. The timer is the periodic coroutine poll_flush (flush, then
sleep; per-iteration failures logged; cancellation landing at the sleep
await). on_cleanup's wiring is covered in test_server_hooks.py.
"""

import asyncio
import contextlib
import json
import logging

import pytest

from buster_ripper import config, dump
from buster_ripper.dump import dump_turn
from buster_ripper.session import session_id

COLLAPSED_FORMATS = ["change-focused", "self-contained"]
_DEFAULT_BYTES = 4 * 1024 * 1024
_DEFAULT_INTERVAL = 10


def _body(system: str, user_texts: list[str]) -> dict:
    """Build a request body with one user message per turn-contributing text."""
    return {
        "system": system,
        "messages": [{"role": "user", "content": t} for t in user_texts],
    }


def _turn_bodies(system: str, n_turns: int) -> list[dict]:
    """Build n_turns bodies whose renderings grow and differ each turn."""
    return [
        _body(system, [f"turn-{i} message {j}" for j in range(i + 1)])
        for i in range(n_turns)
    ]


@pytest.fixture()
def dump_env(tmp_path):
    """Point the dump at a tmp dir and reset module-global session state."""
    config.DUMP_DIR = tmp_path
    config.DUMP_FORMAT = "documented"
    config.DUMP_FLUSH_BYTES = _DEFAULT_BYTES
    config.DUMP_FLUSH_INTERVAL = _DEFAULT_INTERVAL
    dump._sessions.clear()
    yield tmp_path
    dump._sessions.clear()
    config.DUMP_DIR = None
    config.DUMP_FORMAT = "documented"
    config.DUMP_FLUSH_BYTES = _DEFAULT_BYTES
    config.DUMP_FLUSH_INTERVAL = _DEFAULT_INTERVAL


class TestVolumeTrigger:
    """G-005: reaching DUMP_FLUSH_BYTES flushes with no shutdown and no timer."""

    @pytest.mark.parametrize("fmt", COLLAPSED_FORMATS)
    def test_threshold_reached_flushes_at_append_time(self, dump_env, fmt):
        """Crossing the threshold puts the turn on disk immediately."""
        config.DUMP_FORMAT = fmt
        config.DUMP_FLUSH_BYTES = 1
        body = _body("sys-a", ["hello"])
        sid = session_id(body)
        dump_turn(body)

        assert (dump_env / sid / f"{fmt}.log").exists()
        assert dump._sessions[sid].pending == []

    def test_below_threshold_defers(self, dump_env):
        """Turns stay buffered while the running total stays under the bar."""
        config.DUMP_FORMAT = "change-focused"
        config.DUMP_FLUSH_BYTES = 10 ** 9
        for body in _turn_bodies("sys-a", 3):
            dump_turn(body)
        sid = session_id(_body("sys-a", []))

        assert list((dump_env / sid).glob("*.log")) == []
        assert len(dump._sessions[sid].pending) == 3

    def test_check_reads_accumulated_pending_bytes(self, dump_env):
        """The check reads the cumulative field the appends wrote, per session."""
        config.DUMP_FORMAT = "change-focused"
        bodies = _turn_bodies("sys-a", 3)
        sid = session_id(bodies[0])
        config.DUMP_FLUSH_BYTES = 10 ** 9
        dump_turn(bodies[0])
        # Far above one turn's bytes but below the eventual running total.
        config.DUMP_FLUSH_BYTES = dump._sessions[sid].pending_bytes + 10 ** 6
        dump_turn(bodies[1])

        assert not (dump_env / sid / "change-focused.log").exists()

        config.DUMP_FLUSH_BYTES = dump._sessions[sid].pending_bytes
        dump_turn(bodies[2])

        assert (dump_env / sid / "change-focused.log").exists()

    def test_boundary_equality_flushes(self, dump_env):
        """pending_bytes exactly equal to the threshold flushes (>=, not >)."""
        config.DUMP_FORMAT = "change-focused"
        bodies = _turn_bodies("sys-a", 2)
        sid = session_id(bodies[0])
        config.DUMP_FLUSH_BYTES = 10 ** 9
        dump_turn(bodies[0])
        config.DUMP_FLUSH_BYTES = (
            dump._sessions[sid].pending_bytes + len(dump._render_body(bodies[1]))
        )
        dump_turn(bodies[1])

        assert (dump_env / sid / "change-focused.log").exists()

    @pytest.mark.parametrize("fmt", COLLAPSED_FORMATS)
    def test_multibyte_crosses_inclusive_byte_threshold(self, dump_env, fmt, mocker):
        """A rendered non-ASCII turn flushes before timer/shutdown at byte boundary."""
        config.DUMP_FORMAT = fmt
        body = _body("sys-a", ["é" * 1000])
        rendered = dump._render_body(body)
        assert len(rendered) < len(rendered.encode("utf-8"))
        config.DUMP_FLUSH_BYTES = len(rendered.encode("utf-8"))
        write = mocker.spy(dump.Path, "open")

        dump_turn(body)

        sid = session_id(body)
        artifact = dump_env / sid / f"{fmt}.log"
        assert artifact.is_file()
        assert write.call_count == 1
        assert "é" * 1000 in artifact.read_text()
        assert dump._sessions[sid].pending_bytes == 0

    @pytest.mark.parametrize("fmt", COLLAPSED_FORMATS)
    def test_multibyte_below_threshold_remains_pending(self, dump_env, fmt):
        """Strictly below encoded-byte threshold, no write occurs."""
        config.DUMP_FORMAT = fmt
        body = _body("sys-a", ["é" * 1000])
        expected = len(dump._render_body(body).encode("utf-8"))
        config.DUMP_FLUSH_BYTES = expected + 1

        dump_turn(body)

        sid = session_id(body)
        assert dump._sessions[sid].pending_bytes == expected
        assert not (dump_env / sid / f"{fmt}.log").exists()

    @pytest.mark.parametrize("fmt", COLLAPSED_FORMATS)
    def test_unencodable_rendering_does_not_poison_next_turn(self, dump_env, fmt):
        """A JSON-valid lone surrogate cannot partially append a collapsed turn."""
        config.DUMP_FORMAT = fmt
        bad = _body("sys-a", [json.loads(r'"\ud800"')])
        good = _body("sys-a", ["valid"])
        sid = session_id(bad)
        config.DUMP_FLUSH_BYTES = 10 ** 9

        with pytest.raises(UnicodeEncodeError):
            dump_turn(bad)

        state = dump._sessions[sid]
        assert state.pending == []
        assert state.pending_bytes == 0
        assert state.turn == 0
        assert state.prev_text == ""
        dump_turn(good)
        expected = len(dump._render_body(good).encode("utf-8"))
        assert state.turn == 1
        assert state.pending_bytes == expected
        assert len(state.pending) == 1
        assert state.pending[0].startswith("turn 000")
        record = state.pending[0]
        dump.flush_all()
        artifact = (dump_env / sid / f"{fmt}.log").read_text()
        assert artifact == record + "\n"
        assert "valid" in artifact
        assert "\ud800" not in artifact
        assert state.pending == []
        assert state.pending_bytes == 0

    def test_volume_check_never_fires_on_documented_path(self, dump_env):
        """A tiny threshold plus documented turns: no .log, the three files live."""
        config.DUMP_FLUSH_BYTES = 1
        for body in _turn_bodies("sys-a", 3):
            dump_turn(body)
        sid = session_id(_body("sys-a", []))

        assert list((dump_env / sid).glob("*.log")) == []
        assert sorted(p.name for p in (dump_env / sid).iterdir()) == [
            "index.txt", "turn_000.json", "turn_001.diff", "turn_002.diff",
        ]


class TestPollFlush:
    """G-004: the periodic coroutine drains the buffer while the process runs."""

    async def test_tick_flushes_pending_without_shutdown(self, dump_env):
        config.DUMP_FORMAT = "change-focused"
        config.DUMP_FLUSH_INTERVAL = 0
        body = _body("sys-a", ["hello"])
        sid = session_id(body)
        dump_turn(body)

        task = asyncio.create_task(dump.poll_flush())
        for _ in range(3):
            await asyncio.sleep(0)
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task

        assert (dump_env / sid / "change-focused.log").exists()
        assert dump._sessions[sid].pending == []

    async def test_raising_flush_is_logged_and_loop_continues(
        self, dump_env, mocker, caplog
    ):
        """A flush that raises inside the loop is logged; later ticks flush."""
        config.DUMP_FORMAT = "change-focused"
        config.DUMP_FLUSH_INTERVAL = 0
        body = _body("sys-a", ["hello"])
        sid = session_id(body)
        dump_turn(body)
        real_flush = dump.flush_all
        calls = {"n": 0}

        def flaky_flush():
            calls["n"] += 1
            if calls["n"] == 1:
                raise RuntimeError("boom")
            real_flush()

        mocker.patch.object(dump, "flush_all", flaky_flush)
        caplog.set_level(logging.ERROR, logger="buster-ripper")

        task = asyncio.create_task(dump.poll_flush())
        for _ in range(6):
            await asyncio.sleep(0)
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task

        assert calls["n"] >= 2
        assert (dump_env / sid / "change-focused.log").exists()
        assert any("periodic flush failed" in r.message for r in caplog.records)

    async def test_cancelled_timer_stops_cleanly(self, dump_env):
        """Cancellation lands at the sleep await; the task ends cancelled."""
        config.DUMP_FLUSH_INTERVAL = 10
        task = asyncio.create_task(dump.poll_flush())
        await asyncio.sleep(0)

        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task

        assert task.cancelled()

    async def test_timer_noop_on_documented_path(self, dump_env):
        """Documented turns leave nothing pending; ticks write no artifact."""
        config.DUMP_FLUSH_INTERVAL = 0
        for body in _turn_bodies("sys-a", 2):
            dump_turn(body)
        sid = session_id(_body("sys-a", []))

        task = asyncio.create_task(dump.poll_flush())
        for _ in range(3):
            await asyncio.sleep(0)
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task

        assert list((dump_env / sid).glob("*.log")) == []


class TestFlushUsesBufferTimeFormat:
    """QB:dump-flush/format-reread — the flush files under the buffer-time format."""

    def test_midrun_format_change_does_not_misfile_records(self, dump_env):
        """Records appended under change-focused flush to change-focused.log."""
        config.DUMP_FORMAT = "change-focused"
        bodies = _turn_bodies("sys-a", 2)
        sid = session_id(bodies[0])
        for body in bodies:
            dump_turn(body)

        config.DUMP_FORMAT = "self-contained"
        dump.flush_all()

        assert (dump_env / sid / "change-focused.log").exists()
        assert not (dump_env / sid / "self-contained.log").exists()
