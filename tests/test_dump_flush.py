"""Tests for dump.py's flush_all and its durable-writer guards (T002).

Covers the bulk flush (one artifact per session named by the configured
format, records in arrival order, buffer cleared, diff chain untouched, all
no-op shapes), the R2 symlink/non-regular refusal on every write target, and
the R5 artifact identity — one artifact per (sid, format), never merged.
"""

import pytest

from buster_ripper import config, dump
from buster_ripper.dump import SessionState, dump_turn, flush_all
from buster_ripper.session import session_id

COLLAPSED_FORMATS = ["change-focused", "self-contained"]


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
    dump._sessions.clear()
    yield tmp_path
    dump._sessions.clear()
    config.DUMP_DIR = None
    config.DUMP_FORMAT = "documented"


class TestFlushAll:
    """flush_all appends each session's pending batch as ONE write."""

    @pytest.mark.parametrize("fmt", COLLAPSED_FORMATS)
    def test_flush_writes_one_artifact_named_by_format(self, dump_env, fmt):
        """After a flush the session dir holds exactly <format>.log."""
        config.DUMP_FORMAT = fmt
        bodies = _turn_bodies("sys-a", 3)
        sid = session_id(bodies[0])
        for body in bodies:
            dump_turn(body)
        flush_all()

        session_dir = dump_env / sid
        assert [p.name for p in session_dir.iterdir()] == [f"{fmt}.log"]

    @pytest.mark.parametrize("fmt", COLLAPSED_FORMATS)
    def test_artifact_carries_records_in_arrival_order(self, dump_env, fmt):
        """The artifact concatenates the pending records in arrival order."""
        config.DUMP_FORMAT = fmt
        bodies = _turn_bodies("sys-a", 3)
        sid = session_id(bodies[0])
        for body in bodies:
            dump_turn(body)
        state = dump._sessions[sid]
        before = list(state.pending)
        flush_all()

        artifact = (dump_env / sid / f"{fmt}.log").read_text()
        assert artifact == "\n\n".join(before) + "\n"
        assert artifact.index("turn 000") < artifact.index("turn 001")
        assert artifact.index("turn 001") < artifact.index("turn 002")

    @pytest.mark.parametrize("fmt", COLLAPSED_FORMATS)
    def test_flush_clears_buffer_and_never_touches_chain(self, dump_env, fmt):
        """Flush clears pending/pending_bytes; turn/prev_text are untouched."""
        config.DUMP_FORMAT = fmt
        bodies = _turn_bodies("sys-a", 3)
        sid = session_id(bodies[0])
        for body in bodies:
            dump_turn(body)
        state = dump._sessions[sid]
        turn_before = state.turn
        prev_before = state.prev_text
        flush_all()

        assert state.pending == []
        assert state.pending_bytes == 0
        assert state.turn == turn_before
        assert state.prev_text == prev_before

    def test_flush_noop_when_dump_dir_none(self, tmp_path):
        """With DUMP_DIR unset flush_all returns without raising."""
        config.DUMP_DIR = None
        config.DUMP_FORMAT = "change-focused"
        dump._sessions.clear()
        dump._sessions["sid"] = SessionState(
            session_dir=tmp_path, pending=["r0"], pending_bytes=10
        )
        try:
            flush_all()  # must not raise
            assert dump._sessions["sid"].pending == ["r0"]
        finally:
            dump._sessions.clear()
            config.DUMP_FORMAT = "documented"

    def test_flush_noop_when_nothing_pending(self, dump_env):
        """With nothing pending flush_all creates no artifact."""
        config.DUMP_FORMAT = "change-focused"
        bodies = _turn_bodies("sys-a", 2)
        sid = session_id(bodies[0])
        for body in bodies:
            dump_turn(body)
        flush_all()
        dump._sessions.clear()
        flush_all()

        assert list((dump_env / sid).glob("*.log")) == [
            dump_env / sid / "change-focused.log"
        ]

    def test_flush_never_raises_on_write_failure(self, dump_env, mocker):
        """A failing write is logged, not raised; the batch is retained."""
        config.DUMP_FORMAT = "change-focused"
        for body in _turn_bodies("sys-a", 2):
            dump_turn(body)
        mocker.patch("pathlib.Path.open", side_effect=OSError("disk full"))

        flush_all()  # must not raise

        state = next(iter(dump._sessions.values()))
        assert len(state.pending) == 2
        assert state.pending_bytes > 0

    def test_flush_skips_batch_when_format_is_documented(self, dump_env):
        """A pending batch under the documented format flushes to no .log."""
        bodies = _turn_bodies("sys-a", 2)
        sid = session_id(bodies[0])
        dump._sessions[sid] = SessionState(
            session_dir=dump_env / sid, pending=["r0"], pending_bytes=5
        )
        (dump_env / sid).mkdir(parents=True)

        flush_all()

        assert list((dump_env / sid).glob("*.log")) == []
        assert dump._sessions[sid].pending == ["r0"]

    def test_documented_run_then_flush_creates_no_artifact(self, dump_env):
        """flush_all is a no-op after documented turns (nothing pending)."""
        for body in _turn_bodies("sys-a", 2):
            dump_turn(body)
        flush_all()

        session_dir = dump_env / session_id(_body("sys-a", []))
        assert list(session_dir.glob("*.log")) == []


class TestDurableWriterGuards:
    """R2: symlinked/non-regular write targets are refused, never followed."""

    def test_flush_refuses_symlinked_artifact(self, dump_env, tmp_path_factory):
        """A symlink at <format>.log is skipped; the target file is not written."""
        config.DUMP_FORMAT = "change-focused"
        bodies = _turn_bodies("sys-a", 2)
        sid = session_id(bodies[0])
        outside = tmp_path_factory.mktemp("outside") / "captured.log"
        session_dir = dump_env / sid
        session_dir.mkdir(parents=True)
        (session_dir / "change-focused.log").symlink_to(outside)
        for body in bodies:
            dump_turn(body)

        flush_all()  # must not raise

        assert not outside.exists()
        state = dump._sessions[sid]
        assert len(state.pending) == 2

    def test_flush_refuses_dangling_symlink_artifact(self, dump_env):
        """A dangling symlink target is rejected, not created through."""
        config.DUMP_FORMAT = "self-contained"
        bodies = _turn_bodies("sys-a", 1)
        sid = session_id(bodies[0])
        session_dir = dump_env / sid
        session_dir.mkdir(parents=True)
        (session_dir / "self-contained.log").symlink_to(dump_env / "nowhere.log")
        for body in bodies:
            dump_turn(body)

        flush_all()

        assert not (dump_env / "nowhere.log").exists()
        assert len(dump._sessions[sid].pending) == 1

    def test_documented_writer_refuses_symlinked_turn_file(self, dump_env):
        """A symlink at turn_001.diff is refused; the target is not written."""
        bodies = _turn_bodies("sys-a", 2)
        sid = session_id(bodies[0])
        dump_turn(bodies[0])
        session_dir = dump_env / sid
        outside = dump_env / "outside.diff"
        (session_dir / "turn_001.diff").symlink_to(outside)

        with pytest.raises(ValueError, match="non-regular target"):
            dump_turn(bodies[1])

        assert not outside.exists()


def _run_and_flush(fmt: str) -> str:
    """Run two turns under fmt in a fresh state, flush, return the session id."""
    config.DUMP_FORMAT = fmt
    dump._sessions.clear()
    bodies = _turn_bodies("sys-a", 2)
    sid = session_id(bodies[0])
    for body in bodies:
        dump_turn(body)
    flush_all()
    return sid


class TestArtifactIdentity:
    """R5: the artifact identity key is (sid, format) — formats never merge."""

    def test_two_formats_never_merge_artifacts(self, dump_env):
        """Each format owns its own artifact file in the session dir."""
        sid = _run_and_flush("change-focused")
        _run_and_flush("self-contained")

        session_dir = dump_env / sid
        names = sorted(p.name for p in session_dir.iterdir())
        assert names == ["change-focused.log", "self-contained.log"]
        assert "turn 001 rendered" in (
            session_dir / "self-contained.log"
        ).read_text()
        assert "turn 001 diff" in (session_dir / "change-focused.log").read_text()


class TestSessionIsolationAtFlush:
    """VP-T001-session-isolation: two sids flush to two artifacts, independently."""

    @pytest.mark.parametrize("fmt", COLLAPSED_FORMATS)
    def test_two_sessions_flush_independently(self, dump_env, fmt):
        """Each session's artifact carries exactly its own records, in order."""
        config.DUMP_FORMAT = fmt
        bodies_a = _turn_bodies("sys-a", 2)
        bodies_b = _turn_bodies("sys-b", 3)
        for body in bodies_a + bodies_b:
            dump_turn(body)
        sid_a, sid_b = session_id(bodies_a[0]), session_id(bodies_b[0])
        batch_a = list(dump._sessions[sid_a].pending)
        batch_b = list(dump._sessions[sid_b].pending)

        flush_all()

        assert [p.name for p in (dump_env / sid_a).iterdir()] == [f"{fmt}.log"]
        assert [p.name for p in (dump_env / sid_b).iterdir()] == [f"{fmt}.log"]
        artifact_a = (dump_env / sid_a / f"{fmt}.log").read_text()
        artifact_b = (dump_env / sid_b / f"{fmt}.log").read_text()
        assert artifact_a == "\n\n".join(batch_a) + "\n"
        assert artifact_b == "\n\n".join(batch_b) + "\n"
        assert dump._sessions[sid_a].pending == []
        assert dump._sessions[sid_b].pending == []
