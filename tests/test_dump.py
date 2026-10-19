"""Tests for dump.py — format branch and collapsed buffering (T002/T004).

Covers the format dispatch (documented default, closed set, DUMP_DIR guard),
the collapsed buffering (records in arrival order, FK-B rendered-text byte
measure, writes deferred), and the documented path preserved byte-for-byte.
Record shapes and their artifact witnesses live in test_dump_records.py; the
flush and its guards in test_dump_flush.py; triggers in test_dump_triggers.py;
the instrumented write-count measurement in test_dump_write_count.py.
"""

import difflib
import json
import re

import pytest

from buster_ripper import config, dump
from buster_ripper.dump import _render_body, dump_turn
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


def _expected_diff(turn: int, prev_text: str, curr_text: str) -> str:
    """Independently recompute the documented turn diff text."""
    diff_lines = difflib.unified_diff(
        prev_text.splitlines(keepends=True),
        curr_text.splitlines(keepends=True),
        fromfile=f"turn_{turn - 1:03d}",
        tofile=f"turn_{turn:03d}",
        lineterm="\n",
    )
    return "".join(diff_lines)


def _expected_diff_counts(diff_text: str) -> tuple[int, int]:
    """Independently recompute (+added, -removed) from diff text."""
    lines = diff_text.splitlines()
    added = sum(1 for l in lines if l.startswith("+") and not l.startswith("+++"))
    removed = sum(1 for l in lines if l.startswith("-") and not l.startswith("---"))
    return added, removed


def _block_shape_body() -> dict:
    """Body exercising the system-as-list, tools and content-as-list shapes."""
    return {
        "system": [{"type": "text", "text": "shape system"}],
        "tools": [{"name": "grep"}],
        "messages": [
            {"role": "user", "content": [{"type": "text", "text": "block msg"}]},
        ],
    }


def _plain_shape_body() -> dict:
    """Body exercising the plain-string system and content shapes."""
    return {
        "system": "shape system",
        "messages": [
            {"role": "user", "content": "plain msg"},
            {"role": "assistant", "content": "reply"},
        ],
    }


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


class TestFormatDispatch:
    """The format branch selects which writer runs; the set is closed."""

    def test_default_format_is_documented(self, dump_env):
        """With no format configured the documented layout is produced."""
        bodies = _turn_bodies("sys-a", 2)
        sid = session_id(bodies[0])
        for body in bodies:
            dump_turn(body)

        session_dir = dump_env / sid
        assert (session_dir / "turn_000.json").is_file()
        assert (session_dir / "turn_001.diff").is_file()
        assert (session_dir / "index.txt").is_file()
        assert list(session_dir.glob("*.log")) == []

    def test_unknown_format_raises_and_writes_nothing(self, dump_env):
        """A value outside DUMP_FORMATS is a hard error with no fs effect."""
        config.DUMP_FORMAT = "jsonl"
        with pytest.raises(ValueError, match="unknown dump format"):
            dump_turn(_body("sys-a", ["m0"]))
        assert list(dump_env.iterdir()) == []

    def test_dump_dir_none_raises_value_error(self):
        """DUMP_DIR unset is a real guard, not a bare assert (F-5)."""
        config.DUMP_DIR = None
        config.DUMP_FORMAT = "documented"
        dump._sessions.clear()
        try:
            with pytest.raises(ValueError, match="DUMP_DIR"):
                dump_turn(_body("sys-a", ["m0"]))
        finally:
            dump._sessions.clear()
            config.DUMP_FORMAT = "documented"


class TestDocumentedFormatPreserved:
    """The documented writer stays byte-for-byte unchanged (G-008(a))."""

    @pytest.mark.parametrize("fmt", [None, "documented"])
    def test_files_names_and_content_unchanged(self, dump_env, fmt):
        """N turns produce today's three files with today's content."""
        if fmt is not None:
            config.DUMP_FORMAT = fmt
        bodies = _turn_bodies("sys-a", 3)
        sid = session_id(bodies[0])
        for body in bodies:
            dump_turn(body)

        session_dir = dump_env / sid
        rendered = [_render_body(b) for b in bodies]

        assert (session_dir / "turn_000.json").read_text() == json.dumps(
            bodies[0], indent=2, ensure_ascii=False
        )
        assert (session_dir / "turn_001.diff").read_text() == _expected_diff(
            1, rendered[0], rendered[1]
        )
        assert (session_dir / "turn_002.diff").read_text() == _expected_diff(
            2, rendered[1], rendered[2]
        )
        assert list(session_dir.glob("*.log")) == []

    def test_diff_control_lines_are_terminated(self, dump_env):
        """turn_001.diff carries each ---/+++/@@ control line on its own line."""
        bodies = _turn_bodies("sys-a", 2)
        sid = session_id(bodies[0])
        for body in bodies:
            dump_turn(body)

        lines = (dump_env / sid / "turn_001.diff").read_text().splitlines(keepends=True)
        assert lines[0] == "--- turn_000\n"
        assert lines[1] == "+++ turn_001\n"
        hunk = next(ln for ln in lines if ln.startswith("@@"))
        assert hunk.startswith("@@ ") and hunk.endswith(" @@\n"), hunk

    def test_index_line_format_unchanged(self, dump_env):
        """index.txt carries today's FULL and +added/-removed line formats."""
        bodies = _turn_bodies("sys-a", 2)
        sid = session_id(bodies[0])
        for body in bodies:
            dump_turn(body)

        lines = (dump_env / sid / "index.txt").read_text().splitlines()
        assert len(lines) == 2
        assert re.fullmatch(
            r"turn 000  \d\d:\d\d:\d\d  FULL  1 msgs  0 tools  \d+\.\d KB", lines[0]
        )
        diff_text = _expected_diff(1, _render_body(bodies[0]), _render_body(bodies[1]))
        added, removed = _expected_diff_counts(diff_text)
        assert re.fullmatch(
            rf"turn 001  \d\d:\d\d:\d\d  \+{added}/-{removed} lines  2 msgs  0 tools",
            lines[1],
        )


class TestCollapsedBuffering:
    """A collapsed format buffers records; every write is deferred (F-21)."""

    @pytest.mark.parametrize("fmt", COLLAPSED_FORMATS)
    @pytest.mark.parametrize("n_turns", [1, 2, 3, 5])
    def test_buffers_one_record_per_turn(self, dump_env, fmt, n_turns):
        """After N turns the session holds exactly N pending records."""
        config.DUMP_FORMAT = fmt
        bodies = _turn_bodies("sys-a", n_turns)
        for body in bodies:
            dump_turn(body)

        state = dump._sessions[session_id(bodies[0])]
        assert len(state.pending) == n_turns

    @pytest.mark.parametrize("fmt", COLLAPSED_FORMATS)
    @pytest.mark.parametrize("n_turns", [1, 3])
    def test_records_arrive_in_order(self, dump_env, fmt, n_turns):
        """Pending records state their turn numbers in arrival order."""
        config.DUMP_FORMAT = fmt
        bodies = _turn_bodies("sys-a", n_turns)
        for body in bodies:
            dump_turn(body)

        state = dump._sessions[session_id(bodies[0])]
        for i, record in enumerate(state.pending):
            assert record.splitlines()[0].startswith(f"turn {i:03d}")

    @pytest.mark.parametrize("fmt", COLLAPSED_FORMATS)
    @pytest.mark.parametrize("n_turns", [1, 2, 5])
    def test_pending_bytes_measures_rendered_text(self, dump_env, fmt, n_turns):
        """pending_bytes accumulates rendered-text UTF-8 bytes, the FK-B measure.

        The record bytes diverge from this measure by design (records carry
        summary lines and headers), so bytes are pinned to the renderings.
        """
        config.DUMP_FORMAT = fmt
        bodies = _turn_bodies("sys-a", n_turns)
        for body in bodies:
            dump_turn(body)

        state = dump._sessions[session_id(bodies[0])]
        assert state.pending_bytes == sum(len(_render_body(b).encode("utf-8")) for b in bodies)
        assert state.pending_bytes != sum(len(r) for r in state.pending)

    @pytest.mark.parametrize("fmt", COLLAPSED_FORMATS)
    def test_no_artifact_before_flush(self, dump_env, fmt):
        """Before any flush no file exists — every write is deferred."""
        config.DUMP_FORMAT = fmt
        bodies = _turn_bodies("sys-a", 3)
        sid = session_id(bodies[0])
        for body in bodies:
            dump_turn(body)

        session_dir = dump_env / sid
        assert session_dir.is_dir()
        assert list(session_dir.iterdir()) == []

    @pytest.mark.parametrize("fmt", COLLAPSED_FORMATS)
    def test_documented_files_never_written_on_collapsed_path(self, dump_env, fmt):
        """The collapsed path produces no turn files and no index.txt."""
        config.DUMP_FORMAT = fmt
        bodies = _turn_bodies("sys-a", 3)
        sid = session_id(bodies[0])
        for body in bodies:
            dump_turn(body)

        session_dir = dump_env / sid
        assert not (session_dir / "turn_000.json").exists()
        assert not (session_dir / "turn_001.diff").exists()
        assert not (session_dir / "index.txt").exists()

    def test_documented_path_buffers_nothing(self, dump_env):
        """On the documented path the turn is written and nothing is buffered."""
        bodies = _turn_bodies("sys-a", 3)
        sid = session_id(bodies[0])
        for body in bodies:
            dump_turn(body)
            state = dump._sessions[sid]
            assert state.pending == []
            assert state.pending_bytes == 0

    @pytest.mark.parametrize("fmt", COLLAPSED_FORMATS)
    def test_accumulates_across_rendering_shapes(self, dump_env, fmt):
        """The buffer and byte measure hold for every _render_body shape."""
        config.DUMP_FORMAT = fmt
        bodies = [_block_shape_body(), _plain_shape_body()]
        for body in bodies:
            dump_turn(body)

        assert len(dump._sessions) == 1
        state = dump._sessions[session_id(bodies[0])]
        assert len(state.pending) == 2
        assert state.pending_bytes == sum(len(_render_body(b).encode("utf-8")) for b in bodies)

    @pytest.mark.parametrize("fmt", COLLAPSED_FORMATS)
    def test_sessions_buffer_independently(self, dump_env, fmt):
        """Two system prompts produce two states, each with its own records."""
        config.DUMP_FORMAT = fmt
        bodies_a = _turn_bodies("sys-a", 2)
        bodies_b = _turn_bodies("sys-b", 3)
        for body in bodies_a + bodies_b:
            dump_turn(body)

        state_a = dump._sessions[session_id(bodies_a[0])]
        state_b = dump._sessions[session_id(bodies_b[0])]
        assert state_a is not state_b
        assert len(state_a.pending) == 2
        assert len(state_b.pending) == 3

    @pytest.mark.parametrize("fmt", COLLAPSED_FORMATS)
    @pytest.mark.parametrize("text", ["é", "😀", "hello"])
    def test_utf8_pending_bytes_are_session_local(self, dump_env, fmt, text):
        """Each session counts rendered UTF-8 bytes, not records or characters."""
        config.DUMP_FORMAT = fmt
        a = _body("sys-a", [text])
        b = _body("sys-b", ["other"])
        dump_turn(a)
        dump_turn(b)

        state_a = dump._sessions[session_id(a)]
        state_b = dump._sessions[session_id(b)]
        assert state_a.pending_bytes == len(_render_body(a).encode("utf-8"))
        assert state_b.pending_bytes == len(_render_body(b).encode("utf-8"))
        dump.flush_all()
        assert state_a.pending_bytes == state_b.pending_bytes == 0
        assert len(state_a.pending) == len(state_b.pending) == 0

    @pytest.mark.parametrize("fmt", COLLAPSED_FORMATS)
    def test_state_advances_at_append_time(self, dump_env, fmt):
        """turn/prev_text advance on every call, together with the buffer."""
        config.DUMP_FORMAT = fmt
        bodies = _turn_bodies("sys-a", 3)
        sid = session_id(bodies[0])
        for i, body in enumerate(bodies):
            dump_turn(body)
            state = dump._sessions[sid]
            assert state.turn == i + 1
            assert state.prev_text == _render_body(body)
            assert len(state.pending) == i + 1


class TestSessionDirGuard:
    """R2: the session directory itself must not be a symlink or a non-directory."""

    def test_symlinked_session_dir_is_refused(self, dump_env, tmp_path_factory):
        """A pre-existing live symlink at DUMP_DIR/<sid> gets no write and survives."""
        body = _body("sys-a", ["m0"])
        sid = session_id(body)
        outside = tmp_path_factory.mktemp("outside")
        (dump_env / sid).symlink_to(outside)

        with pytest.raises(ValueError, match="session dir"):
            dump_turn(body)

        assert list(outside.iterdir()) == []
        assert (dump_env / sid).is_symlink()
        assert sid not in dump._sessions

    def test_regular_file_at_session_dir_is_refused(self, dump_env):
        """A plain file where the session directory belongs is refused, not replaced."""
        body = _body("sys-a", ["m0"])
        sid = session_id(body)
        (dump_env / sid).write_text("not a directory")

        with pytest.raises(ValueError, match="session dir"):
            dump_turn(body)

        assert (dump_env / sid).read_text() == "not a directory"


class TestDocumentedRunWitness:
    """claim-93's documented half, as one fixed-format run.

    The format is never changed between turns: documented turns write their
    files immediately, the pending list stays empty, and no collapsed .log
    artifact is ever created.
    """

    def test_immediate_files_empty_pending_no_log(self, dump_env):
        """Each turn lands on disk at once; nothing buffers; no .log exists."""
        bodies = _turn_bodies("sys-a", 3)
        sid = session_id(bodies[0])
        for i, body in enumerate(bodies):
            dump_turn(body)
            state = dump._sessions[sid]
            expected = f"turn_{i:03d}.json" if i == 0 else f"turn_{i:03d}.diff"
            assert (dump_env / sid / expected).is_file()
            assert state.pending == []
            assert state.pending_bytes == 0

        assert len((dump_env / sid / "index.txt").read_text().splitlines()) == 3
        assert list((dump_env / sid).glob("*.log")) == []
