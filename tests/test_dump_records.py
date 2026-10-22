"""Tests for the collapsed record shapes and their artifact witnesses (T002/T004).

Record-shape classes moved out of test_dump.py when T004 grew that file past
the FL-LNCP review band (QB:test-dump/file-size): change-focused and
self-contained record construction, plus the artifact-level witnesses T004
adds — a reader inspecting only ``DUMP_DIR/<sid>/<format>.log`` can locate a
changed turn (VP-T002-change-readable, per format) and recover turn 0's full
body and rendering (VP-T002-turn0-available, per format).
"""

import difflib
import json
import re

import pytest

from buster_ripper import config, dump
from buster_ripper.dump import dump_turn
from buster_ripper.dump_render import _render_body
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


def _changed_turn_bodies() -> list[dict]:
    """Three turns where turn 2 rewrites one known message.

    The reader witness's known change: turn 2 replaces ``stable two`` with
    ``REWRITTEN-MARKER`` while every other message stays put.
    """
    return [
        _body("sys-a", ["keep one", "stable two"]),
        _body("sys-a", ["keep one", "stable two", "third message"]),
        _body("sys-a", ["keep one", "REWRITTEN-MARKER", "third message"]),
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


def _record_for_turn(artifact: str, turn: int) -> str:
    """Reader-style extraction of one turn's record from the artifact.

    A reader locates a turn by its summary line (``turn NNN  HH:MM:SS ...``)
    and reads forward to the next record's summary line — the grep/less
    workflow the README documents.
    """
    lines = artifact.splitlines(keepends=True)
    starts = [i for i, l in enumerate(lines) if l.startswith(f"turn {turn:03d}  ")]
    assert starts, f"turn {turn:03d} not found in artifact"
    rest = lines[starts[0] + 1:]
    end = next(
        (j for j, l in enumerate(rest) if re.match(r"turn \d{3}  ", l)), len(rest)
    )
    return "".join(lines[starts[0]:starts[0] + 1 + end])


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


class TestChangeFocusedRecords:
    """The change-focused record: turn 0 body+rendering, later turns' diffs."""

    def test_turn_zero_record_carries_body_and_rendering(self, dump_env):
        """Turn 0's record holds the FULL summary, JSON body and rendering."""
        config.DUMP_FORMAT = "change-focused"
        body = _body("sys-a", ["m0", "m1"])
        dump_turn(body)

        record = dump._sessions[session_id(body)].pending[0]
        rendered = _render_body(body)
        assert record.splitlines()[0].startswith("turn 000")
        assert "  FULL  " in record.splitlines()[0]
        assert "--- turn 000 body ---" in record
        assert json.dumps(body, indent=2, ensure_ascii=False) in record
        assert "--- turn 000 rendered ---" in record
        assert rendered in record

    def test_later_turn_record_control_lines_are_terminated(self, dump_env):
        """Turn 1's record carries each ---/+++/@@ control line on its own line."""
        config.DUMP_FORMAT = "change-focused"
        bodies = _turn_bodies("sys-a", 2)
        sid = session_id(bodies[0])
        for body in bodies:
            dump_turn(body)

        lines = dump._sessions[sid].pending[1].splitlines(keepends=True)
        assert "--- turn_000\n" in lines
        assert "+++ turn_001\n" in lines
        hunk = next(ln for ln in lines if ln.startswith("@@"))
        assert hunk.startswith("@@ ") and hunk.endswith(" @@\n"), hunk

    def test_later_turn_record_carries_diff(self, dump_env):
        """Turn ≥ 1's record holds the summary and the unified diff text."""
        config.DUMP_FORMAT = "change-focused"
        bodies = _turn_bodies("sys-a", 3)
        sid = session_id(bodies[0])
        for body in bodies:
            dump_turn(body)

        records = dump._sessions[sid].pending
        rendered = [_render_body(b) for b in bodies]
        diff_1 = _expected_diff(1, rendered[0], rendered[1])
        diff_2 = _expected_diff(2, rendered[1], rendered[2])

        assert "--- turn 001 diff ---" in records[1]
        assert diff_1 in records[1]
        assert "--- turn 002 diff ---" in records[2]
        assert diff_2 in records[2]
        assert rendered[1] not in records[1]


class TestSelfContainedRecords:
    """The self-contained record: full rendering every turn, turn 0 also body."""

    def test_turn_zero_record_carries_body_and_rendering(self, dump_env):
        """Turn 0's record holds the FULL summary, JSON body and rendering."""
        config.DUMP_FORMAT = "self-contained"
        body = _body("sys-a", ["m0", "m1"])
        dump_turn(body)

        record = dump._sessions[session_id(body)].pending[0]
        assert "  FULL  " in record.splitlines()[0]
        assert "--- turn 000 body ---" in record
        assert json.dumps(body, indent=2, ensure_ascii=False) in record
        assert "--- turn 000 rendered ---" in record
        assert _render_body(body) in record

    def test_every_turn_carries_complete_rendering(self, dump_env):
        """Each turn's record holds the full rendering; no diff text is stored."""
        config.DUMP_FORMAT = "self-contained"
        bodies = _turn_bodies("sys-a", 3)
        sid = session_id(bodies[0])
        for body in bodies:
            dump_turn(body)

        records = dump._sessions[sid].pending
        rendered = [_render_body(b) for b in bodies]
        for i in (1, 2):
            assert rendered[i] in records[i]
            assert f"--- turn {i:03d} rendered ---" in records[i]
            assert f"+++ turn {i:03d}" not in records[i]

    def test_summary_counts_computed_without_storing_diff(self, dump_env):
        """Turn ≥ 1's summary carries +added/-removed from a discarded diff."""
        config.DUMP_FORMAT = "self-contained"
        bodies = _turn_bodies("sys-a", 2)
        sid = session_id(bodies[0])
        for body in bodies:
            dump_turn(body)

        record = dump._sessions[sid].pending[1]
        diff_text = _expected_diff(1, _render_body(bodies[0]), _render_body(bodies[1]))
        lines = diff_text.splitlines()
        added = sum(1 for l in lines if l.startswith("+") and not l.startswith("+++"))
        removed = sum(1 for l in lines if l.startswith("-") and not l.startswith("---"))
        assert f"+{added}/-{removed} lines" in record.splitlines()[0]
        assert "@@ " not in record


class TestSessionStateDefaults:
    """A fresh SessionState owns an empty, private buffer."""

    def test_new_state_has_empty_private_buffer(self):
        """Two states' pending lists are distinct objects, both empty."""
        from buster_ripper.dump import SessionState

        state_a = SessionState()
        state_b = SessionState()

        assert state_a.pending == []
        assert state_b.pending == []
        assert state_a.pending is not state_b.pending
        assert state_a.pending_bytes == 0


class TestArtifactReadability:
    """VP-T002-change-readable, per collapsed format, against the artifact.

    F-11's two readings: change-focused holds as written (the reader sees the
    differing content in turn 2's record); self-contained becomes a
    reader-performed comparison of turn 2's record against turn 1's.
    """

    def _flushed_artifact(self, dump_env, fmt) -> str:
        """Run the changed-turn session under fmt, flush, return the artifact."""
        config.DUMP_FORMAT = fmt
        bodies = _changed_turn_bodies()
        sid = session_id(bodies[0])
        for body in bodies:
            dump_turn(body)
        dump.flush_all()
        return (dump_env / sid / f"{fmt}.log").read_text()

    def test_change_focused_reader_sees_the_change(self, dump_env):
        """Turn 2's diff record itself shows the rewritten and removed lines."""
        artifact = self._flushed_artifact(dump_env, "change-focused")
        turn2 = _record_for_turn(artifact, 2)

        added = [l for l in turn2.splitlines() if l.startswith("+") and "REWRITTEN-MARKER" in l]
        removed = [l for l in turn2.splitlines() if l.startswith("-") and "stable two" in l]
        assert added, "turn 2's record must show the changed content"
        assert removed, "turn 2's record must show what it replaced"

    def test_self_contained_reader_compares_records(self, dump_env):
        """The reader diffs turn 2's full record against turn 1's by eye."""
        artifact = self._flushed_artifact(dump_env, "self-contained")
        turn1 = _record_for_turn(artifact, 1)
        turn2 = _record_for_turn(artifact, 2)

        assert "REWRITTEN-MARKER" in turn2
        assert "REWRITTEN-MARKER" not in turn1
        assert "stable two" in turn1


class TestArtifactTurnZero:
    """VP-T002-turn0-available, per collapsed format: turn 0 recoverable."""

    @pytest.mark.parametrize("fmt", COLLAPSED_FORMATS)
    def test_body_and_rendering_recoverable_from_artifact(self, dump_env, fmt):
        """The JSON body round-trips and the rendering is present in full."""
        config.DUMP_FORMAT = fmt
        body = _body("sys-a", ["alpha", "beta"])
        sid = session_id(body)
        dump_turn(body)
        dump.flush_all()

        record = _record_for_turn((dump_env / sid / f"{fmt}.log").read_text(), 0)
        json_part = record.split("--- turn 000 body ---\n", 1)[1]
        json_part = json_part.split("\n--- turn 000 rendered ---", 1)[0]
        rendered_part = record.split("--- turn 000 rendered ---\n", 1)[1]

        assert json.loads(json_part) == body
        assert rendered_part.startswith(_render_body(body))
