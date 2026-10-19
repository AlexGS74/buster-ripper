"""Instrumented write-count tests for the dump writers (T004).

Binds the plan's settled write-operation unit (DC-claim-3-r1): one
write-bearing file operation on a dump artifact — ``write_text``, or
``open(..., "a"/"w")`` + ``write``. Directory creation is not counted; the
SQLite stats-DB path (session.py) routes through ``sqlite3.connect`` and can
never appear, so it is structurally outside the measured surface. G-001
scopes the measure to write-bearing operations issued against DUMP_DIR, so
the recorder filters on that root.

One hook captures the whole unit: CPython 3.13's ``Path.write_text`` funnels
through ``Path.open(mode="w")`` (pathlib/_abc.py:644-652), so patching
``Path.open`` for append/write modes and recording the returned file's
``write`` sees both shapes with no double counting.

These are also G-008(b)'s paired runs: the collapsed F vs documented 2N
measurement at the same N, per shipped collapsed format.
"""

import dataclasses
from pathlib import Path

import pytest

from buster_ripper import config, dump
from buster_ripper.dump import dump_turn, flush_all
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


def _run_fixed_format(system: str, n_turns: int) -> list[dict]:
    """Dump n turns for one sid under the currently configured format."""
    bodies = _turn_bodies(system, n_turns)
    for body in bodies:
        dump_turn(body)
    return bodies


@dataclasses.dataclass
class WriteEvent:
    """One write-bearing operation on a dump artifact."""
    path: Path
    mode: str
    byte_len: int


class _RecordingFile:
    """File wrapper recording each write() issued against a dump artifact."""

    def __init__(self, handle, path, mode, sink):
        self._handle = handle
        self._sink = sink
        self._meta = (path, mode)

    def write(self, data):
        self._sink(*self._meta, data)
        return self._handle.write(data)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self._handle.close()


class _WriteHook:
    """Instruments Path.open so write-bearing ops under a root are recorded."""

    def __init__(self, root: Path):
        self.root = root
        self.events: list[WriteEvent] = []

    def install(self, mocker) -> "_WriteHook":
        """Patch Path.open for the test's duration."""
        hook = self
        real_open = Path.open

        def recording_open(path, mode="r", *args, **kwargs):
            handle = real_open(path, mode, *args, **kwargs)
            if not hook.covers(path, mode):
                return handle
            return _RecordingFile(handle, path, mode, hook.record)

        mocker.patch.object(Path, "open", recording_open)
        return self

    def covers(self, path: Path, mode: str) -> bool:
        """True for append/write opens of paths inside the dump root."""
        return "r" not in mode and path.is_relative_to(self.root)

    def record(self, path: Path, mode: str, data) -> None:
        """Append one write event, measuring the payload's UTF-8 bytes."""
        size = len(data.encode("utf-8")) if isinstance(data, str) else len(data)
        self.events.append(WriteEvent(path, mode, size))

    def bytes_total(self) -> int:
        """Sum of every recorded payload — bytes written, not call count."""
        return sum(e.byte_len for e in self.events)

    @staticmethod
    def batch_bytes(batch: list[str]) -> int:
        """The payload size a flush of `batch` must carry (one write)."""
        return len(("\n\n".join(batch) + "\n").encode("utf-8"))


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


class TestWriteUnit:
    """The instrument counts exactly the settled unit (DC-claim-3-r1)."""

    def test_mkdir_and_outside_paths_are_not_counted(self, dump_env, mocker):
        """Session-dir creation happens but is not a counted write-bearing op."""
        hook = _WriteHook(dump_env).install(mocker)
        mkdir = mocker.spy(Path, "mkdir")

        config.DUMP_FORMAT = "change-focused"
        config.DUMP_FLUSH_BYTES = 10 ** 9
        for body in _turn_bodies("sys-a", 2):
            dump_turn(body)
        flush_all()

        assert mkdir.call_count >= 1
        assert [e.mode for e in hook.events] == ["a"]
        assert all(e.path.is_relative_to(dump_env) for e in hook.events)
        assert all(e.path.is_file() for e in hook.events)


class TestOneWritePerFlush:
    """A flush of K ≥ 2 pending turns performs exactly one write."""

    @pytest.mark.parametrize("fmt", COLLAPSED_FORMATS)
    def test_flush_of_k_pending_turns_is_one_write(self, dump_env, mocker, fmt):
        """K=3 turns below the threshold → one append whose payload is pinned."""
        hook = _WriteHook(dump_env).install(mocker)
        config.DUMP_FORMAT = fmt
        config.DUMP_FLUSH_BYTES = 10 ** 9
        sid = session_id(_body("sys-a", []))
        bodies = _turn_bodies("sys-a", 3)
        for body in bodies:
            dump_turn(body)
        batch = list(dump._sessions[sid].pending)

        flush_all()

        assert len(hook.events) == 1
        event = hook.events[0]
        assert event.path == dump_env / sid / f"{fmt}.log"
        assert event.mode == "a"
        assert event.byte_len == _WriteHook.batch_bytes(batch)
        assert event.path.read_text() == "\n\n".join(batch) + "\n"


class TestFlushCountIsF:
    """N turns flushed F times cost F writes — the count a user controls."""

    @pytest.mark.parametrize("fmt", COLLAPSED_FORMATS)
    def test_injected_f_flushes_write_f_times_batch_only(
        self, dump_env, mocker, fmt
    ):
        """F manual flushes → F appends; each carries only its own batch.

        The bytes written across the session grow with F, not N² (G-001
        witness B excluded): the on-disk artifact is exactly the batches.
        """
        hook = _WriteHook(dump_env).install(mocker)
        config.DUMP_FORMAT = fmt
        config.DUMP_FLUSH_BYTES = 10 ** 9
        sid = session_id(_body("sys-a", []))

        batches = []
        for body in _turn_bodies("sys-a", 4):
            dump_turn(body)
            batches.append(list(dump._sessions[sid].pending))
            flush_all()

        assert len(hook.events) == 4  # F == N: the reduction's floor case
        assert [e.byte_len for e in hook.events] == [
            _WriteHook.batch_bytes(b) for b in batches
        ]
        artifact = dump_env / sid / f"{fmt}.log"
        assert hook.bytes_total() == artifact.stat().st_size


class TestDocumentedCount:
    """The documented path at the same N counts 2N (the pair's baseline)."""

    def test_documented_run_counts_2n_writes(self, dump_env, mocker):
        """N=3 turns → 6 write-bearing ops: 3 content writes + 3 index appends."""
        hook = _WriteHook(dump_env).install(mocker)

        for body in _turn_bodies("sys-a", 3):
            dump_turn(body)

        assert len(hook.events) == 6
        assert sorted(e.mode for e in hook.events) == ["a", "a", "a", "w", "w", "w"]


class TestPairedMeasurement:
    """G-008(b)'s paired runs: collapsed F vs documented 2N at the same N.

    Each run is fixed-configuration (claim-93): one sid's turns all dump
    under one format; the format changes only between the two independent
    runs, never between a run's turns.
    """

    @pytest.mark.parametrize("fmt", COLLAPSED_FORMATS)
    def test_collapsed_f_beats_documented_2n_at_same_n(self, dump_env, mocker, fmt):
        """N=4: documented writes 2N, collapsed writes F=1 — the measurement."""
        hook = _WriteHook(dump_env).install(mocker)
        config.DUMP_FLUSH_BYTES = 10 ** 9
        n = 4

        _run_fixed_format("sys-doc", n)
        documented_writes = len(hook.events)

        hook.events.clear()
        config.DUMP_FORMAT = fmt
        _run_fixed_format("sys-col", n)
        flush_all()
        collapsed_writes = len(hook.events)

        assert (documented_writes, collapsed_writes) == (2 * n, 1)
        assert 1 - collapsed_writes / documented_writes == 1 - 1 / 8


class TestFileCount:
    """VP-T002-file-count-reduced: 1 artifact collapsed vs N + 1 documented."""

    @pytest.mark.parametrize("fmt", COLLAPSED_FORMATS)
    def test_one_artifact_vs_n_plus_one_files(self, dump_env, fmt):
        """Same N, fixed formats per run: 1 file collapsed; N + 1 documented."""
        n = 4

        doc_bodies = _run_fixed_format("sys-doc", n)
        config.DUMP_FORMAT = fmt
        col_bodies = _run_fixed_format("sys-col", n)
        flush_all()

        doc_dir = dump_env / session_id(doc_bodies[0])
        col_dir = dump_env / session_id(col_bodies[0])
        assert len(list(doc_dir.iterdir())) == n + 1
        assert [p.name for p in col_dir.iterdir()] == [f"{fmt}.log"]
