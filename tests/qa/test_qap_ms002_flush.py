"""Flush-trigger MS-002 gates: volume, shutdown, interval and timer resilience."""

from __future__ import annotations

import json
import os
import signal

import pytest

from tests.qa.harness import (
    assert_upstream_response,
    log_artifacts,
    read_text,
    session_id,
    turn_body,
)
from tests.qa.harness_process import (
    Proxy,
    poll_until,
)
from tests.qa.qap_ms002_support import (
    COLLAPSED_FORMATS,
    THRESHOLD_AT_TURN_2,
    TIMER_OFF_INTERVAL,
    UNREACHABLE_BYTES,
    big_text,
    summary_lines,
)


# TC-105 / TC-109 — paired instrumented runs (the G-008(b) measurement)

N_TURNS = 6  # F=2: flush after turn 2 and after turn 5


def _counted_run(tmp_path, tag, fmt):
    system = f"qa-ms002-count-{tag}"
    dump = tmp_path / f"dump-{tag}"
    report = tmp_path / f"report-{tag}.json"
    extra = ["--dump-flush-interval", str(TIMER_OFF_INTERVAL),
             "--dump-flush-bytes", str(THRESHOLD_AT_TURN_2)]
    if fmt is not None:
        extra = ["--dump-format", fmt, *extra]
    proxy = Proxy(dump, tmp_path / f"logs-{tag}", extra_args=extra,
                  counted_report=report)
    try:
        for i in range(N_TURNS):
            st, body = proxy.post(turn_body(system, big_text(f"count-{tag}-t{i}")))
            assert_upstream_response(st, body)
    finally:
        rc = proxy.stop()
    assert rc == 0
    with open(report, encoding="utf-8") as f:
        rep = json.load(f)
    sid_dir = os.path.join(str(dump), session_id(system))
    return rep, sid_dir


def test_tc105_tc109_write_and_file_counts_paired_runs(tmp_path):
    results = {}
    for fmt in (*COLLAPSED_FORMATS, None):  # None = documented format
        tag = fmt or "documented"
        rep, sid_dir = _counted_run(tmp_path, tag, fmt)
        results[tag] = (rep, sid_dir)

    comparison = []
    for tag, (rep, sid_dir) in results.items():
        listing = sorted(os.listdir(sid_dir))
        comparison.append(
            f"[TC-105 comparison] N={N_TURNS} format={tag} "
            f"write_ops={rep['total_ops']} write_bytes={rep['total_bytes']} "
            f"files={len(listing)}"
        )
    print("\n".join(comparison))

    for fmt in COLLAPSED_FORMATS:
        rep, sid_dir = results[fmt]
        artifact = os.path.join(sid_dir, f"{fmt}.log")
        final_size = os.path.getsize(artifact)
        # F controlled by the injected threshold: exactly two flushes.
        assert rep["total_ops"] == 2, f"{fmt}: expected F=2 write ops, got {rep['total_ops']}"
        assert all(os.path.abspath(o["path"]) == os.path.abspath(artifact) for o in rep["ops"])
        # Witness A excluded: one write per flush of K >= 2 pending turns.
        # Witness B excluded: bytes track the pending batches, not the
        # accumulated artifact (no rewrite-per-flush).
        assert all(0 < o["bytes"] < final_size for o in rep["ops"])
        assert sum(o["bytes"] for o in rep["ops"]) == final_size
        # TC-109: file count 1 collapsed ...
        assert os.listdir(sid_dir) == [f"{fmt}.log"]
        assert len(summary_lines(read_text(artifact))) == N_TURNS

    rep, sid_dir = results["documented"]
    # ... versus N + 1 documented, at 2N write-bearing operations.
    expected_files = (
        {"turn_000.json", "index.txt"}
        | {f"turn_{i:03d}.diff" for i in range(1, N_TURNS)}
    )
    assert sorted(os.listdir(sid_dir)) == sorted(expected_files)
    assert rep["total_ops"] == 2 * N_TURNS
    index_text = read_text(os.path.join(sid_dir, "index.txt"))
    assert len([l for l in index_text.splitlines() if l.strip()]) == N_TURNS
    assert json.loads(read_text(os.path.join(sid_dir, "turn_000.json")))["model"] == (
        "sloppus"
    )
    # No .log artifact may appear on the documented path.
    assert not log_artifacts(str(tmp_path / "dump-documented"))


# TC-106 — VP-T002-volume-flush

def test_tc106_volume_flush(tmp_path):
    system = "qa-ms002-volume"
    dump = tmp_path / "dump"
    proxy = Proxy(
        dump,
        tmp_path / "logs",
        extra_args=[
            "--dump-format", "change-focused",
            "--dump-flush-interval", str(TIMER_OFF_INTERVAL),  # timer cannot fire
            "--dump-flush-bytes", str(THRESHOLD_AT_TURN_2),
        ],
    )
    try:
        assert_upstream_response(*proxy.post(turn_body(system, big_text("volume-t0"))))
        assert_upstream_response(*proxy.post(turn_body(system, big_text("volume-t1"))))
        assert log_artifacts(str(dump)) == [], "pending bytes below threshold: nothing on disk"
        st, body = proxy.post(turn_body(system, big_text("volume-t2")))
        assert_upstream_response(st, body)
        assert proxy.proc.poll() is None, "no shutdown, process still running"
    finally:
        rc = proxy.stop()
    assert rc == 0

    artifact = proxy.artifact_path(system, "change-focused")
    text = read_text(artifact)
    ids = [int(line.split()[1]) for line in summary_lines(text)]
    assert ids == [0, 1, 2], "the threshold flush carries exactly turns 0..2"


# TC-110 — VP-T003-shutdown-flush (real SIGINT and SIGTERM)

@pytest.mark.parametrize("sig", [signal.SIGINT, signal.SIGTERM], ids=["sigint", "sigterm"])
def test_tc110_shutdown_flush(tmp_path, sig):
    system = "qa-ms002-shutdown"
    dump = tmp_path / "dump"
    proxy = Proxy(
        dump,
        tmp_path / "logs",
        extra_args=[
            "--dump-format", "change-focused",
            "--dump-flush-interval", str(TIMER_OFF_INTERVAL),
            "--dump-flush-bytes", str(UNREACHABLE_BYTES),
        ],
    )
    try:
        for i in range(3):
            st, body = proxy.post(turn_body(system, f"shutdown-t{i} payload"))
            assert_upstream_response(st, body)
        # Sensitivity: turns are buffered, nothing flushed yet.
        assert log_artifacts(str(dump)) == [], "no flush before the signal"
        proxy.proc.send_signal(sig)  # real signal delivery
        proxy.proc.wait(timeout=30)
    finally:
        rc = proxy.stop(sig=signal.SIGKILL)  # already exited; idempotent

    stderr = proxy.stderr()
    assert rc == 0, f"clean exit expected, got {rc}; stderr tail: {stderr[-2000:]!r}"
    assert "Traceback" not in stderr
    text = read_text(proxy.artifact_path(system, "change-focused"))
    ids = [int(line.split()[1]) for line in summary_lines(text)]
    assert ids == [0, 1, 2], "buffered turns must be on disk, in order, after the signal"


# TC-111 — VP-T003-interval-flush

def test_tc111_interval_flush(tmp_path):
    system = "qa-ms002-interval"
    dump = tmp_path / "dump"
    proxy = Proxy(
        dump,
        tmp_path / "logs",
        extra_args=[
            "--dump-format", "change-focused",
            "--dump-flush-interval", "3",  # small
            "--dump-flush-bytes", str(UNREACHABLE_BYTES),  # unreachable
        ],
    )
    try:
        artifact = proxy.artifact_path(system, "change-focused")
        assert not os.path.exists(artifact), "sensitivity: nothing before the turn"
        st, body = proxy.post(turn_body(system, "interval-t0 sentinel-intvl"))
        assert_upstream_response(st, body)
        # Quiescence: poll the artifact, never a fixed sleep.
        ok = poll_until(
            lambda: os.path.exists(artifact)
            and "sentinel-intvl" in read_text(artifact),
            deadline=15,
        )
        assert ok, "the appended turn must reach the artifact while the process runs"
        assert proxy.proc.poll() is None
    finally:
        rc = proxy.stop()
    assert rc == 0


# TC-112 — VP-T003-timer-resilience

def test_tc112_timer_resilience(tmp_path):
    system = "qa-ms002-resilience"
    dump = tmp_path / "dump"
    proxy = Proxy(
        dump,
        tmp_path / "logs",
        extra_args=[
            "--dump-format", "change-focused",
            "--dump-flush-interval", "2",
            "--dump-flush-bytes", str(UNREACHABLE_BYTES),
        ],
    )
    artifact = proxy.artifact_path(system, "change-focused")
    try:
        st, _ = proxy.post(turn_body(system, "resilience-t0"))
        assert_upstream_response(st, _)
        ok = poll_until(
            lambda: os.path.exists(artifact) and "turn 000" in read_text(artifact),
            deadline=15,
        )
        assert ok, "first tick must flush turn 0"

        # Inject a raising flush: the artifact becomes unwritable for the
        # owner, so every tick's flush raises. The loop must log and continue.
        os.chmod(artifact, 0o444)
        try:
            for i in (1, 2):
                st, body = proxy.post(turn_body(system, f"resilience-t{i}"))
                assert_upstream_response(st, body)
            ok = poll_until(
                lambda: "PermissionError" in proxy.stderr(), deadline=20
            )
            assert ok, "the failing tick must be logged"

            # A later tick, after the fault clears, must flush the retained batch.
            os.chmod(artifact, 0o644)
            ok = poll_until(
                lambda: "turn 001" in read_text(artifact)
                and "turn 002" in read_text(artifact),
                deadline=20,
            )
            assert ok, "a later tick must flush successfully after the raise"
        finally:
            os.chmod(artifact, 0o644)
        assert proxy.proc.poll() is None, "process healthy throughout"
    finally:
        rc = proxy.stop()
    # Clean exit: the injected PermissionError is logged but must not raise out of on_cleanup.
    assert rc == 0
    ids = [int(line.split()[1]) for line in summary_lines(read_text(artifact))]
    assert ids == [0, 1, 2]
