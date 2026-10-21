"""QAP-MS002 — MS-002 integration gates: the collapsed path becomes observable.

Covers (qa-plan.md §QA Gate Matrix + §QA Author Passes) the TC↔VP map: TC-101
VP-T001-session-isolation (+ R5 sid near-miss), TC-102 VP-T002-collapsed-artifact
(+ R5 (sid, format) near-miss), TC-103 VP-T002-change-readable, TC-104
VP-T002-turn0-available, TC-105 VP-T002-write-count-reduced (G-008(b)), TC-106
VP-T002-volume-flush, TC-109 VP-T002-file-count-reduced, TC-110
VP-T003-shutdown-flush (SIGINT/SIGTERM), TC-111 VP-T003-interval-flush, TC-112
VP-T003-timer-resilience; the hostile tier (R1/R2) maps to
VP-T002-collapsed-artifact and VP-T002-volume-flush.

Run explicitly (not in the hermetic default suite):
``.venv/bin/python -m pytest tests/qa/test_qap_ms002.py -o addopts=``.

The declared live upstream is required: gates fail closed without it and assert on
dump artifacts and process behaviour, never model output (a proxy-local 502 is not a
baseline).
"""

from __future__ import annotations

import json
import os
import re
import signal

import pytest

from tests.qa.harness import (
    assert_upstream_response,
    log_artifacts,
    read_text,
    record_section,
    session_id,
    turn_body,
)
from tests.qa.harness_process import (
    Proxy,
    poll_until,
)

COLLAPSED_FORMATS = ("change-focused", "self-contained")

# ~TURN_SIZE per turn plus small overhead: 2 pending turns stay below these
# thresholds and 3 cross, with ~20 KB margin on each side.
TURN_SIZE = 40_000
THRESHOLD_AT_TURN_2 = 100_000
UNREACHABLE_BYTES = 10 ** 12
TIMER_OFF_INTERVAL = 3600


def big_text(marker: str, fill: str = "x") -> str:
    return marker + " " + fill * TURN_SIZE


def summary_lines(artifact_text: str) -> list[str]:
    """The one-line-per-turn summary lines, in artifact order."""
    return re.findall(r"(?m)^turn \d{3}\b.*$", artifact_text)


# TC-101 — VP-T001-session-isolation (+ R5: 1-char near-miss system prompts)

def test_tc101_session_isolation_near_miss_prompts(tmp_path):
    dump = tmp_path / "dump"
    s1 = "qa-ms002-isolation-prompt-A0"
    s2 = "qa-ms002-isolation-prompt-B0"  # differs from s1 by exactly one char
    proxy = Proxy(
        dump,
        tmp_path / "logs",
        extra_args=[
            "--dump-format", "change-focused",
            "--dump-flush-interval", "2",
            "--dump-flush-bytes", str(UNREACHABLE_BYTES),
        ],
    )
    try:
        st1, b1 = proxy.post(turn_body(s1, big_text("sentinel-one t0")))
        st2, b2 = proxy.post(turn_body(s2, big_text("sentinel-two t0")))
        assert_upstream_response(st1, b1)
        assert_upstream_response(st2, b2)
        a1 = proxy.artifact_path(s1, "change-focused")
        a2 = proxy.artifact_path(s2, "change-focused")
        ok = poll_until(
            lambda: os.path.exists(a1)
            and os.path.exists(a2)
            and "sentinel-one" in read_text(a1)
            and "sentinel-two" in read_text(a2),
            deadline=15,
        )
        assert ok, "both sessions' artifacts must appear while the process runs (timer flush)"
    finally:
        rc = proxy.stop()
    assert rc == 0

    t1, t2 = read_text(proxy.artifact_path(s1, "change-focused")), read_text(
        proxy.artifact_path(s2, "change-focused")
    )
    # Near-miss prompts must NOT merge: distinct computed sids, distinct dirs.
    sid1, sid2 = session_id(s1), session_id(s2)
    assert sid1 != sid2
    assert os.listdir(str(dump)) == sorted([sid1, sid2])
    # Each artifact carries only its own session's records.
    assert "sentinel-one" in t1 and "sentinel-two" not in t1
    assert "sentinel-two" in t2 and "sentinel-one" not in t2


# TC-102 / TC-103 / TC-104 — artifact shape, readability, turn 0 (per format)

@pytest.mark.parametrize("fmt", COLLAPSED_FORMATS)
def test_tc102_103_104_collapsed_artifact_shape_readability_turn0(tmp_path, fmt):
    system = f"qa-ms002-shape-{fmt}"
    dump = tmp_path / "dump"
    proxy = Proxy(
        dump,
        tmp_path / "logs",
        extra_args=[
            "--dump-format", fmt,
            "--dump-flush-interval", str(TIMER_OFF_INTERVAL),
            "--dump-flush-bytes", str(THRESHOLD_AT_TURN_2),
        ],
    )
    try:
        # Sensitivity: below the threshold (after turn 1) nothing is on disk.
        assert_upstream_response(*proxy.post(turn_body(system, big_text("shape-t0 marker-zero"))))
        assert_upstream_response(*proxy.post(turn_body(system, "shared-prefix old-marker-ZZZQ " + "x" * TURN_SIZE)))
        assert log_artifacts(str(dump)) == [], "no artifact before the threshold is reached"
        st, body = proxy.post(turn_body(system, "shared-prefix new-marker-YYYW " + "y" * TURN_SIZE))
        assert_upstream_response(st, body)
    finally:
        rc = proxy.stop()
    assert rc == 0

    sid = session_id(system)
    sid_dir = os.path.join(str(dump), sid)
    # TC-102: exactly one artifact, named by the configured value; no
    # documented files on the collapsed path.
    assert os.listdir(sid_dir) == [f"{fmt}.log"]

    text = read_text(os.path.join(sid_dir, f"{fmt}.log"))

    # TC-102: three records in arrival order (summary lines carry turn ids).
    ids = [int(line.split()[1]) for line in summary_lines(text)]
    assert ids == [0, 1, 2]

    # TC-103: summary-line forms — FULL for turn 0, +A/-R counts for turn >= 1.
    lines = summary_lines(text)
    assert "FULL" in lines[0]
    for line in lines[1:]:
        assert re.search(r"\+\d+\s*/\s*-\d+", line), f"summary line lost change counts: {line!r}"

    # TC-103: turn 2's change is locatable and identifiable, per format.
    if fmt == "change-focused":
        assert re.search(r"(?m)^-.*old-marker-ZZZQ", text), "diff must show the removed content"
        assert re.search(r"(?m)^\+.*new-marker-YYYW", text), "diff must show the added content"
    else:
        r1 = record_section(text, "--- turn 001 rendered ---")
        r2 = record_section(text, "--- turn 002 rendered ---")
        assert "old-marker-ZZZQ" in r1 and "new-marker-YYYW" not in r1
        assert "new-marker-YYYW" in r2 and "old-marker-ZZZQ" not in r2

    # TC-104: turn 0's raw normalized JSON body AND its rendering recoverable.
    body_block = record_section(text, "--- turn 000 body ---").strip()
    parsed = json.loads(body_block)
    assert parsed["model"] == "sloppus"
    assert parsed["system"] == system
    assert "marker-zero" in parsed["messages"][0]["content"]
    rendered_block = record_section(text, "--- turn 000 rendered ---")
    assert "marker-zero" in rendered_block
    assert "[message 0 / user]" in rendered_block


# Identity near-miss (R5): the (sid, format) key must not merge formats

def test_identity_sid_format_near_miss_two_artifacts(tmp_path):
    system = "qa-ms002-identity-same-prompt"
    dump = tmp_path / "dump"
    common = [
        "--dump-flush-interval", str(TIMER_OFF_INTERVAL),
        "--dump-flush-bytes", "1",  # flush after every appended turn
    ]
    proxy_a = Proxy(dump, tmp_path / "logs-a",
                    extra_args=["--dump-format", "change-focused", *common])
    try:
        st, body = proxy_a.post(turn_body(system, "identity-run-one sentinel-uno"))
        assert_upstream_response(st, body)
    finally:
        rc_a = proxy_a.stop()
    assert rc_a == 0

    cf_path = os.path.join(str(dump), session_id(system), "change-focused.log")
    before = read_text(cf_path)

    proxy_b = Proxy(dump, tmp_path / "logs-b",
                    extra_args=["--dump-format", "self-contained", *common])
    try:
        st, body = proxy_b.post(turn_body(system, "identity-run-two sentinel-dos"))
        assert_upstream_response(st, body)
    finally:
        rc_b = proxy_b.stop()
    assert rc_b == 0

    sid_dir = os.path.join(str(dump), session_id(system))
    assert sorted(os.listdir(sid_dir)) == ["change-focused.log", "self-contained.log"]
    # The first artifact is untouched by the second format's run: the identity
    # key is (sid, format), so the two runs must not merge into one artifact.
    assert read_text(cf_path) == before
    sc_text = read_text(os.path.join(sid_dir, "self-contained.log"))
    assert "sentinel-dos" in sc_text and "sentinel-uno" not in sc_text


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


# Hostile tier — R1 (parse boundary) -> VP-T002-collapsed-artifact

def test_hostile_non_utf8_and_empty_body_no_write(tmp_path):
    system = "qa-ms002-hostile-parse-health"
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
        # Non-UTF-8 bytes: typed stop, no dump write.
        st, body = proxy.post(b'\xff\xfe{"model": "x"}')
        assert 400 <= st < 500, f"non-UTF-8 body must be a client error, got {st}"
        # Empty body: typed stop, no dump write.
        st, body = proxy.post(b"")
        assert 400 <= st < 500, f"empty body must be a client error, got {st}"
        entries = sorted(os.listdir(str(dump)))
        assert entries == [], f"no session dir may be created for unparsable bodies: {entries}"
        # Process still serves: a valid request reaches the dump and live upstream.
        st, body = proxy.post(turn_body(system, "hostile-health sentinel-hp"))
        assert_upstream_response(st, body)
        assert os.path.isdir(proxy.session_dir(system)), "capture must happen for a valid body"
    finally:
        rc = proxy.stop()
    assert rc == 0
    assert "sentinel-hp" in read_text(proxy.artifact_path(system, "change-focused"))


def test_hostile_duplicate_json_keys_last_wins_single_record(tmp_path):
    system = "qa-ms002-hostile-dupkeys"
    dump = tmp_path / "dump"
    proxy = Proxy(
        dump,
        tmp_path / "logs",
        extra_args=[
            "--dump-format", "change-focused",
            "--dump-flush-interval", str(TIMER_OFF_INTERVAL),
            "--dump-flush-bytes", "1",
        ],
    )
    try:
        raw = (
            b'{"model": "m-first", "model": "sloppus", "max_tokens": 16,'
            b' "system": ' + json.dumps(system).encode() +
            b', "messages": [{"role": "user", "content": "dupkeys sentinel-dq"}]}'
        )
        st, body = proxy.post(raw)
        # Duplicate keys are valid JSON: forwarded to the upstream, not rejected
        # as malformed by the proxy.
        assert_upstream_response(st, body)
    finally:
        rc = proxy.stop()
    assert rc == 0

    text = read_text(proxy.artifact_path(system, "change-focused"))
    parsed = json.loads(record_section(text, "--- turn 000 body ---").strip())
    assert parsed["model"] == "sloppus", "the record carries the parsed body (last key wins)"
    assert "m-first" not in text, "no duplicate-key corruption may leak into the record"


def test_hostile_trailing_newline_body_is_valid(tmp_path):
    system = "qa-ms002-hostile-trailingnl"
    dump = tmp_path / "dump"
    proxy = Proxy(
        dump,
        tmp_path / "logs",
        extra_args=[
            "--dump-format", "change-focused",
            "--dump-flush-interval", str(TIMER_OFF_INTERVAL),
            "--dump-flush-bytes", "1",
        ],
    )
    try:
        raw = json.dumps(turn_body(system, "trailingnl sentinel-tn")).encode() + b"\n\n"
        st, body = proxy.post(raw)
        assert_upstream_response(st, body)
    finally:
        rc = proxy.stop()
    assert rc == 0
    assert "sentinel-tn" in read_text(proxy.artifact_path(system, "change-focused"))


# Hostile tier — R2 (writer into pre-existing trees) -> VP-T002-volume-flush

def test_hostile_dangling_session_dir_symlink_no_write(tmp_path):
    system = "qa-ms002-hostile-dangling"
    dump = tmp_path / "dump"
    outside = tmp_path / "outside"
    outside.mkdir()
    dangling_target = outside / "dangling-target"
    os.makedirs(str(dump))
    sid = session_id(system)
    os.symlink(str(dangling_target), os.path.join(str(dump), sid))

    proxy = Proxy(
        dump,
        tmp_path / "logs",
        extra_args=[
            "--dump-format", "change-focused",
            "--dump-flush-interval", str(TIMER_OFF_INTERVAL),
            "--dump-flush-bytes", "1",
        ],
    )
    try:
        st, body = proxy.post(turn_body(system, "dangling sentinel-dl"))
        assert_upstream_response(st, body)
        assert not dangling_target.exists(), "no write may land at the dangling target"
        assert sorted(os.listdir(str(dump))) == [sid], "no real dir may replace the symlink"
        # Process still serves afterwards.
        st, body = proxy.post(turn_body(system, "dangling sentinel-dl-2"))
        assert_upstream_response(st, body)
    finally:
        rc = proxy.stop()
    # rc == 0 is the containment discriminator: the failing flush is logged
    # (traceback formatting included, by design) but must not raise out of
    # on_cleanup, and nothing may be written through the dangling symlink.
    assert rc == 0


def test_hostile_artifact_symlink_escape_no_write(tmp_path):
    system = "qa-ms002-hostile-escape"
    fmt = "change-focused"
    dump = tmp_path / "dump"
    outside = tmp_path / "outside"
    outside.mkdir()
    canary = outside / "canary.log"
    sid = session_id(system)
    sid_dir = os.path.join(str(dump), sid)
    os.makedirs(sid_dir)
    os.symlink(str(canary), os.path.join(sid_dir, f"{fmt}.log"))

    proxy = Proxy(
        dump,
        tmp_path / "logs",
        extra_args=[
            "--dump-format", fmt,
            "--dump-flush-interval", str(TIMER_OFF_INTERVAL),
            "--dump-flush-bytes", "1",
        ],
    )
    try:
        st, body = proxy.post(turn_body(system, "escape sentinel-es"))
        assert_upstream_response(st, body)
        assert not canary.exists(), "the flush must not write through the symlinked artifact"
        assert sorted(os.listdir(sid_dir)) == [f"{fmt}.log"], "session dir unchanged"
        assert os.path.islink(os.path.join(sid_dir, f"{fmt}.log"))
        st, body = proxy.post(turn_body(system, "escape sentinel-es-2"))
        assert_upstream_response(st, body)
    finally:
        rc = proxy.stop()
    # rc == 0 is the containment discriminator: the refused write is logged
    # but must not escape on_cleanup, and the canary must stay untouched.
    assert rc == 0
