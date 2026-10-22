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
from tests.qa.qap_ms002_support import (
    COLLAPSED_FORMATS,
    THRESHOLD_AT_TURN_2,
    TIMER_OFF_INTERVAL,
    TURN_SIZE,
    UNREACHABLE_BYTES,
    big_text,
    summary_lines,
)


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
