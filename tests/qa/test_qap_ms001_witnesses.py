"""Sensitivity witnesses for the MS-001 documented-format gates."""

from __future__ import annotations

import difflib
import json
import urllib.request

import pytest

from tests.qa.harness import (
    assert_upstream_response,
    billing_system,
    listing,
    read_text,
    turn_body,
)
from tests.qa.harness_process import (
    Proxy,
    free_port,
)
from tests.qa.qap_ms001_support import (
    CHANGE_CLAUSE,
    FULL_CLAUSE,
    KB_CLAUSE,
    MSGS_CLAUSE,
    README_LINE_PATTERN,
    TOOLS_CLAUSE,
    assert_wellformed_diff,
    changed_lines,
    diff_changed_lines,
    documented_files,
    no_dump_config_override,
    production_diff_text,
    rendering_section,
)
from tests.qa.test_qap_ms001 import N_TURNS


def test_tc108_sensitivity_diff_source_and_normalized_body_distinguish_drift():
    """Check 4 without a process: the assertions fail on both ``YR-QA-1`` content drifts.

    Red when the diff comparison stops comparing changed lines, or the body comparison
    accepts a body that still carries the billing nonce.
    """
    renderings = [
        "\n[system]\nqa-clean\n\n[message 0 / user]\nfirst\n",
        "\n[system]\nqa-clean\n\n[message 0 / user]\nsecond\n",
    ]
    expected = changed_lines(renderings[0], renderings[1])
    assert expected == ["-first\n", "+second\n"], expected

    # YR-QA-1's second arm: a diff computed over raw request JSON bodies. It carries
    # every unified-diff marker the old gate accepted, and reports +1/-1 — the count the
    # index's CHANGE_CLAUSE also accepts. Only the source comparison distinguishes it.
    first = json.dumps(
        {"model": "qa", "system": "qa-clean", "messages": [{"role": "user", "content": "first"}]}
    )
    second = json.dumps(
        {"model": "qa", "system": "qa-clean", "messages": [{"role": "user", "content": "second"}]}
    )
    raw_diff = "\n".join(difflib.unified_diff([first], [second], lineterm="")) + "\n"
    assert "---" in raw_diff and "+++" in raw_diff and "@@" in raw_diff
    assert diff_changed_lines(raw_diff) == [f"-{first}\n", f"+{second}\n"]
    assert diff_changed_lines(raw_diff) != expected, (
        "a raw-request-body diff must not satisfy the rendering comparison"
    )

    # YR-QA-1's first arm: the pre-normalization body. A nonce-bearing system
    # differs from the nonce-free body the sid is keyed on and must not compare equal.
    clean = {
        "model": "qa",
        "system": "qa-clean",
        "messages": [{"role": "user", "content": "first"}],
    }
    drifted = {**clean, "system": billing_system("qa-clean", "qa-nonce-01")}
    assert drifted != clean, "the body comparison must reject a nonce-bearing body"
    assert "qa-nonce-01" in drifted["system"] and "qa-nonce-01" not in clean["system"]


def test_tc108_sensitivity_documented_line_pattern_distinguishes_drift():
    """Check 4 without a process: the assertions above can actually fail.

    The matcher must accept every line of the README's own published example and reject the
    drift forms ``VP-T002-documented-format-preserved`` exists to catch.
    """
    published = [
        "turn 000  10:31:02  FULL  3 msgs  83 tools  412.3 KB",
        "turn 001  10:32:18  +47/-3 lines  5 msgs  83 tools",
        "turn 002  10:33:05  +892/-0 lines  7 msgs  83 tools   ← large file read",
    ]
    for line in published:
        assert README_LINE_PATTERN.match(line), line
        assert MSGS_CLAUSE.search(line) and TOOLS_CLAUSE.search(line), line
    assert FULL_CLAUSE.search(published[0]) and KB_CLAUSE.search(published[0])
    assert all(CHANGE_CLAUSE.search(l) for l in published[1:])

    drift = [
        "turn 000  10:31:02  3 msgs  83 tools  412.3 KB",   # FULL marker gone
        "turn 0  10:31:02  FULL  3 msgs  83 tools  412.3 KB",  # unpadded turn number
        "turn 001  +47/-3 lines  5 msgs  83 tools",         # timestamp column gone
        "turn 001  10:32:18  5 msgs  83 tools",             # change column gone
        "turn 000  10:31:02  FULL  3 msgs",                 # tools column gone
    ]
    for line in drift:
        assert not README_LINE_PATTERN.match(line) or not (
            (FULL_CLAUSE.search(line) and KB_CLAUSE.search(line))
            if "000" in line.split()[1]
            else (CHANGE_CLAUSE.search(line) and TOOLS_CLAUSE.search(line))
        ), f"the drift form {line!r} was accepted"
    for name in ("change-focused.log", "self-contained.log"):
        assert name not in documented_files(N_TURNS)


def test_sensitivity_upstream_oracle_distinguishes_transport_failure_from_a_relayed_body():
    """Check 4 for ``assert_upstream_response``, the oracle every scenario in this pass
    states its trigger through.

    No process: the helper is a pure predicate over ``(status, body)``. Red when the status
    limb is loosened past 500, or the dropped ``"Cannot connect" not in body`` limb is
    restored (``QB:qa-harness/static-body-cannot-connect-limb``; chronicle 077).
    """
    # The proxy's own transport failure: answered 502, and it must not pass.
    for status, body in (
        (500, b"Internal Server Error"),
        (502, b'{"error": "Upstream request failed"}'),
        (503, b'{"error": "Upstream request failed"}'),
    ):
        with pytest.raises(AssertionError, match="valid request failed"):
            assert_upstream_response(status, body)

    # An answer that came from the peer — including the peer's own client error, which the
    # live service returns for a test model name, and which is explicitly not a QA oracle.
    for status, body in (
        (200, b'{"id":"chatcmpl-1","choices":[]}'),
        (400, b'{"error":"invalid model"}'),
        (404, b'{"error":"The model `qa-gate-model` does not exist"}'),
        (429, b"rate limited"),
    ):
        assert_upstream_response(status, body)

    # The exact shape the removed limb mis-attributed: a body naming a connection failure,
    # relayed from the peer behind a status the proxy did not produce. Not a proxy fault.
    assert_upstream_response(200, b"upstream said: Cannot connect to its own backend")
    assert_upstream_response(404, b"Cannot connect")


def _synthetic_self_contained(renderings: list[str]) -> str:
    """A self-contained artifact with the writer's published record separators."""
    records = [
        f"turn {i:03d}  10:31:02  FULL  1 msgs  0 tools  0.0 KB\n"
        f"--- turn {i:03d} body ---\n{{}}\n"
        f"--- turn {i:03d} rendered ---\n{rendering}"
        for i, rendering in enumerate(renderings)
    ]
    return "\n\n".join(records) + "\n"


def test_sensitivity_s21_ce1_glued_control_line_is_rejected():
    """S21 CE-1/N5: a glued control line is not a unified diff and must fail the gate."""
    previous = "\n\n\n"
    current = (
        "\n\n[message 0 / user]\na\n\n\n\n"
        + "─" * 80
        + "\n\n[message 1 / user]\n\n"
    )
    wellformed = production_diff_text(previous, current)
    assert_wellformed_diff(wellformed, "turn_001.diff")
    expected = changed_lines(previous, current)
    assert "+\n" in expected, "the blank inserted line is a changed line"

    # The pre-repair writer: ``lineterm=""`` glues each control line to the line
    # that follows it, so the artifact is not a unified diff.
    glued = "".join(difflib.unified_diff(
        previous.splitlines(keepends=True),
        current.splitlines(keepends=True),
        fromfile="previous", tofile="current", lineterm="",
    ))
    with pytest.raises(AssertionError, match="turn_001.diff"):
        assert_wellformed_diff(glued, "turn_001.diff")

    # A mid-diff removed line whose text begins with ``--`` is content, not a
    # second file header; dropping the rest of the diff there would hide changes.
    hostile = "--- previous\n+++ current\n@@ -1,2 +1,2 @@\n--- marker\n+kept\n"
    assert diff_changed_lines(hostile) == ["--- marker\n", "+kept\n"]


def test_sensitivity_s21_ce2_line_endings_are_line_identity():
    """S21 CE-2: CRLF versus LF is a changed line under the writer's identity."""
    previous = "\n\n[message 0 / user]\nline1\r\nline2\n"
    current = "\n\n[message 0 / user]\nline1\nline2\n"
    expected = changed_lines(previous, current)
    observed = diff_changed_lines(production_diff_text(previous, current))
    assert expected == observed, (expected, observed)
    assert "-line1\r\n" in expected and "+line1\n" in expected, expected

    # A writer that normalizes line endings must not pass.
    mutated = production_diff_text(previous, current).replace("-line1\r\n", "-line1\n")
    assert diff_changed_lines(mutated) != expected


def test_sensitivity_s21_ce2_artifact_read_preserves_line_endings(tmp_path):
    """S21 CE-2: the gate's artifact read must keep the renderings' line endings.

    Production computes its diff over the raw renderings *before* the write; the gate derives
    the expected diff from the ``self-contained`` artifact afterwards, so a universal-newline
    read would let a line-ending-only change arrive equal on the expected side.
    """
    previous = "\n\n[message 0 / user]\nline1\r\nline2\n"
    current = "\n\n[message 0 / user]\nline1\nline2\n"
    artifact = tmp_path / "self-contained.log"
    artifact.write_text(
        _synthetic_self_contained([previous, current]), encoding="utf-8", newline=""
    )
    diff_file = tmp_path / "turn_001.diff"
    diff_file.write_text(
        production_diff_text(previous, current), encoding="utf-8", newline=""
    )

    # The repaired path: both sides carry the writer's own line identity.
    raw = read_text(str(artifact), newline="")
    recovered = [
        rendering_section(raw, f"--- turn {i:03d} rendered ---") for i in range(2)
    ]
    expected = changed_lines(recovered[0], recovered[1])
    observed = diff_changed_lines(read_text(str(diff_file), newline=""))
    assert expected == observed == ["-line1\r\n", "+line1\n"], (expected, observed)
    assert recovered == [previous, current], recovered

    # The pre-repair path — universal newlines on both artifact reads — is exactly
    # the CE-2 failure shape, and must stay distinguishable from the repaired one.
    normalized = [
        rendering_section(
            read_text(str(artifact)), f"--- turn {i:03d} rendered ---"
        )
        for i in range(2)
    ]
    assert normalized == [previous.replace("\r\n", "\n")] * 2, normalized
    assert changed_lines(normalized[0], normalized[1]) == []
    assert diff_changed_lines(read_text(str(diff_file))) == ["-line1\n", "+line1\n"]

    # A writer that drops the line-ending change must not pass.
    mutated = production_diff_text(recovered[0], recovered[1]).replace(
        "-line1\r\n", "-line1\n"
    )
    assert diff_changed_lines(mutated) != expected


def test_sensitivity_s21_ce3_renderings_keep_their_trailing_newline():
    """S21 CE-3: a trailing-newline-only change is visible in both directions.

    The artifact adds its own record-separator newlines; only those are removed.
    """
    previous = "\n\n[message 0 / user]\nfoo\n"
    current = "\n\n[message 0 / user]\nfoo\n\n"
    artifact = _synthetic_self_contained([previous, current])
    recovered = [
        rendering_section(artifact, f"--- turn {i:03d} rendered ---")
        for i in range(2)
    ]
    assert recovered == [previous, current], recovered

    expected = changed_lines(recovered[0], recovered[1])
    observed = diff_changed_lines(production_diff_text(recovered[0], recovered[1]))
    assert expected == observed == ["+\n"], (expected, observed)

    # A writer that omits the trailing-newline change must not pass.
    mutated = production_diff_text(recovered[0], recovered[1])[: -len("+\n")]
    assert diff_changed_lines(mutated) != expected


def test_sensitivity_s22_proxy_post_ignores_ambient_proxy_settings(tmp_path, monkeypatch):
    """S22: the oracle's client hop must reach the local proxy, not an ambient proxy.

    ``http_proxy`` points at a dead port and the cached default opener is dropped, so only
    ``Proxy.post``'s own ``ProxyHandler({})`` opener reaches the local proxy. The dump proves
    the local proxy is the host that answered.
    """
    system = "qa-ms001-ambient-proxy"
    dump = tmp_path / "dump"
    dead_port = free_port()  # nothing listens there
    dead = f"http://127.0.0.1:{dead_port}"
    monkeypatch.setenv("http_proxy", dead)
    monkeypatch.setenv("HTTP_PROXY", dead)
    monkeypatch.delenv("no_proxy", raising=False)
    monkeypatch.delenv("NO_PROXY", raising=False)
    monkeypatch.setattr(urllib.request, "_opener", None, raising=False)

    proxy = Proxy(dump, tmp_path / "logs")  # documented: the per-turn write proves the host
    try:
        assert_upstream_response(*proxy.post(turn_body(system, "ambient-proxy sentinel")))
    finally:
        rc = proxy.stop()
    assert rc == 0
    assert listing(proxy.session_dir(system)) == sorted(documented_files(1)), (
        "the local proxy must be the host that answered and dumped the turn"
    )
