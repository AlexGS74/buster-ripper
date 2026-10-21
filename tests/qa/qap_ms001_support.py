"""Drift-observation helpers for the MS-001 documented-format gates."""

from __future__ import annotations

import difflib
import json
import os
import re
import time

import pytest

from tests.qa.harness import read_text, record_section

COLLAPSED = "change-focused"


def documented_files(n: int) -> set[str]:
    """Today's layout for a session of ``n`` turns: ``N + 1`` files."""
    return {"turn_000.json", "index.txt"} | {f"turn_{i:03d}.diff" for i in range(1, n)}

# The published reader contract. README.md §Reading a collapsed artifact documents the
# pattern a human/grep consumer matches dump lines with; §Reading the index documents the
# columns. These are the only sources for "today's line format" a black-box gate may use.
README_LINE_PATTERN = re.compile(r"^turn [0-9]{3}  [0-9]{2}:[0-9]{2}")
FULL_CLAUSE = re.compile(r"\bFULL\b")
KB_CLAUSE = re.compile(r"\b\d+(?:\.\d+)? KB\b")
CHANGE_CLAUSE = re.compile(r"\+\d+/-\d+ lines\b")
MSGS_CLAUSE = re.compile(r"\b\d+ msgs\b")
TOOLS_CLAUSE = re.compile(r"\b\d+ tools\b")


@pytest.fixture(autouse=True)
def no_dump_config_override(monkeypatch):
    """``VP-T002-default-encoding`` requires *no config override*, not merely no CLI flag."""
    for name in [k for k in os.environ if "DUMP" in k.upper()]:
        monkeypatch.delenv(name, raising=False)


def marker(tag: str, i: int) -> str:
    return f"{tag}-t{i}-uniq{int(time.monotonic() * 1000) % 100000:05d}-{i}"


def nonce(tag: str, i: int) -> str:
    """A per-request billing nonce (``README.md`` §Normalizations 1), unique per turn."""
    return f"qa-nonce-{tag}-{i}-{int(time.monotonic() * 1000) % 100000:05d}"


def index_lines(sid_dir: str) -> list[str]:
    text = read_text(os.path.join(sid_dir, "index.txt"))
    return [ln for ln in text.splitlines() if ln.strip()]


def production_diff_text(previous: str, current: str) -> str:
    """The documented writer's published diff text for two public renderings.

    ``README.md:165`` publishes ``turn_NNN.diff`` as a unified diff, so its
    ``---``/``+++``/``@@`` control lines are terminated and each changed line
    stands on its own line. The gate recomputes that text from the renderings alone;
    it never calls the implementation's own renderer or diff helper.
    """
    return "".join(difflib.unified_diff(
        previous.splitlines(keepends=True),
        current.splitlines(keepends=True),
        fromfile="previous", tofile="current", lineterm="\n",
    ))


def changed_lines(previous: str, current: str) -> list[str]:
    """The ``-``/``+`` content lines of ``turn NNN-1 -> turn NNN``, per README.md:165."""
    return diff_changed_lines(production_diff_text(previous, current))


def diff_changed_lines(diff_text: str) -> list[str]:
    """The ``-``/``+`` content lines of a well-formed unified diff.

    The leading ``---``/``+++`` file-header pair is control text; every later
    line beginning with ``+``/``-`` is a changed content line. A removed/added
    line whose own text begins with ``--``/``++`` stays content, because only the
    leading pair is control text.
    """
    return [ln for ln in _content_lines(diff_text) if ln and ln[0] in "+-"]


def _content_lines(diff_text: str) -> list[str]:
    """Split a well-formed unified diff after its leading file-header pair."""
    return diff_text.splitlines(keepends=True)[2:]


def assert_wellformed_diff(diff_text: str, name: str) -> None:
    """The ``---``/``+++``/``@@`` control lines stand on their own terminated lines.

    A writer whose control lines are glued to the line that follows is not a
    unified diff: its first physical line carries the file-header pair and the hunk
    header, and a hunk header runs into its first changed line (``S21`` N5).
    """
    lines = diff_text.splitlines(keepends=True)
    assert len(lines) >= 3, f"{name} is not a unified diff: {diff_text!r}"
    assert re.fullmatch(r"--- [^\n]*\n", lines[0]) and "+++" not in lines[0], (
        f"{name} does not open with a terminated file header on its own line: {lines[0]!r}"
    )
    assert re.fullmatch(r"\+\+\+ [^\n]*\n", lines[1]), (
        f"{name}'s added-file header is not its own terminated line: {lines[1]!r}"
    )
    hunks = [ln for ln in lines if ln.startswith("@@")]
    assert hunks, f"{name} has no hunk header"
    assert all(re.fullmatch(r"@@ [^\n]* @@\n", ln) for ln in hunks), (
        f"{name} has a hunk header that is not terminated on its own line: {hunks!r}"
    )


def rendering_section(artifact_text: str, header: str) -> str:
    """One turn's full rendering from a collapsed artifact, separator removed.

    The artifact separates records with ``\\n\\n`` and closes the last record with
    ``\\n``; the rendering itself always ends with ``\\n``. Remove only those
    separator newlines — never ``.strip()``, which erases a trailing newline the
    production diff reports (``S21`` CE-3).
    """
    start = artifact_text.find(header) + len(header)
    section = record_section(artifact_text, header)
    assert section.startswith("\n"), (
        f"record {header!r} must follow its header's line terminator"
    )
    body = section[1:]
    separator = "\n" if start + len(section) == len(artifact_text) else "\n\n"
    assert body.endswith(separator), (
        f"record {header!r} is not followed by the artifact's record separator"
    )
    return body[: -len(separator)]


def _assert_turn_zero_body(
    sid_dir: str, markers: list[str], expected_body: dict | None
) -> None:
    # turn_000.json — "full normalized request body (turn 0)"
    body = json.loads(read_text(os.path.join(sid_dir, "turn_000.json")))
    assert isinstance(body, dict) and body.get("model"), "turn_000.json is not a request body"
    assert markers[0] in json.dumps(body), "turn_000.json lost the turn's content"
    if expected_body is not None:
        missing = sorted(set(expected_body) - set(body))
        unexpected = sorted(set(body) - set(expected_body))
        differing = sorted(k for k in set(body) & set(expected_body) if body[k] != expected_body[k])
        assert body == expected_body, (
            "turn_000.json is not the normalized request body: "
            f"missing={missing} unexpected={unexpected} differing={differing}"
        )


def _assert_index_line_shape(i: int, line: str) -> None:
    assert README_LINE_PATTERN.match(line), (
        f"index.txt line {i} does not match the README reader pattern "
        f"{README_LINE_PATTERN.pattern!r}: {line!r}"
    )
    cols = line.split()
    assert cols[0] == "turn" and cols[1] == f"{i:03d}", (
        f"documented line {i} lost its turn number: {line!r}"
    )
    assert MSGS_CLAUSE.search(line), f"documented line lost its msgs column: {line!r}"
    assert TOOLS_CLAUSE.search(line), f"documented line lost its tools column: {line!r}"


def _assert_turn_columns(i: int, line: str) -> None:
    if i == 0:
        assert FULL_CLAUSE.search(line), f"turn 000 lost the documented FULL marker: {line!r}"
        assert KB_CLAUSE.search(line), f"turn 000 lost the documented size column: {line!r}"
    else:
        assert CHANGE_CLAUSE.search(line), (
            f"turn {i:03d} lost the documented +A/-R lines column: {line!r}"
        )


def _assert_index_lines(sid_dir: str, markers: list[str]) -> None:
    # index.txt — "one-line summary per turn", in the documented shape
    lines = index_lines(sid_dir)
    assert len(lines) == len(markers), f"index.txt must carry one line per turn: {lines!r}"
    for i, line in enumerate(lines):
        _assert_index_line_shape(i, line)
        _assert_turn_columns(i, line)


def assert_documented_contents(
    sid_dir: str,
    markers: list[str],
    expected_body: dict | None = None,
    renderings: list[str] | None = None,
) -> None:
    """Every clause of ``VP-T002-documented-format-preserved``'s "today's content and line
    format", read off ``README.md`` §Dump mode — the public contract, not the implementation.

    ``expected_body`` is the **complete** normalized request body the gate expects in
    ``turn_000.json`` (``README.md:164``: "full normalized request body"). ``renderings``
    are the ``self-contained`` artifact's per-turn renderings (``README.md:199``), from which
    the documented diff's source is derived (``README.md:165``). Both are supplied by the
    caller, so no expectation is produced by the code under test.
    """
    _assert_turn_zero_body(sid_dir, markers, expected_body)
    _assert_index_lines(sid_dir, markers)

    # turn_NNN.diff — "unified diff: turn NNN-1 -> turn NNN". Raw read: the
    # writer computed this text before the write, over the renderings' own line
    # endings, so the comparison must not normalize them (S21 CE-2).
    for i in range(1, len(markers)):
        diff = read_text(os.path.join(sid_dir, f"turn_{i:03d}.diff"), newline="")
        assert_wellformed_diff(diff, f"turn_{i:03d}.diff")
        assert any(ln.startswith("+") and markers[i] in ln for ln in diff.splitlines()), (
            f"turn_{i:03d}.diff does not add the turn's own content"
        )
        assert any(ln.startswith("-") and markers[i - 1] in ln for ln in diff.splitlines()), (
            f"turn_{i:03d}.diff does not remove the previous turn's content"
        )
        if renderings is not None:
            expected = changed_lines(renderings[i - 1], renderings[i])
            observed = diff_changed_lines(diff)
            assert observed == expected, (
                f"turn_{i:03d}.diff is not computed over the turn's rendering: "
                f"expected={expected!r} observed={observed!r}"
            )


def count_report(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def ops_by_name(report: dict) -> dict[str, int]:
    out: dict[str, int] = {}
    for op in report["ops"]:
        out[os.path.basename(op["path"])] = out.get(os.path.basename(op["path"]), 0) + 1
    return out
