"""QAP-MS001 — MS-001 integration gates: the format selector and the documented path.

Covers (``qa-plan.md`` §QA Gate Matrix / §Scenario Contracts):

- TC-100 ``VP-T001-writes-deferred`` — a collapsed format defers every write until a flush
- TC-001 ``VP-T001-dump-failure-isolated`` — a raising dump/flush still yields the upstream
  response body, on the documented write path and on the collapsed flush path
- TC-108 ``VP-T002-documented-format-preserved`` — today's three files, today's content and
  line format, at ``2N`` write-bearing operations (``G-008``(a)'s only witness)
- TC-107 ``VP-T002-default-encoding`` — nothing configured produces the documented layout
- Hostile tier (``codify-adversarial-surface-discipline`` R1/R2): unknown selector values, the
  request-body parse boundary, and the **documented** writer's symlink discipline — R2's
  sibling audit of the writer ``MS-002`` did not cover
- Identity tier (R5): the near-miss ``sid`` pair the session key must not merge, and the
  ``(sid, format)`` identity against a pre-existing collapsed artifact

Re-entered at 077-forward-qa (``.sessions/codify/<this session>/iterations/077-forward-qa.md``) to
drain the two ``forward-qa``-owned backlog rows whose named trigger is a touch of
``tests/qa/harness.py`` (``QB:qa-harness/unexercised-path-keyword``,
``QB:qa-harness/static-body-cannot-connect-limb``).
The gates above are unchanged by that drain; the one test added for it is
``test_sensitivity_upstream_oracle_distinguishes_transport_failure_from_a_relayed_body``, the
process-free witness that the shared oracle still refuses a proxy-local transport failure after
its body limb was dropped.

Re-entered at 096-forward-qa for ``T021`` (``YR-QA-1``): ``assert_documented_contents`` now
compares ``turn_000.json`` against the **complete** normalized request body and each
``turn_NNN.diff`` against a diff derived from the ``self-contained`` artifact's public per-turn
renderings — the documented contract's own rendering — so the two content-drift shapes
``YR-QA-1`` named fail this gate. ``TC-107``/``TC-108`` drive nonce-bearing bodies so the
route's normalize-then-dump ordering is observable; every nonce-free scenario keeps its body
bytes and ``sid`` derivation unchanged.

Re-entered at 099-forward-qa for the run-49 gate-repair wave (``T021.001``/``T021.002``,
``S21`` CE-1/CE-2/CE-3 and ``S22``): the changed-line comparison now uses the writer's
own line identity on both sides — ``splitlines(keepends=True)``, the ``""`` join, and one
structural reader that recovers a hunk's first changed line from the ``@@`` control line — and
the renderings keep their trailing newline instead of being ``.strip()``ped. ``Proxy.post`` builds
its opener with ``ProxyHandler({})`` so the oracle judges the local proxy under ambient proxy
settings. No scenario semantics changed.

Re-entered at 108-forward-qa for ``T021.003`` (``S21`` CE-2 residual): the gate's
expected side reads the ``self-contained`` artifact's renderings and the observed
``turn_NNN.diff`` in **raw** text mode (``read_text(..., newline="")``), so the expected
diff is computed over the same line endings production diffed before the write. The
focused regression round-trips a CRLF rendering through the writer's artifact shape and
is red on the pre-repair universal-newline read. No scenario semantics changed.

**Contract directive honored** (``qa-plan.md`` §Phase Requirements, 022): no gate in this file
passes ``--dump-flush-interval`` or ``--dump-flush-bytes``. TC-100 therefore bounds its
pre-flush window by *observation* (the artifact is append-only and never removed, so a
snapshot with no artifact is, by that very fact, a pre-flush snapshot) rather than by a
sleep.

Run explicitly (NOT part of the hermetic default suite)::

    .venv/bin/python -m pytest tests/qa/test_qap_ms001.py -o addopts=

The declared live upstream is required; gates fail closed on the harness preflight and assert
on dump artifacts and process behavior, never on model output.
"""

from __future__ import annotations

import difflib
import json
import os
import re
import time
import urllib.request

import pytest

from tests.qa.harness import (
    Proxy,
    assert_upstream_response,
    billing_system,
    free_port,
    log_artifacts,
    poll_until,
    read_text,
    record_section,
    rglob_files,
    session_id,
    startup_refused,
    turn_body,
)

COLLAPSED = "change-focused"
N_TURNS = 3


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


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

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


def listing(path: str) -> list[str]:
    return sorted(os.listdir(path)) if os.path.isdir(path) else []


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

    # index.txt — "one-line summary per turn", in the documented shape
    lines = index_lines(sid_dir)
    assert len(lines) == len(markers), f"index.txt must carry one line per turn: {lines!r}"
    for i, line in enumerate(lines):
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
        if i == 0:
            assert FULL_CLAUSE.search(line), f"turn 000 lost the documented FULL marker: {line!r}"
            assert KB_CLAUSE.search(line), f"turn 000 lost the documented size column: {line!r}"
        else:
            assert CHANGE_CLAUSE.search(line), (
                f"turn {i:03d} lost the documented +A/-R lines column: {line!r}"
            )

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


# ---------------------------------------------------------------------------
# TC-100 — VP-T001-writes-deferred: a collapsed format defers every write
# ---------------------------------------------------------------------------

def test_tc100_collapsed_format_defers_every_write_until_a_flush(tmp_path):
    system = "qa-ms001-defer"
    dump = tmp_path / "dump"
    report_path = tmp_path / "report-tc100.json"
    proxy = Proxy(
        dump,
        tmp_path / "logs",
        extra_args=["--dump-format", COLLAPSED],  # no flush parameter: the directive stands
        counted_report=report_path,
    )
    sid_dir = proxy.session_dir(system)
    artifact = proxy.artifact_path(system, COLLAPSED)
    markers = [marker("defer", i) for i in range(N_TURNS)]
    pre_flush_snapshots: list[tuple[float, list[str]]] = []
    try:
        t0 = time.monotonic()
        for i, m in enumerate(markers):
            assert_upstream_response(*proxy.post(turn_body(system, m)))
            # The artifact is append-only and never removed, so a snapshot holding no
            # artifact IS a pre-flush snapshot: the absence is a direct observation.
            has_artifact = bool(log_artifacts(str(dump)))
            if not has_artifact:
                snapshot = listing(sid_dir)
                pre_flush_snapshots.append((time.monotonic() - t0, snapshot))
                assert snapshot == [], (
                    f"turn {i} was acknowledged with {snapshot!r} already on disk — "
                    f"a collapsed format must defer every write to a flush"
                )
        assert pre_flush_snapshots, (
            "no pre-flush snapshot was observable: the first flush beat the first request "
            "acknowledgement, so the deferral cannot be witnessed in this environment"
        )
        # The contrast the absence assertion needs is the post-run state below: the same
        # directory, once a flush has run, holds exactly one artifact carrying those turns.
    finally:
        rc = proxy.stop()

    assert rc == 0, f"process must stay healthy across the deferral, got rc={rc}"
    # Nothing per-turn was ever produced on the collapsed path (the VP's named witness).
    assert listing(str(dump)) == [session_id(system)]
    assert listing(sid_dir) == [f"{COLLAPSED}.log"], (
        f"collapsed path must hold one artifact and no documented file: {listing(sid_dir)!r}"
    )
    assert not [p for p in rglob_files(str(dump)) if not p.endswith(".log")], (
        "a documented-layout file appeared on the collapsed path"
    )
    # The deferral ends at a flush, and the flush writes the buffered turns into that one file.
    text = read_text(artifact)
    for i, m in enumerate(markers):
        assert m in text, f"buffered turn {i} was lost, never written"
    report = count_report(str(report_path))
    assert report["ops"], "the collapsed path wrote nothing at all"
    names = set(ops_by_name(report))
    assert names == {f"{COLLAPSED}.log"}, (
        f"every write-bearing operation must target the single collapsed artifact: {names}"
    )
    assert len(log_artifacts(str(dump))) == 1, "one artifact per session, not per pending turn"
    print(
        f"[TC-100] pre-flush snapshots={[(round(t, 2), s) for t, s in pre_flush_snapshots]} "
        f"turns={N_TURNS} write_ops={report['total_ops']} write_bytes={report['total_bytes']}"
    )


# ---------------------------------------------------------------------------
# TC-107 — VP-T002-default-encoding: nothing configured is the documented layout
# ---------------------------------------------------------------------------

def test_tc107_no_format_configured_produces_the_documented_layout(tmp_path):
    system = "qa-ms001-default"
    dump = tmp_path / "dump"
    proxy = Proxy(dump, tmp_path / "logs")  # no --dump-format: the default clause under test
    sid_dir = proxy.session_dir(system)  # the sid keys on the nonce-free system
    markers = [marker("default", i) for i in range(N_TURNS)]
    nonces = [nonce("default", i) for i in range(N_TURNS)]
    bodies = [turn_body(system, m, nonce=n) for m, n in zip(markers, nonces)]
    try:
        assert not os.path.isdir(sid_dir), "sensitivity: no session dir before any request"
        for body in bodies:
            assert_upstream_response(*proxy.post(body))
    finally:
        rc = proxy.stop()
    assert rc == 0

    # Witness W1 of G-008, the exact shape US-012 forbids: a collapsed artifact with
    # nothing configured.
    assert log_artifacts(str(dump)) == [], (
        "a collapsed .log artifact appeared with no --dump-format configured"
    )
    assert listing(sid_dir) == sorted(documented_files(N_TURNS)), (
        f"the default must be today's three files: {listing(sid_dir)!r}"
    )
    # "Full normalized request body" (README.md:164): the complete value set, with the
    # nonce-bearing system replaced by the nonce-free text the sid is keyed on.
    assert_documented_contents(
        sid_dir, markers, expected_body={**bodies[0], "system": system}
    )
    for n in nonces:
        for name in listing(sid_dir):
            assert n not in read_text(os.path.join(sid_dir, name)), (
                f"the billing nonce survived into {name}: the default path must dump the "
                "normalized body"
            )


# ---------------------------------------------------------------------------
# TC-108 — VP-T002-documented-format-preserved: layout, contents, and the 2N count
# ---------------------------------------------------------------------------

def test_tc108_documented_format_preserves_contents_at_2n_writes(tmp_path):
    system = "qa-ms001-documented"
    dump = tmp_path / "dump"
    report_path = tmp_path / "report-tc108.json"
    proxy = Proxy(
        dump,
        tmp_path / "logs",
        extra_args=["--dump-format", "documented"],
        counted_report=report_path,
    )
    sid_dir = proxy.session_dir(system)
    markers = [marker("documented", i) for i in range(N_TURNS)]
    nonces = [nonce("documented", i) for i in range(N_TURNS)]
    bodies = [turn_body(system, m, nonce=n) for m, n in zip(markers, nonces)]

    # The rendering oracle. `self-contained` stores every turn's complete rendering
    # (README.md:199), so a documented diff over consecutive renderings is derivable
    # from a public artifact with no call into the implementation's renderer. Its own
    # dump root keeps this scenario's documented listing unambiguous.
    sc_dump = tmp_path / "dump-renderings"
    sc_proxy = Proxy(
        sc_dump,
        tmp_path / "logs-renderings",
        extra_args=["--dump-format", "self-contained"],
    )
    sc_sid_dir = sc_proxy.session_dir(system)
    try:
        assert listing(sid_dir) == [], "sensitivity: nothing on disk before the first turn"
        for body in bodies:
            assert_upstream_response(*proxy.post(body))
        for body in bodies:
            assert_upstream_response(*sc_proxy.post(body))
    finally:
        rc = proxy.stop()
        sc_rc = sc_proxy.stop()
    assert rc == 0
    assert sc_rc == 0

    assert listing(sid_dir) == sorted(documented_files(N_TURNS)), (
        f"today's layout is N + 1 files: {listing(sid_dir)!r}"
    )
    assert listing(sc_sid_dir) == ["self-contained.log"], (
        f"the rendering oracle must hold its one artifact: {listing(sc_sid_dir)!r}"
    )
    # Raw read: these renderings are the expected side of the diff comparison and
    # must carry the line endings production diffed before the write (S21 CE-2).
    sc_text = read_text(os.path.join(sc_sid_dir, "self-contained.log"), newline="")
    renderings = [
        rendering_section(sc_text, f"--- turn {i:03d} rendered ---")
        for i in range(N_TURNS)
    ]
    assert_documented_contents(
        sid_dir,
        markers,
        expected_body={**bodies[0], "system": system},
        renderings=renderings,
    )
    for n in nonces:
        for root in (sid_dir, sc_sid_dir):
            for name in listing(root):
                assert n not in read_text(os.path.join(root, name)), (
                    f"the billing nonce survived into {name}: every writer must dump the "
                    "normalized body"
                )
    assert log_artifacts(str(dump)) == [], "no .log artifact on the documented path"

    # The instrumented measure M1: 2N write-bearing operations, decomposed so a
    # rewrite-per-turn or a lost index write cannot hide behind the total.
    report = count_report(str(report_path))
    assert report["total_ops"] == 2 * N_TURNS, (
        f"documented format must cost 2N={2 * N_TURNS} write ops, got {report['total_ops']}: "
        f"{ops_by_name(report)}"
    )
    assert ops_by_name(report) == {
        "index.txt": N_TURNS,
        "turn_000.json": 1,
        "turn_001.diff": 1,
        "turn_002.diff": 1,
    }, f"per-path write decomposition drifted: {ops_by_name(report)}"
    root = os.path.abspath(str(sid_dir))
    assert all(os.path.dirname(os.path.abspath(o["path"])) == root for o in report["ops"]), (
        "a documented write escaped this session's directory"
    )
    assert sum(o["bytes"] for o in report["ops"]) > 0
    # The documented half of the G-008(b) pair (MS-002's TC-105 supplies the collapsed half).
    print(
        f"[TC-108 documented half] N={N_TURNS} format=documented "
        f"write_ops={report['total_ops']} write_bytes={report['total_bytes']} "
        f"files={len(listing(sid_dir))} expected_ops=2N={2 * N_TURNS} "
        f"expected_files=N+1={N_TURNS + 1}"
    )


def test_tc108_sensitivity_diff_source_and_normalized_body_distinguish_drift():
    """Check 4 without a process: the strengthened assertions can fail on exactly the
    content drifts ``YR-QA-1`` measured.

    Red when: the diff comparison stops comparing changed lines (so any marker-bearing
    unified diff passes, ``yr-p002b``), or the body comparison accepts a body that still
    carries the billing nonce (``yr-p002a``).
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
    """Check 4 for the oracle every scenario in this pass states its trigger through.

    ``assert_upstream_response`` is the only place ``TC-100``/``TC-107``/``TC-108``/``TC-001``
    record that a request actually reached the peer, so the drain of
    ``QB:qa-harness/static-body-cannot-connect-limb`` has to show the surviving oracle still
    draws that line — and draws it in the right place. No process: the helper is a pure
    predicate over ``(status, body)``.

    Red when: the status limb is loosened past 500 (a gate then passes against a dead
    transport), or the dropped ``"Cannot connect" not in body`` limb is restored (a peer
    answer legitimately saying so then fails a proxy-side oracle) — ``QB:qa-harness`` row,
    073-01-yield §Findings D-2 (``QB:qa-harness/static-body-cannot-connect-limb``).
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
    """S21 CE-1/N5: a glued control line is not a unified diff and must fail the gate.

    Red before the operand: the coarse check asked only for the ``---``/``+++``/``@@``
    substrings, which the pre-repair writer's glued artifact carries, so the gate
    mirrored the malformed shape instead of rejecting it.
    """
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
    """S21 CE-2: CRLF versus LF is a changed line under the writer's identity.

    Red on the pre-repair gate: its expected side split with ``splitlines()`` and saw
    no change while the artifact reported ``-line1``/``+line1``.
    """
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

    Production computes its diff over the raw renderings *before* the write; the
    gate derives the expected diff from the ``self-contained`` artifact afterwards.
    A universal-newline read converts ``\\r\\n``/``\\r`` to ``\\n``, so a
    line-ending-only change arrives equal on the expected side while the written
    ``turn_NNN.diff`` still reports it: ``expected=[]`` against
    ``observed=['-line1\\n', '+line1\\n']``.

    Red on the pre-repair reader, which has no raw mode and reproduces that exact
    pair. Green when the gate reads both artifacts raw.
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
    Red on the pre-repair gate: ``.strip()`` erased the rendering's trailing
    newline, so the added blank line was invisible on the expected side.
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

    Sets ``http_proxy`` to a dead port with ``no_proxy`` unset and drops the
    cached default opener, so a pre-repair ``urlopen`` would rebuild it from that
    environment and send the request to the dead port. Red on the pre-repair
    harness (``URLError`` escapes); green when ``Proxy.post`` builds its own
    ``ProxyHandler({})`` opener. The dump proves the local proxy is the host that
    answered.
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


# ---------------------------------------------------------------------------
# TC-001 — VP-T001-dump-failure-isolated
# ---------------------------------------------------------------------------

def _read_only_session_dir(dump_root: str, system: str) -> str:
    """Fault injection at the process level: the session directory cannot be written."""
    sid_dir = os.path.join(os.path.abspath(str(dump_root)), session_id(system))
    os.makedirs(sid_dir)
    os.chmod(sid_dir, 0o500)  # r-x: every create/write inside raises PermissionError
    return sid_dir


def test_tc001_documented_write_failure_still_returns_the_upstream_response(tmp_path):
    broken = "qa-ms001-fault-documented"
    control = "qa-ms001-fault-control"
    dump = tmp_path / "dump"
    sid_dir = _read_only_session_dir(dump, broken)
    proxy = Proxy(dump, tmp_path / "logs")  # documented: the write path raises per turn
    try:
        assert_upstream_response(*proxy.post(turn_body(broken, "faulted-turn")))
        # A dump failure must not become a proxy failure: the same request on a healthy
        # session answers with the same status.
        status_broken = proxy.post(turn_body(broken, "faulted-turn-2"))[0]
        status_control = proxy.post(turn_body(control, "faulted-turn-2"))[0]
        assert status_broken == status_control, (
            f"the dump fault changed the response status: {status_broken} vs {status_control}"
        )
        assert status_control < 500
        ok = poll_until(lambda: "PermissionError" in proxy.stderr(), deadline=15)
        assert ok, "a dump failure must be logged, not swallowed silently"
        assert proxy.proc.poll() is None, "the process must keep serving"
    finally:
        os.chmod(sid_dir, 0o755)
        rc = proxy.stop()
    assert rc == 0
    # The healthy sibling session proves capture still works after the fault.
    assert listing(proxy.session_dir(control)) == sorted(documented_files(1)), (
        f"capture after the fault: {listing(proxy.session_dir(control))!r}"
    )


def test_tc001_collapsed_flush_failure_still_returns_the_upstream_response(tmp_path):
    system = "qa-ms001-fault-collapsed"
    dump = tmp_path / "dump"
    sid_dir = _read_only_session_dir(dump, system)
    proxy = Proxy(
        dump,
        tmp_path / "logs",
        extra_args=["--dump-format", COLLAPSED],  # no flush parameter: the flush is shutdown's
    )
    try:
        # The append path buffers in memory, so the raise surfaces at the flush; the
        # intercepted request must still be answered with the upstream's body.
        assert_upstream_response(*proxy.post(turn_body(system, "collapsed-faulted-turn")))
        assert listing(sid_dir) == [], "the collapsed path must not write at append time"
    finally:
        # The fault stays injected across the shutdown flush — that flush is the code under
        # test here. Clearing the read-only bit first would let the flush succeed.
        rc = proxy.stop()
    os.chmod(sid_dir, 0o755)
    assert rc == 0, f"a raising flush must not escape on_cleanup as a fatal exit, rc={rc}"
    assert "PermissionError" in proxy.stderr(), "the failing flush must be logged"


# ---------------------------------------------------------------------------
# Hostile tier R1 — the format selector's parse boundary (and the closed value set)
# ---------------------------------------------------------------------------

UNKNOWN_FORMATS = ["documentedd", "change_focused", "DOCUMENTED", " documented", ""]


def test_hostile_r1_unknown_dump_format_is_refused_and_writes_nothing(tmp_path):
    """A mistyped selector must not merge with the default identity TC-107 gates.

    If ``documentedd`` silently fell back to ``documented``, TC-107's default clause would
    become unfalsifiable from the CLI: the two identities would be one.
    """
    failures = []
    for i, value in enumerate(UNKNOWN_FORMATS):
        dump = tmp_path / f"dump-{i}"
        rc, err = startup_refused(["--dump-format", value], dump, tmp_path / f"logs-{i}")
        if rc == 0:
            failures.append(f"{value!r}: exited 0 — the value was accepted")
            continue
        if "documented" not in err or COLLAPSED not in err:
            failures.append(f"{value!r}: refusal did not name the allowed set: {err[-300:]!r}")
        if os.path.isdir(str(dump)) and rglob_files(str(dump)):
            failures.append(f"{value!r}: refused but wrote into {dump}: {rglob_files(str(dump))}")
    assert not failures, "\n".join(failures)


# ---------------------------------------------------------------------------
# Hostile tier R1 — the request-body parse boundary, on the documented writer
# ---------------------------------------------------------------------------

def test_hostile_r1_body_parse_boundaries_on_the_documented_writer(tmp_path):
    dup = "qa-ms001-hostile-dupkeys"
    nl = "qa-ms001-hostile-trailingnl"
    dump = tmp_path / "dump"
    proxy = Proxy(dump, tmp_path / "logs")  # documented default: the writer under audit
    try:
        # Non-UTF-8 and empty bodies: typed client errors, and no tree created for input
        # that was never parsed.
        for label, raw in (("non-UTF-8", b'\xff\xfe{"model": "x"}'), ("empty", b"")):
            status, body = proxy.post(raw)
            assert 400 <= status < 500, f"{label} body must be a client error, got {status}"
            assert b"Traceback" not in body, f"{label} body leaked a traceback to the client"
        assert listing(str(dump)) == [], (
            f"a session dir was created for unparsable bodies: {listing(str(dump))!r}"
        )

        # Duplicate keys are valid JSON: stdlib last-wins, and the record must not carry
        # duplicate corruption.
        raw = (
            b'{"model": "m-first", "model": "sloppus", "max_tokens": 16,'
            b' "system": ' + json.dumps(dup).encode()
            + b', "messages": [{"role": "user", "content": "dupkeys sentinel-dq"}]}'
        )
        assert_upstream_response(*proxy.post(raw))

        # A trailing newline is not a second request.
        raw = json.dumps(turn_body(nl, "trailingnl sentinel-tn")).encode() + b"\n\n"
        assert_upstream_response(*proxy.post(raw))
    finally:
        rc = proxy.stop()
    assert rc == 0

    dup_dir = proxy.session_dir(dup)
    body = json.loads(read_text(os.path.join(dup_dir, "turn_000.json")))
    assert body["model"] == "sloppus", "the record must carry the parsed body (last key wins)"
    assert "m-first" not in read_text(os.path.join(dup_dir, "turn_000.json")), (
        "duplicate-key corruption leaked into the documented artifact"
    )
    assert len(index_lines(proxy.session_dir(nl))) == 1, "a trailing newline duplicated the turn"


# ---------------------------------------------------------------------------
# Hostile tier R2 — the documented writer into pre-existing trees (sibling audit)
# ---------------------------------------------------------------------------

def test_hostile_r2_documented_writer_does_not_escape_the_dump_root(tmp_path):
    dangling = "qa-ms001-r2-dangling"
    rootlink = "qa-ms001-r2-rootlink"
    escaped = "qa-ms001-r2-indexlink"
    healthy = "qa-ms001-r2-sentinel"
    dump = tmp_path / "dump"
    os.makedirs(str(dump))
    outside = tmp_path / "outside"
    outside.mkdir()
    dangling_target = outside / "dangling-target"
    linked_dir = outside / "linked-dir"
    linked_dir.mkdir()
    canary = outside / "canary.txt"

    os.symlink(str(dangling_target), os.path.join(str(dump), session_id(dangling)))
    os.symlink(str(linked_dir), os.path.join(str(dump), session_id(rootlink)))
    escaped_dir = os.path.join(str(dump), session_id(escaped))
    os.makedirs(escaped_dir)
    os.symlink(str(canary), os.path.join(escaped_dir, "index.txt"))

    proxy = Proxy(dump, tmp_path / "logs")  # documented default
    try:
        # (a) a dangling symlink standing where the session directory belongs
        assert_upstream_response(*proxy.post(turn_body(dangling, "r2 dangling sentinel-dl")))
        assert not dangling_target.exists(), "no write may land at the dangling target"
        assert os.path.islink(os.path.join(str(dump), session_id(dangling))), (
            "the writer replaced the symlink to escape the fault"
        )
        # (b) a live symlink standing where the session directory belongs, pointing at a
        # real directory outside the dump root
        assert_upstream_response(*proxy.post(turn_body(rootlink, "r2 rootlink sentinel-rl")))
        assert listing(str(linked_dir)) == [], (
            f"the writer wrote through the session-dir symlink: {listing(str(linked_dir))!r}"
        )
        assert os.path.islink(os.path.join(str(dump), session_id(rootlink))), (
            "the writer replaced the session-dir symlink to route around the fault"
        )
        # (c) a symlinked regular target inside a pre-existing session directory
        assert_upstream_response(*proxy.post(turn_body(escaped, "r2 indexlink sentinel-il")))
        assert not canary.exists(), "the documented writer wrote through a symlinked target"
        assert os.path.islink(os.path.join(escaped_dir, "index.txt"))
        # (d) the process still serves, and a healthy session still captures
        assert_upstream_response(*proxy.post(turn_body(healthy, "r2 sentinel-hp")))
    finally:
        rc = proxy.stop()
    assert rc == 0, f"a faulted dump write must not become fatal, rc={rc}"
    # R2's contract is containment, not wording: neither hostile target received a byte and
    # neither symlink was replaced by a real path to route around the fault. What the writer
    # logged for these refusals is recorded in the iteration report.
    assert listing(proxy.session_dir(healthy)) == sorted(documented_files(1)), (
        "a healthy session must still capture after the refused writes"
    )


# ---------------------------------------------------------------------------
# Identity tier R5 — the session key must not merge near-misses; (sid, format) is the identity
# ---------------------------------------------------------------------------

def test_r5_near_miss_session_ids_are_not_merged(tmp_path):
    a = "qa-ms001-identity-alpha-0001"
    b = "qa-ms001-identity-alpha-0002"  # differs in the last character only
    dump = tmp_path / "dump"
    os.makedirs(str(dump))
    foreign_log = os.path.join(str(dump), session_id(a), COLLAPSED + ".log")
    os.makedirs(os.path.dirname(foreign_log))
    with open(foreign_log, "w", encoding="utf-8") as f:
        f.write("foreign collapsed artifact — must survive untouched\n")
    before = read_text(foreign_log)

    proxy = Proxy(dump, tmp_path / "logs")  # documented: the identity axis under test
    markers = {
        a: [marker("idA", i) for i in range(2)],
        b: [marker("idB", i) for i in range(2)],
    }
    try:
        # Interleaved, so a global (rather than per-session) turn counter is visible.
        for i in range(2):
            assert_upstream_response(*proxy.post(turn_body(a, markers[a][i])))
            assert_upstream_response(*proxy.post(turn_body(b, markers[b][i])))
    finally:
        rc = proxy.stop()
    assert rc == 0

    dir_a, dir_b = proxy.session_dir(a), proxy.session_dir(b)
    assert session_id(a) != session_id(b), "the sid itself merged the near-miss prompts"
    assert listing(dir_b) == sorted(documented_files(2)), (
        f"session b's layout is not its own: {listing(dir_b)!r}"
    )
    assert listing(dir_a) == sorted(documented_files(2) | {COLLAPSED + ".log"}), (
        f"session a's layout is not its own: {listing(dir_a)!r}"
    )
    # Independent per-session counters, and the near-miss identities never merge in content.
    for sid_dir, own_prompt, foreign_prompt in ((dir_a, a, b), (dir_b, b, a)):
        assert len(index_lines(sid_dir)) == 2, f"{sid_dir}: turn count merged"
        own_body = read_text(os.path.join(sid_dir, "turn_000.json"))
        assert markers[own_prompt][0] in own_body, f"{sid_dir}: lost its own turn 0"
        for foreign in markers[foreign_prompt]:
            assert foreign not in own_body, f"{sid_dir}: absorbed {foreign_prompt}'s content"
        assert os.path.exists(os.path.join(sid_dir, "turn_001.diff")), (
            f"{sid_dir}: both sessions must own a turn_001.diff (per-session counters)"
        )
    # README §Pre-existing session directories: the documented writer neither migrates nor
    # removes files it does not own — and its identity is (sid, format), not sid.
    assert read_text(foreign_log) == before, "the documented writer touched a foreign .log"
