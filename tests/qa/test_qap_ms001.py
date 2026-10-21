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

The four re-entry repairs (077/096/099/108) and the pre-repair failure shapes they
closed are recorded in
``.sessions/codify/02.00-2026-09-13-message-dump-bulk-flush--9bf7afff5573/chronicle.md``.

**Contract directive honored** (``qa-plan.md`` §Phase Requirements, 022): no gate here
passes a flush parameter; TC-100 bounds its pre-flush window by observation, not a sleep.

Run explicitly (NOT part of the hermetic default suite)::

    .venv/bin/python -m pytest tests/qa/test_qap_ms001.py tests/qa/test_qap_ms001_witnesses.py tests/qa/test_qap_ms001_tiers.py -o addopts=

The three modules together are the QAP-MS001 entry point; the gate module alone
collects only the three tests that remain in it.

The declared live upstream is required; gates fail closed on the harness preflight and assert
on dump artifacts and process behavior, never on model output.
"""

from __future__ import annotations

import os
import time

from tests.qa.harness import (
    assert_upstream_response,
    listing,
    log_artifacts,
    read_text,
    rglob_files,
    session_id,
    turn_body,
)
from tests.qa.harness_process import (
    Proxy,
)

from tests.qa.qap_ms001_support import (
    COLLAPSED,
    assert_documented_contents,
    count_report,
    documented_files,
    marker,
    no_dump_config_override,
    nonce,
    ops_by_name,
    rendering_section,
)

N_TURNS = 3


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
