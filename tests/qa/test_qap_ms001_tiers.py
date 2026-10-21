"""Fault isolation and the R1/R2/R5 tiers for the MS-001 dump writer."""

from __future__ import annotations

import json
import os

from tests.qa.harness import (
    assert_upstream_response,
    listing,
    read_text,
    rglob_files,
    session_id,
    turn_body,
)
from tests.qa.harness_process import (
    Proxy,
    poll_until,
    startup_refused,
)

from tests.qa.qap_ms001_support import (
    COLLAPSED,
    documented_files,
    index_lines,
    marker,
    no_dump_config_override,
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
