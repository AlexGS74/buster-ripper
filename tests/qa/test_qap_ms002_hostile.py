"""Hostile-tier MS-002 gates: the R1 parse boundary and the R2 pre-existing-tree writers."""

from __future__ import annotations

import json
import os

from tests.qa.harness import (
    assert_upstream_response,
    read_text,
    record_section,
    session_id,
    turn_body,
)
from tests.qa.harness_process import Proxy
from tests.qa.qap_ms002_support import (
    TIMER_OFF_INTERVAL,
    UNREACHABLE_BYTES,
)


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
