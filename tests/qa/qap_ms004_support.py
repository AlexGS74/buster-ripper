"""Shared constants and assertion helpers for the MS-004 QA gate."""

from __future__ import annotations

import json
import os
from urllib.parse import urlparse

from tests.qa.harness import turn_body
from tests.qa.harness_process import Proxy
from tests.qa.tls_peer import TlsPeer

#: Generous peer-probe bound; TC-114's paired no-opt-out run shares it, so absence is non-vacuous.
PATH_DEADLINE = 20.0

CERT_MARKER = "CERTIFICATE_VERIFY_FAILED"
CERT_REASON = "certificate verify failed"

#: The three outbound call sites the plan names for VP-T006-posture-is-process-wide.
MODEL_PROBE = "/v1/models"
METRICS_POLL = "/metrics"
PROXIED = "/v1/messages"
OUTBOUND_PATHS = {MODEL_PROBE, METRICS_POLL, PROXIED}


# helpers

def start_peer() -> TlsPeer:
    peer = TlsPeer().start()
    # The peer's own health is a precondition, not a conclusion: the certificate
    # must be loadable and servable, or every refusal below would be vacuous.
    ok, detail = peer.trusted_probe()
    assert ok, f"the gate's own peer is not serving its certificate: {detail}"
    # Then forget that probe: the scenarios below claim things about what the
    # PROXY reached, and the gate's own control traffic must not be counted as
    # the subject's. The control survives in peer.control_probes.
    peer.reset()
    assert peer.paths() == set(), "the peer's own log did not clear"
    return peer


def read_report(proxy: Proxy, report_path: str) -> dict:
    assert os.path.exists(report_path), (
        "the instrumented entry wrote no report — the process did not reach its "
        f"exit path (stderr tail: {proxy.stderr()[-800:]!r})"
    )
    with open(report_path, encoding="utf-8") as f:
        return json.load(f)


def call_paths(report: dict) -> set[str]:
    return {urlparse(c["path"]).path for c in report.get("calls", [])}


def assert_single_client(report: dict) -> dict:
    """The plan's one-connector assertion, once, for every scenario that needs it."""
    assert report["total_connectors"] == 1, (
        f"expected exactly one TCPConnector for the process, got "
        f"{report['total_connectors']}: {json.dumps(report['connectors'], indent=2)}"
    )
    assert report["total_sessions"] == 1, (
        f"expected exactly one ClientSession for the process, got "
        f"{report['total_sessions']}: {json.dumps(report['sessions'], indent=2)}"
    )
    connector = report["connectors"][0]
    session = report["sessions"][0]
    assert session["connector_id"] == connector["id"], (
        "the one ClientSession is not built on the one TCPConnector"
    )
    assert session["ssl_kwarg"] == "<absent>", (
        "a per-session ssl argument is a second posture source — the plan settles "
        f"one connector, one posture: {session}"
    )
    return connector


def assert_all_calls_through(report: dict, connector: dict, peer_netloc: str) -> None:
    assert report["calls"], "the proxy made no outbound call at all"
    for call in report["calls"]:
        assert call["connector_id"] == connector["id"], (
            f"an outbound call bypassed the single connector: {call}"
        )
        assert call["session_id"] == report["sessions"][0]["id"], (
            f"an outbound call bypassed the single session: {call}"
        )
        parsed = urlparse(call["path"])
        assert parsed.scheme == "https", f"unexpected scheme on the wire: {call}"
        assert parsed.netloc == peer_netloc, (
            f"the gate left its own peer — {call['path']} is not the amendment's "
            f"loopback upstream ({peer_netloc})"
        )


def assert_cert_verification_failure(status: int, body: bytes, stderr: str) -> str:
    """The failure's *class*, not merely "some error" — the plan's evidence line."""
    combined = body.decode("utf-8", "replace") + "\n" + stderr
    low = combined.lower()
    assert CERT_MARKER.lower() in low and CERT_REASON in low, (
        f"the failure is not a certificate-verification error "
        f"(status={status}, body={body[:400]!r}, stderr tail={stderr[-1200:]!r})"
    )
    return combined


def assert_clean_shutdown(proxy: Proxy, rc: int) -> None:
    stderr = proxy.stderr()
    assert rc == 0, f"proxy exited {rc} after SIGTERM; stderr={stderr[-1200:]!r}"
    assert "Traceback (most recent call last)" not in stderr, (
        f"shutdown or the refused upstream left a traceback: {stderr[-1200:]!r}"
    )


def messages_body(system: str, text: str) -> dict:
    """One /v1/messages turn, in the harness's documented request shape."""
    return turn_body(system, text)
