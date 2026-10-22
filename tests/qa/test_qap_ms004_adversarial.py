"""Adversarial-tier MS-004 gates: the R5 trust-identity pair and the R1 flag/URL boundaries."""

from __future__ import annotations

import signal

from tests.qa.harness_process import Proxy, poll_until
from tests.qa.qap_ms004_support import (
    CERT_MARKER,
    MODEL_PROBE,
    PATH_DEADLINE,
    assert_cert_verification_failure,
    assert_clean_shutdown,
    assert_single_client,
    messages_body,
    read_report,
    start_peer,
)
from tests.qa.tls_peer import MARKER, TlsPeer


# Adversarial tier

def test_r5_trust_identity_pair_is_not_merged(tmp_path):
    """TC-003's sensitivity check: same peer, port and certificate bytes, only the
    client's trust decision differs — a broken peer or an anonymous accept fails here.
    """
    peer = TlsPeer().start()
    try:
        trusted_ok, trusted_detail = peer.trusted_probe()
        untrusted_ok, untrusted_detail = peer.untrusted_probe()
        assert trusted_ok, f"a client that trusts the cert was not served: {trusted_detail}"
        assert not untrusted_ok, (
            f"a client with the default trust store was served: {untrusted_detail}"
        )
        assert CERT_MARKER.lower() in untrusted_detail.lower(), untrusted_detail
        assert peer.count(MODEL_PROBE) == 1, peer.summary()
        # One material, two trust postures, two outcomes: that difference is the
        # merge this tier forbids. The peer-side record of the *refused*
        # handshake is deliberately not asserted here — it depends on how the
        # client tears the connection down (a stdlib client that sends
        # close_notify on abort leaves no accept error at all), so it is
        # witnessed for the proxy's own client in TC-003 instead.
        outcomes = {p["label"]: p["answered"] for p in peer.control_probes}
        assert outcomes == {"trusted": True, "untrusted": False}, (
            f"{outcomes} — {peer.summary()}"
        )
    finally:
        peer.stop()


def test_r1_duplicate_flag_occurrence_honored_once(tmp_path):
    """The new flag's own parse boundary: a repeated ``is_flag=True`` occurrence
    must not error out, invert, or spawn a second posture."""
    peer = start_peer()
    report_path = str(tmp_path / "probe.json")
    proxy = Proxy(
        tmp_path / "dump", tmp_path / "logs",
        extra_args=["--skip-tls-verify", "--skip-tls-verify"],
        upstream=peer.url(), require_live_upstream=False, probe_report=report_path,
    )
    try:
        status, body = proxy.post(messages_body("qap-ms004-r1-dupflag", "r1 t0"))
        assert status == 200 and MARKER in body, (
            f"repeating the opt-out changed the outcome: {status} {body[:300]!r} "
            f"stderr={proxy.stderr()[-800:]!r}"
        )
        assert proxy.proc.poll() is None
    finally:
        rc = proxy.stop(signal.SIGTERM)
        peer.stop()
    report = read_report(proxy, report_path)
    connector = assert_single_client(report)
    assert connector["ssl_kwarg"] == "False", connector
    assert_clean_shutdown(proxy, rc)


def _run_url_shape_arm(tmp_path, tag, extra_args):
    """One URL-shape arm: peer, proxy on the hostile URL, the arm's witness, the report."""
    hostile_suffix = "/qa-base/../"
    peer = start_peer()
    report_path = str(tmp_path / f"probe-{tag}.json")
    proxy = Proxy(
        tmp_path / f"dump-{tag}", tmp_path / f"logs-{tag}",
        extra_args=extra_args,
        upstream=peer.url() + hostile_suffix,
        require_live_upstream=False, probe_report=report_path,
    )
    try:
        if extra_args:
            assert poll_until(lambda: bool(peer.paths()), deadline=PATH_DEADLINE), (
                f"the opt-out stopped working for a URL with a path prefix: "
                f"{peer.summary()} stderr={proxy.stderr()[-800:]!r}"
            )
            assert not peer.handshake_aborts, peer.summary()
        else:
            status, body = proxy.post(messages_body("qap-ms004-r1-url-verify", "r1u t0"))
            assert status != 200 and MARKER not in body, (
                f"W1 through a URL shape: {status} {body[:300]!r}"
            )
            assert_cert_verification_failure(status, body, proxy.stderr())
            assert poll_until(lambda: bool(peer.handshake_aborts), deadline=PATH_DEADLINE), (
                peer.summary()
            )
            assert peer.paths() == set(), peer.summary()
    finally:
        rc = proxy.stop(signal.SIGTERM)
        peer.stop()
    report = read_report(proxy, report_path)
    connector = assert_single_client(report)
    assert_clean_shutdown(proxy, rc)
    return report, connector


def test_r1_upstream_url_shape_does_not_change_posture(tmp_path):
    """A trailing slash and a dot segment in ``--upstream`` must not become a way
    past the posture — in either direction."""
    # verifying posture: still refused, whatever the URL shape
    _, connector_a = _run_url_shape_arm(tmp_path, "verify", [])
    assert connector_a["ssl_kwarg"] == "True"

    # opted-out posture: still accepted, whatever the URL shape. Which path the
    # dot segment normalizes to is not this gate's subject — completing the
    # handshake under the opt-out is.
    _, connector_b = _run_url_shape_arm(tmp_path, "optout", ["--skip-tls-verify"])
    assert connector_b["ssl_kwarg"] == "False"
