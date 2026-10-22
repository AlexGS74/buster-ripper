"""QAP-MS004 — MS-004 integration gates: the upstream-TLS verification opt-out.

Covers (qa-plan.md §QA Gate Matrix + §Scenario Contracts): TC-003
VP-T006-verify-on-by-default and W1 (the untrusted peer refused with a
certificate-verification error), TC-113 VP-T006-optout-honored (the same peer and
request with ``--skip-tls-verify``), TC-114 VP-T006-posture-is-process-wide (one
connector, one posture, for the proxied request, the ``/v1/models`` probe and the
``/metrics`` poll) and W2 (both postures behaving identically); the adversarial
tier adds the R5 trust-identity near-miss pair and the R1 flag-repeat and
upstream-URL-shape boundaries and lives in
`tests/qa/test_qap_ms004_adversarial.py`. R2 and R4 have no surface in this
milestone.

Harness (``qa-project`` §Upstream + §Capability Declarations): this gate starts its
own loopback self-signed HTTPS peer with ``require_live_upstream=False`` and asserts
every outbound call went to it; readiness is a port poll inside ``Proxy``, settling
is bounded polling of the peer's request log and quiescence is process exit — no
fixed sleeps, no telemetry.

Run explicitly (not in the hermetic default suite):
``.venv/bin/python -m pytest tests/qa/test_qap_ms004.py tests/qa/test_qap_ms004_adversarial.py -o addopts=``.
"""

from __future__ import annotations

import json
import signal
from urllib.parse import urlparse

from tests.qa.harness_process import Proxy, poll_until
from tests.qa.qap_ms004_support import (
    METRICS_POLL,
    MODEL_PROBE,
    OUTBOUND_PATHS,
    PATH_DEADLINE,
    PROXIED,
    assert_all_calls_through,
    assert_cert_verification_failure,
    assert_clean_shutdown,
    assert_single_client,
    call_paths,
    messages_body,
    read_report,
    start_peer,
)
from tests.qa.tls_peer import MARKER, TlsPeer


# TC-003 — VP-T006-verify-on-by-default

def _assert_refused_before_the_peer(proxy, peer):
    """No opt-out: the request fails as a certificate error and the peer serves nothing."""
    status, body = proxy.post(messages_body("qap-ms004-tc003", "tc003 t0"))
    # Visible result: the request does not complete against that upstream.
    assert status != 200, f"W1: untrusted peer returned 200: {body[:300]!r}"
    assert status >= 400, f"expected a server-side failure, got {status}: {body[:300]!r}"
    assert MARKER not in body, f"W1: the caller received the peer's answer: {body[:300]!r}"
    # The failure's error class, not merely "some 5xx".
    assert_cert_verification_failure(status, body, proxy.stderr())
    # Independent witness: the attempt happened and the client aborted the
    # handshake after evaluating the certificate.
    assert poll_until(lambda: bool(peer.handshake_aborts), deadline=PATH_DEADLINE), (
        f"no aborted TLS attempt was recorded — the peer was never reached, so the "
        f"refusal below is not evidence of a posture: {peer.summary()}"
    )
    assert peer.paths() == set(), (
        f"W1: the peer served a request with no opt-out: {peer.summary()}"
    )
    return status, body


def test_tc003_untrusted_peer_refused_by_default(tmp_path):
    """No opt-out: the untrusted-certificate HTTPS upstream is not reached, and
    the failure's class is certificate verification."""
    peer = start_peer()
    report_path = str(tmp_path / "probe.json")
    proxy = Proxy(
        tmp_path / "dump", tmp_path / "logs",
        extra_args=["--verbose"],
        upstream=peer.url(), require_live_upstream=False, probe_report=report_path,
    )
    try:
        status, _ = _assert_refused_before_the_peer(proxy, peer)
        # Lifecycle: the refusal leaves the proxy serving, identically.
        assert proxy.proc.poll() is None, "a refused upstream took the process down"
        status2, body2 = proxy.post(messages_body("qap-ms004-tc003", "tc003 t1 — still serving"))
        assert status2 == status and MARKER not in body2, (
            f"the first refusal changed how the proxy answers the same request: "
            f"{status2} {body2[:300]!r}"
        )
    finally:
        rc = proxy.stop(signal.SIGTERM)
        peer.stop()
    report = read_report(proxy, report_path)
    connector = assert_single_client(report)
    assert connector["ssl_kwarg"] == "True", (
        f"the one connector was not built in the verifying posture: {connector}"
    )
    assert_all_calls_through(report, connector, urlparse(peer.url()).netloc)
    assert_clean_shutdown(proxy, rc)


# TC-113 — VP-T006-optout-honored

def _assert_optout_reaches_every_site(peer):
    """The opt-out carried the internal call sites, the proxied hit, and no aborts."""
    assert poll_until(lambda: {MODEL_PROBE, METRICS_POLL} <= peer.paths(),
                      deadline=PATH_DEADLINE), (
        f"the opt-out did not carry the internal call sites: {peer.summary()}"
    )
    assert peer.count_suffix(PROXIED) >= 1, peer.summary()
    assert not peer.handshake_aborts, (
        f"a verifying handshake was attempted despite the opt-out: {peer.summary()}"
    )


def test_tc113_same_peer_accepted_with_optout(tmp_path):
    """With the opt-out, the same peer and the same request complete."""
    peer = start_peer()
    report_path = str(tmp_path / "probe.json")
    proxy = Proxy(
        tmp_path / "dump", tmp_path / "logs",
        extra_args=["--skip-tls-verify"],
        upstream=peer.url(), require_live_upstream=False, probe_report=report_path,
    )
    try:
        status, body = proxy.post(messages_body("qap-ms004-tc113", "tc113 t0"))
        assert status == 200, (
            f"the opt-out did not reach the connector: status={status} "
            f"body={body[:300]!r} stderr={proxy.stderr()[-800:]!r}"
        )
        assert MARKER in body, f"the answer did not come from the gate's peer: {body[:300]!r}"
        _assert_optout_reaches_every_site(peer)
        assert proxy.proc.poll() is None
    finally:
        rc = proxy.stop(signal.SIGTERM)
        peer.stop()
    report = read_report(proxy, report_path)
    connector = assert_single_client(report)
    assert connector["ssl_kwarg"] == "False", (
        f"W2: the flag was accepted but the connector kept the verifying posture: "
        f"{connector}"
    )
    assert_all_calls_through(report, connector, urlparse(peer.url()).netloc)
    assert_clean_shutdown(proxy, rc)


# TC-114 — VP-T006-posture-is-process-wide (paired runs, one peer shape)

def _drive_all_call_sites(proxy: Proxy, peer: TlsPeer, system: str) -> dict:
    """The startup-driven probe + poll, then one proxied turn; settle on the peer."""
    internal_at_peer = poll_until(
        lambda: {MODEL_PROBE, METRICS_POLL} <= peer.paths(), deadline=PATH_DEADLINE
    )
    status, body = proxy.post(messages_body(system, "tc114 proxied turn"))
    proxied_at_peer = poll_until(
        lambda: peer.count_suffix(PROXIED) >= 1, deadline=PATH_DEADLINE
    )
    return {
        "internal_paths_at_peer": internal_at_peer,
        "proxied_at_peer": proxied_at_peer,
        "status": status,
        "body": body,
    }


def _assert_optout_everywhere(result, peer):
    """Run A's claim: with the opt-out, every call site reached the peer."""
    assert result["internal_paths_at_peer"], peer.summary()
    assert result["status"] == 200 and MARKER in result["body"], (
        f"proxied call site did not complete: {result['status']} {result['body'][:300]!r}"
    )
    missing = {p for p in OUTBOUND_PATHS if peer.count_suffix(p) == 0}
    assert not missing, (
        f"W2-shaped gap: the opt-out did not govern every call site; missing "
        f"{sorted(missing)} of {sorted(OUTBOUND_PATHS)} "
        f"(peer saw {sorted(peer.paths())})"
    )


def _assert_refused_everywhere(result, peer, proxy):
    """Run B's claim: with no opt-out, no call site reached the peer."""
    assert result["status"] != 200 and MARKER not in result["body"], (
        f"W1: a call site reached the untrusted peer: {result['status']} {result['body'][:300]!r}"
    )
    assert_cert_verification_failure(result["status"], result["body"], proxy.stderr())
    assert poll_until(lambda: bool(peer.handshake_aborts), deadline=PATH_DEADLINE), (
        peer.summary()
    )
    assert peer.paths() == set(), (
        f"W2: some call site skipped verification while another refused; the peer "
        f"served {sorted(peer.paths())} with no opt-out — {peer.summary()}"
    )


def _run_arm(tmp_path, tag, extra_args):
    """One TC-114 arm: peer, proxy, every call site, the arm's evidence, the report."""
    peer = start_peer()
    report_path = str(tmp_path / f"probe-{tag}.json")
    proxy = Proxy(
        tmp_path / f"dump-{tag}", tmp_path / f"logs-{tag}",
        extra_args=extra_args,
        upstream=peer.url(), require_live_upstream=False, probe_report=report_path,
    )
    try:
        result = _drive_all_call_sites(proxy, peer, f"qap-ms004-tc114-{tag}")
        if extra_args:
            _assert_optout_everywhere(result, peer)
        else:
            _assert_refused_everywhere(result, peer, proxy)
    finally:
        rc = proxy.stop(signal.SIGTERM)
        peer.stop()
    report = read_report(proxy, report_path)
    connector = assert_single_client(report)
    assert_all_calls_through(report, connector, urlparse(peer.url()).netloc)
    assert_clean_shutdown(proxy, rc)
    return report, connector


def test_tc114_one_posture_governs_every_outbound_call(tmp_path):
    """One opt-out, one connector: every call site carries the same posture.

    Run A proves all three are observable at the peer inside ``PATH_DEADLINE``; run
    B shares that bound, so "the peer saw nothing" is a refusal, not a short wait.
    """
    # --- run A: opt-out, every call site completes -------------------------
    report_a, connector_a = _run_arm(tmp_path, "optout", ["--skip-tls-verify"])
    assert connector_a["ssl_kwarg"] == "False", connector_a
    assert OUTBOUND_PATHS <= call_paths(report_a), (
        f"instrumented call sites incomplete: {sorted(call_paths(report_a))}"
    )

    # --- run B: no opt-out, no call site completes -------------------------
    report_b, connector_b = _run_arm(tmp_path, "default", [])
    assert connector_b["ssl_kwarg"] == "True", connector_b
    assert OUTBOUND_PATHS <= call_paths(report_b), (
        f"the default posture was not exercised at every call site, so the absence "
        f"claim above is unproven: {sorted(call_paths(report_b))}"
    )

    # --- both postures from one connector ---------------------------------
    assert connector_a["ssl_kwarg"] != connector_b["ssl_kwarg"], (
        "W2: the opt-out and the default produced the same connector posture"
    )
    assert report_a["total_connectors"] == report_b["total_connectors"] == 1
    for report, connector in ((report_a, connector_a), (report_b, connector_b)):
        assert report["distinct_connector_ids_serving_calls"] == [connector["id"]], (
            "more than one connector served the outbound calls in this run: "
            f"{json.dumps(report['calls'], indent=2)}"
        )
