"""QAP-MS004 — MS-004 integration gates: the upstream-TLS verification opt-out.

Covers (qa-plan.md §QA Gate Matrix / §Scenario Contracts, the
``TC-003 / TC-113 / TC-114`` group):

- TC-003  VP-T006-verify-on-by-default   — the untrusted peer is refused, and the
  refusal is a certificate-verification error. Forbidden witness **W1**: a
  self-signed peer accepted with no opt-out.
- TC-113  VP-T006-optout-honored         — the same peer, the same request, with
  ``--skip-tls-verify``, completes.
- TC-114  VP-T006-posture-is-process-wide — one connector, one posture, for the
  proxied request **and** the ``/v1/models`` probe and the ``/metrics`` poll.
  Forbidden witness **W2**: the flag accepted, the global set, but both postures
  behaving identically.
- Adversarial tier (codify-adversarial-surface-discipline): the trust-identity
  near-miss pair (R5 — same material, two client trust postures), a duplicate
  occurrence of the new flag and an upstream-URL shape variant (R1 — the parse
  boundaries this milestone adds). ``R2`` (writer into a pre-existing tree) and
  ``R4`` (payload decoder census) have no surface in this milestone; the reason
  is recorded in the iteration report, not silently skipped.

Harness notes:

- This is the ONE gate the ``qa-project`` §Upstream amendment exempts from the
  live remote upstream: it starts its own loopback self-signed HTTPS peer
  (:mod:`tests.qa.tls_peer`, committed certificate material) and passes
  ``require_live_upstream=False``, so ``hyperipper.local`` reachability is not a
  precondition here. The gate additionally asserts that every outbound call the
  proxy made went to the peer — the amendment's "the remote upstream is not used
  by this gate" clause, checked rather than assumed.
- Synchronization is the declared contract only: readiness by port poll (inside
  ``Proxy``), settling by bounded polling of an application-side observable (the
  peer's request log), quiescence by waiting for process exit before reading the
  instrumented report. No fixed sleeps.
- No telemetry is asserted: the project declares no telemetry-verification
  capability and ``FK-008`` option 1 ships no runtime posture record.

Run explicitly (NOT part of the hermetic default suite)::

    .venv/bin/python -m pytest tests/qa/test_qap_ms004.py -o addopts=
"""

from __future__ import annotations

import json
import os
import signal
from urllib.parse import urlparse

from tests.qa.harness import turn_body
from tests.qa.harness_process import Proxy, poll_until
from tests.qa.tls_peer import MARKER, TlsPeer

#: The peer is probed at process start; this bound is generous, and the paired
#: no-opt-out run in TC-114 uses the identical bound so its absence claim is
#: witnessed by the run that does observe them (non-vacuous by construction).
PATH_DEADLINE = 20.0

CERT_MARKER = "CERTIFICATE_VERIFY_FAILED"
CERT_REASON = "certificate verify failed"

#: The three outbound call sites the plan names for VP-T006-posture-is-process-wide.
MODEL_PROBE = "/v1/models"
METRICS_POLL = "/metrics"
PROXIED = "/v1/messages"
OUTBOUND_PATHS = {MODEL_PROBE, METRICS_POLL, PROXIED}


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# TC-003 — VP-T006-verify-on-by-default
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# TC-113 — VP-T006-optout-honored
# ---------------------------------------------------------------------------

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
        assert poll_until(lambda: {MODEL_PROBE, METRICS_POLL} <= peer.paths(),
                          deadline=PATH_DEADLINE), (
            f"the opt-out did not carry the internal call sites: {peer.summary()}"
        )
        status, body = proxy.post(messages_body("qap-ms004-tc113", "tc113 t0"))
        assert status == 200, (
            f"the opt-out did not reach the connector: status={status} "
            f"body={body[:300]!r} stderr={proxy.stderr()[-800:]!r}"
        )
        assert MARKER in body, f"the answer did not come from the gate's peer: {body[:300]!r}"
        assert peer.count_suffix(PROXIED) >= 1, peer.summary()
        assert not peer.handshake_aborts, (
            f"a verifying handshake was attempted despite the opt-out: {peer.summary()}"
        )
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


# ---------------------------------------------------------------------------
# TC-114 — VP-T006-posture-is-process-wide (paired runs, one peer shape)
# ---------------------------------------------------------------------------

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


def test_tc114_one_posture_governs_every_outbound_call(tmp_path):
    """One opt-out, one connector: the ``/metrics`` poll and the ``/v1/models``
    probe carry the same posture as the proxied request.

    Run A (opt-out) is the positive control that proves all three call sites are
    observable at the peer inside ``PATH_DEADLINE``; run B (default) uses the
    identical bound, so "the peer saw nothing on any path" is a refusal by every
    call site and not a too-short wait.
    """
    # --- run A: opt-out, every call site completes -------------------------
    peer_a = start_peer()
    report_a_path = str(tmp_path / "probe-a.json")
    proxy_a = Proxy(
        tmp_path / "dump-a", tmp_path / "logs-a",
        extra_args=["--skip-tls-verify"],
        upstream=peer_a.url(), require_live_upstream=False, probe_report=report_a_path,
    )
    try:
        a = _drive_all_call_sites(proxy_a, peer_a, "qap-ms004-tc114-optout")
        assert a["internal_paths_at_peer"], peer_a.summary()
        assert a["status"] == 200 and MARKER in a["body"], (
            f"proxied call site did not complete: {a['status']} {a['body'][:300]!r}"
        )
        missing_a = {p for p in OUTBOUND_PATHS if peer_a.count_suffix(p) == 0}
        assert not missing_a, (
            f"W2-shaped gap: the opt-out did not govern every call site; missing "
            f"{sorted(missing_a)} of {sorted(OUTBOUND_PATHS)} "
            f"(peer saw {sorted(peer_a.paths())})"
        )
    finally:
        rc_a = proxy_a.stop(signal.SIGTERM)
        peer_a.stop()
    report_a = read_report(proxy_a, report_a_path)
    connector_a = assert_single_client(report_a)
    assert connector_a["ssl_kwarg"] == "False", connector_a
    assert OUTBOUND_PATHS <= call_paths(report_a), (
        f"instrumented call sites incomplete: {sorted(call_paths(report_a))}"
    )
    assert_all_calls_through(report_a, connector_a, urlparse(peer_a.url()).netloc)
    assert_clean_shutdown(proxy_a, rc_a)

    # --- run B: no opt-out, no call site completes -------------------------
    peer_b = start_peer()
    report_b_path = str(tmp_path / "probe-b.json")
    proxy_b = Proxy(
        tmp_path / "dump-b", tmp_path / "logs-b",
        extra_args=[],
        upstream=peer_b.url(), require_live_upstream=False, probe_report=report_b_path,
    )
    try:
        b = _drive_all_call_sites(proxy_b, peer_b, "qap-ms004-tc114-default")
        assert b["status"] != 200 and MARKER not in b["body"], (
            f"W1: a call site reached the untrusted peer: {b['status']} {b['body'][:300]!r}"
        )
        assert_cert_verification_failure(b["status"], b["body"], proxy_b.stderr())
        assert poll_until(lambda: bool(peer_b.handshake_aborts), deadline=PATH_DEADLINE), (
            peer_b.summary()
        )
        assert peer_b.paths() == set(), (
            f"W2: some call site skipped verification while another refused; the peer "
            f"served {sorted(peer_b.paths())} with no opt-out — {peer_b.summary()}"
        )
    finally:
        rc_b = proxy_b.stop(signal.SIGTERM)
        peer_b.stop()
    report_b = read_report(proxy_b, report_b_path)
    connector_b = assert_single_client(report_b)
    assert connector_b["ssl_kwarg"] == "True", connector_b
    assert OUTBOUND_PATHS <= call_paths(report_b), (
        f"the default posture was not exercised at every call site, so the absence "
        f"claim above is unproven: {sorted(call_paths(report_b))}"
    )
    assert_all_calls_through(report_b, connector_b, urlparse(peer_b.url()).netloc)
    assert_clean_shutdown(proxy_b, rc_b)

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


# ---------------------------------------------------------------------------
# Adversarial tier
# ---------------------------------------------------------------------------

def test_r5_trust_identity_pair_is_not_merged(tmp_path):
    """Same peer, same port, same certificate bytes: the only variable is the
    client's trust decision, and the two outcomes must be observably different.

    This is the sensitivity check for TC-003 — it fails if a "refusal" is really
    a broken peer (both postures fail) and if a "success" is really an anonymous
    accept (both postures pass).
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


def test_r1_upstream_url_shape_does_not_change_posture(tmp_path):
    """A trailing slash and a dot segment in ``--upstream`` must not become a way
    past the posture — in either direction."""
    hostile_suffix = "/qa-base/../"
    # verifying posture: still refused, whatever the URL shape
    peer_a = start_peer()
    report_a_path = str(tmp_path / "probe-a.json")
    proxy_a = Proxy(
        tmp_path / "dump-a", tmp_path / "logs-a",
        extra_args=[],
        upstream=peer_a.url() + hostile_suffix,
        require_live_upstream=False, probe_report=report_a_path,
    )
    try:
        status, body = proxy_a.post(messages_body("qap-ms004-r1-url-verify", "r1u t0"))
        assert status != 200 and MARKER not in body, (
            f"W1 through a URL shape: {status} {body[:300]!r}"
        )
        assert_cert_verification_failure(status, body, proxy_a.stderr())
        assert poll_until(lambda: bool(peer_a.handshake_aborts), deadline=PATH_DEADLINE), (
            peer_a.summary()
        )
        assert peer_a.paths() == set(), peer_a.summary()
    finally:
        rc_a = proxy_a.stop(signal.SIGTERM)
        peer_a.stop()
    report_a = read_report(proxy_a, report_a_path)
    assert assert_single_client(report_a)["ssl_kwarg"] == "True"
    assert_clean_shutdown(proxy_a, rc_a)

    # opted-out posture: still accepted, whatever the URL shape. Which path the
    # dot segment normalizes to is not this gate's subject — completing the
    # handshake under the opt-out is.
    peer_b = start_peer()
    report_b_path = str(tmp_path / "probe-b.json")
    proxy_b = Proxy(
        tmp_path / "dump-b", tmp_path / "logs-b",
        extra_args=["--skip-tls-verify"],
        upstream=peer_b.url() + hostile_suffix,
        require_live_upstream=False, probe_report=report_b_path,
    )
    try:
        assert poll_until(lambda: bool(peer_b.paths()), deadline=PATH_DEADLINE), (
            f"the opt-out stopped working for a URL with a path prefix: "
            f"{peer_b.summary()} stderr={proxy_b.stderr()[-800:]!r}"
        )
        assert not peer_b.handshake_aborts, peer_b.summary()
    finally:
        rc_b = proxy_b.stop(signal.SIGTERM)
        peer_b.stop()
    report_b = read_report(proxy_b, report_b_path)
    assert assert_single_client(report_b)["ssl_kwarg"] == "False"
    assert_clean_shutdown(proxy_b, rc_b)
