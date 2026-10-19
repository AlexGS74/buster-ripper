"""The loopback self-signed HTTPS peer the ``MS-004`` gate owns.

``.agents/skills/qa-project/SKILL.md`` §Upstream carries a stated amendment
scoped to this one gate: the upstream-TLS gate does not use the live remote
upstream — it starts its own loopback self-signed HTTPS peer, because neither
posture of the upstream-TLS opt-out can be observed against a plain-HTTP host.
No other gate may import this peer, and this gate must not require the remote
upstream.

Certificate material is committed with the gate (``FK-009`` option 1):
``tls_peer_cert.pem`` / ``tls_peer_key.pem``, ``CN=127.0.0.1`` with
``X509v3 SubjectAlternativeName = IP:127.0.0.1`` and no DNS name — so a
hostname-identity near-miss (connect as ``localhost``) is NOT available from
this material: a ``localhost`` refusal on a host is not attributable to a
trust decision (055 §Revisions item 1 — nothing here listens on ``::1``).
The near-miss this gate actually uses is the trust-anchor pair
(:meth:`TlsPeer.trusted_probe` / :meth:`TlsPeer.untrusted_probe`) over
identical bytes; a DNS-name variant would need a second certificate.

The peer is the observation channel for every assertion this gate makes about
outbound calls: it records each *completed* TLS request with its method and
path, and separately records handshakes the client aborted (which is what a
verifying client does with an untrusted certificate). Absence of a completed
request is only evidence because the aborted-handshake record proves the
attempt happened, and because the trusted-anchor control probe
(:meth:`TlsPeer.trusted_probe`) proves the peer serves this certificate
successfully to a client that trusts it.
"""

from __future__ import annotations

import json
import os
import socket
import ssl
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

_HERE = os.path.dirname(os.path.abspath(__file__))
CERT_PEM = os.path.join(_HERE, "tls_peer_cert.pem")
KEY_PEM = os.path.join(_HERE, "tls_peer_key.pem")

#: Body text only this peer can produce — proof that a response came from here.
MARKER = b"qa-ms004-peer-answered"

#: Exception/text fragments that mean "the client aborted the TLS handshake".
_TLS_ABORT_SIGNS = ("SSL", "sslv3", "TLSV1_ALERT", "WRONG_VERSION", "EOF", "RESET", "SHUTDOWN")


def _is_tls_abort(exc: BaseException) -> bool:
    text = f"{type(exc).__name__} {exc}"
    return any(sign.lower() in text.lower() for sign in _TLS_ABORT_SIGNS)


class _PeerServer(ThreadingHTTPServer):
    """Loopback HTTPS server with per-connection accept accounting."""

    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, host: str, port: int, context: ssl.SSLContext, peer: "TlsPeer"):
        self.peer = peer
        super().__init__((host, port), _PeerHandler)
        self.socket = context.wrap_socket(self.socket, server_side=True)

    def get_request(self):
        # SSLSocket.accept() performs the server-side handshake, so a client
        # that rejects the certificate aborts here rather than at request time.
        try:
            return super().get_request()
        except Exception as exc:  # noqa: BLE001 - classify, never swallow
            self.peer._record_accept_error(exc)
            raise

    def handle_error(self, request, client_address):
        # A client that closes a keep-alive connection mid-read is normal here;
        # record it instead of letting socketserver print tracebacks into the
        # pytest stderr the gates read as evidence.
        self.peer._record_request_error(sys.exc_info()[1])


class _PeerHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server: _PeerServer

    # -- plumbing --------------------------------------------------------
    def log_message(self, fmt, *args):  # silence stderr noise; paths are recorded instead
        return

    def _drain_body(self) -> None:
        length = int(self.headers.get("Content-Length") or 0)
        if length:
            self.rfile.read(length)

    def _send(self, status: int, payload: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def _serve(self, method: str) -> None:
        path = self.path.split("?", 1)[0]
        if method == "POST":
            self._drain_body()
        self.server.peer._record(method, path)
        if path == "/v1/models":
            body = json.dumps(
                {"object": "list", "data": [{"id": "sloppus", "object": "model"}]}
            ).encode("utf-8")
            self._send(200, body, "application/json")
        elif path == "/metrics":
            body = (
                b"# HELP vllm:gpu_cache_usage_perc GPU KV-cache usage.\n"
                b"# TYPE vllm:gpu_cache_usage_perc gauge\n"
                b'vllm:gpu_cache_usage_perc{model_name="qa"} 0.42\n'
                b"{marker}\n".replace(b"{marker}", MARKER)
            )
            self._send(200, body, "text/plain; version=0.0.4")
        elif path in ("/v1/messages", "/v1/chat/completions", "/v1/messages/count_tokens"):
            body = json.dumps(
                {
                    "id": "qa-peer-msg",
                    "type": "message",
                    "role": "assistant",
                    "model": "sloppus",
                    "content": [{"type": "text", "text": MARKER.decode("ascii")}],
                    "stop_reason": "end_turn",
                    "usage": {"input_tokens": 1, "output_tokens": 1},
                }
            ).encode("utf-8")
            self._send(200, body, "application/json")
        else:
            self._send(200, b'{"ok": true, "object": "list", "data": []}', "application/json")

    def do_GET(self):
        self._serve("GET")

    def do_POST(self):
        self._serve("POST")

    def do_OPTIONS(self):
        self._serve("OPTIONS")


class TlsPeer:
    """A running loopback self-signed HTTPS peer.

    The certificate covers ``IP:127.0.0.1`` and no DNS name. The peer is bound
    to that one address, and the identity controls this gate uses are the two
    client trust postures over the *same* material (:meth:`trusted_probe` /
    :meth:`untrusted_probe`), not a second hostname — so nothing here depends
    on how the host resolves ``localhost``.
    """

    def __init__(self, cert_path: str = CERT_PEM, key_path: str = KEY_PEM):
        self.cert_path = cert_path
        self.key_path = key_path
        self.host = "127.0.0.1"
        self.port: int | None = None
        self._server: _PeerServer | None = None
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._t0 = 0.0
        self.requests: list[dict] = []          # completed TLS requests
        self.handshake_aborts: list[dict] = []  # client aborted the handshake
        self.accept_errors: list[dict] = []     # every accept failure, classified
        self.request_errors: list[dict] = []    # post-handshake connection errors
        self.control_probes: list[dict] = []    # the gate's own trust controls

    def reset(self) -> None:
        """Forget everything the peer recorded so far.

        The gate's own health control is not the subject's traffic: a scenario
        that claims "the peer served nothing to the proxy" must not be counting
        the probe that proved the peer was alive. Control probes are kept
        separately (:attr:`control_probes`), so clearing the subject log never
        hides the fact that the peer was verified serving.
        """
        with self._lock:
            self.requests.clear()
            self.handshake_aborts.clear()
            self.accept_errors.clear()
            self.request_errors.clear()
            self._t0 = time.monotonic()

    # -- lifecycle -------------------------------------------------------
    def start(self) -> "TlsPeer":
        context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        context.load_cert_chain(certfile=self.cert_path, keyfile=self.key_path)
        with socket.socket() as probe:
            probe.bind((self.host, 0))
            self.port = probe.getsockname()[1]
        self._server = _PeerServer(self.host, self.port, context, self)
        self._t0 = time.monotonic()
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
            self._server = None
        if self._thread is not None:
            self._thread.join(timeout=5)
            self._thread = None

    def __enter__(self) -> "TlsPeer":
        return self.start()

    def __exit__(self, *exc_info):
        self.stop()
        return False

    # -- recording -------------------------------------------------------
    def _record(self, method: str, path: str) -> None:
        with self._lock:
            self.requests.append(
                {"method": method, "path": path, "t": time.monotonic() - self._t0}
            )

    def _record_accept_error(self, exc: BaseException) -> None:
        record = self._error_record(exc)
        record["tls_abort"] = _is_tls_abort(exc)
        with self._lock:
            self.accept_errors.append(record)
            if record["tls_abort"]:
                self.handshake_aborts.append(dict(record))

    def _record_request_error(self, exc: BaseException | None) -> None:
        """Post-handshake connection errors (a client closing keep-alive, etc.)."""
        if exc is None:
            return
        with self._lock:
            self.request_errors.append(self._error_record(exc))

    def _error_record(self, exc: BaseException) -> dict:
        return {
            "class": type(exc).__name__,
            "str": str(exc)[:400],
            "errno": getattr(exc, "errno", None),
            "t": time.monotonic() - self._t0,
        }

    # -- observation -----------------------------------------------------
    def paths(self) -> set[str]:
        with self._lock:
            return {r["path"] for r in self.requests}

    def count(self, path: str) -> int:
        with self._lock:
            return sum(1 for r in self.requests if r["path"] == path)

    def count_suffix(self, suffix: str) -> int:
        """Same count, tolerant of a URL prefix in ``--upstream`` (hostile tier)."""
        with self._lock:
            return sum(1 for r in self.requests if r["path"].endswith(suffix))

    def summary(self) -> str:
        with self._lock:
            return json.dumps(
                {
                    "requests": self.requests,
                    "handshake_aborts": self.handshake_aborts,
                    "accept_errors": self.accept_errors,
                    "request_errors": self.request_errors,
                    "control_probes": self.control_probes,
                },
                indent=2,
            )

    def url(self, host: str | None = None) -> str:
        assert self.port is not None, "peer not started"
        return f"https://{host or self.host}:{self.port}"

    # -- controls --------------------------------------------------------
    def trusted_probe(self, timeout: float = 10.0) -> tuple[bool, str]:
        """Handshake and request as a client that trusts THIS certificate.

        Positive control for the peer itself: the material is well-formed,
        unexpired and hostname-matching, so a verifying client accepts it.
        Without this, "the proxy refused the peer" could be a broken peer
        rather than a posture the proxy held.
        """
        return self._client_probe(
            ssl.create_default_context(cafile=self.cert_path), "trusted", timeout=timeout
        )

    def untrusted_probe(self, timeout: float = 10.0) -> tuple[bool, str]:
        """Handshake as a client with the default trust store.

        The same peer, the same port, the same bytes on the wire — the only
        variable is whether the client trusts it. This is the identity
        near-miss pair the adversarial tier needs: a trust decision that
        merged the two postures would report the same outcome for both.
        """
        return self._client_probe(ssl.create_default_context(), "untrusted", timeout=timeout)

    def _client_probe(
        self, context: ssl.SSLContext, label: str, timeout: float
    ) -> tuple[bool, str]:
        assert self.port is not None, "peer not started"
        target = self.host
        request = (
            f"GET /v1/models HTTP/1.1\r\nHost: {target}\r\nConnection: close\r\n\r\n"
        ).encode("ascii")
        detail: str
        try:
            with socket.create_connection((target, self.port), timeout=timeout) as raw:
                with context.wrap_socket(raw, server_hostname=target) as tls:
                    body = _read_http(tls, request)
        except Exception as exc:  # noqa: BLE001 - reported, not swallowed
            detail = f"{type(exc).__name__}: {exc}"
            answered = False
        else:
            answered = body.startswith(b"HTTP/1.1 200") and (
                MARKER in body or b"sloppus" in body
            )
            detail = body[:300].decode("utf-8", "replace")
        with self._lock:
            self.control_probes.append(
                {
                    "label": label,
                    "answered": answered,
                    "detail": detail[:300],
                    "t": time.monotonic() - self._t0,
                }
            )
        return (answered, detail)


def _read_http(sock, request: bytes, limit: int = 4096) -> bytes:
    sock.sendall(request)
    chunks = []
    total = 0
    while total < limit:
        try:
            chunk = sock.recv(1024)
        except OSError:
            break
        if not chunk:
            break
        chunks.append(chunk)
        total += len(chunk)
    return b"".join(chunks)


def openssl_version() -> str:
    try:
        return subprocess.run(
            ["openssl", "version"], capture_output=True, text=True, timeout=5
        ).stdout.strip()
    except Exception:  # noqa: BLE001 - diagnostics only
        return "unknown"
