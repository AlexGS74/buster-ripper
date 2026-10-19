"""Shared process-level QA harness for the buster-ripper gates.

Drives the real proxy as a subprocess over real HTTP with real signals, per
``.agents/skills/qa-project/SKILL.md``. These gates need the remote upstream
(``http://hyperipper.local:30000``), real subprocesses, and real signals; they
are selected explicitly and are NOT part of the hermetic default
``uv run pytest`` run::

    .venv/bin/python -m pytest tests/qa -o addopts=

The endpoint is the live remote model server. Corrected 2026-10-17 from
``:30001``, which is a peer Buster Ripper proxy on the same host rather than the
model server (``decisions.md`` §US-009, user's dated correction); the app's own
``--port`` default and ``README.md``'s ``:30001`` occurrences are the proxy's
listen port and are unchanged.

Synchronization follows the declared contract: readiness by polling the
listening port, quiescence by waiting for process exit or polling the
artifact/stderr — never a fixed sleep.

Exception to the upstream requirement: ``.agents/skills/qa-project/SKILL.md``
§Upstream carries a stated amendment scoped to the ``MS-004`` upstream-TLS
gate, which names its own loopback HTTPS peer instead of the live remote
upstream. ``Proxy(..., upstream=..., require_live_upstream=False)`` is that
amendment's harness hook; no other gate passes it.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PROXY_BIN = os.path.join(REPO_ROOT, ".venv", "bin", "buster-ripper")
COUNTED_ENTRY = os.path.join(REPO_ROOT, "tests", "qa", "counted_proxy_entry.py")
TLS_PROBE_ENTRY = os.path.join(REPO_ROOT, "tests", "qa", "tls_probe_entry.py")
# The gate upstream is the model server. Corrected 2026-10-17 from :30001, a peer
# Buster Ripper proxy rather than the model server (decisions.md §US-009).
UPSTREAM = "http://hyperipper.local:30000"

def require_upstream() -> None:
    """Fail closed before a gate can run against a DNS/connection-error baseline.

    Probe the declared endpoint directly, without ambient HTTP proxy settings.
    An HTTP error still establishes that the remote service answered; DNS,
    connection, and timeout failures do not.
    """
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with opener.open(f"{UPSTREAM}/v1/models", timeout=5) as response:
            if response.status >= 500:
                raise RuntimeError(f"required live upstream {UPSTREAM} returned {response.status}")
            response.read(1)
    except urllib.error.HTTPError as exc:
        if exc.code >= 500:
            raise RuntimeError(f"required live upstream {UPSTREAM} returned {exc.code}") from exc
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise RuntimeError(f"required live upstream {UPSTREAM} unavailable: {exc}") from exc


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def session_id(system_prompt: str) -> str:
    """First 12 hex chars of SHA-256(system prompt text)."""
    return hashlib.sha256(system_prompt.encode("utf-8")).hexdigest()[:12]


def wait_tcp(port: int, deadline: float = 15.0) -> bool:
    """Readiness: poll until the listening port accepts a connection."""
    end = time.monotonic() + deadline
    while time.monotonic() < end:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.5):
                return True
        except OSError:
            time.sleep(0.1)
    return False


def poll_until(predicate, deadline: float = 15.0, interval: float = 0.2) -> bool:
    """Quiescence: wait on an explicit application signal, never a fixed sleep."""
    end = time.monotonic() + deadline
    while time.monotonic() < end:
        if predicate():
            return True
        time.sleep(interval)
    return False


def billing_system(system: str, nonce: str) -> str:
    """The framework's billing-header form of a system prompt.

    ``README.md`` §Normalizations 1: Claude Code injects
    ``x-anthropic-billing-header: ... cch=<nonce>`` at position 0 of the system
    prompt on every request and buster-ripper strips it. A gate that must observe
    that stripping sends this form; a nonce-free body is identical before and
    after normalization and cannot distinguish the ordering.
    """
    return (
        "x-anthropic-billing-header: cc_version=2.1.50.f15; "
        f"cc_entrypoint=cli; cch={nonce};\n{system}"
    )


def turn_body(
    system: str, user_text: str, model: str = "sloppus", nonce: str | None = None
) -> dict:
    """One /v1/messages request body carrying a single user message.

    ``nonce`` selects the framework's billing-header form of ``system`` (the
    normalization gate); omitting it keeps every existing caller's body bytes and
    ``session_id`` derivation unchanged.
    """
    return {
        "model": model,
        "max_tokens": 16,
        "system": billing_system(system, nonce) if nonce else system,
        "messages": [{"role": "user", "content": user_text}],
    }


def record_section(artifact_text: str, header: str) -> str:
    """One record's body from a collapsed artifact, bounded by its public headers.

    ``README.md`` §Reading a collapsed artifact publishes the section headers
    (``--- turn NNN body ---``, ``--- turn NNN rendered ---``,
    ``--- turn NNN diff ---``) and the one-line-per-turn summary lines; a record
    runs from its header to the next summary line or section header.
    """
    i = artifact_text.find(header)
    assert i >= 0, f"artifact is missing section header {header!r}"
    start = i + len(header)
    m = re.search(r"(?m)^(?:--- turn |turn \d{3}\b)", artifact_text[start:])
    return artifact_text[start:start + m.start()] if m else artifact_text[start:]


def read_text(path: str, newline: str | None = None) -> str:
    """Read a dump artifact as text, tolerating a hostile byte in a captured body.

    ``newline`` is passed to ``open`` unchanged. The default (``None``) is the
    universal-newline mode the JSON, index and nonce reads want. A comparison over
    the writer's own line identity passes ``newline=""``: the documented diff is
    computed over the renderings *before* the write, so the expected side must see
    the same ``\\r\\n``/``\\r`` bytes production diffed (``S21`` CE-2).
    """
    with open(path, encoding="utf-8", errors="replace", newline=newline) as f:
        return f.read()


def rglob_files(root: str) -> list[str]:
    """Every file under ``root``, whatever the layout."""
    out = []
    for dirpath, _dirnames, filenames in os.walk(root):
        for name in filenames:
            out.append(os.path.join(dirpath, name))
    return out


def log_artifacts(dump_dir: str) -> list[str]:
    """The collapsed ``*.log`` artifacts anywhere under a dump root.

    The documented path must produce none of these; the collapsed path must
    produce exactly one per configured format.
    """
    return sorted(str(p) for p in rglob_files(dump_dir) if p.endswith(".log"))


def assert_upstream_response(status: int, body: bytes) -> None:
    """Valid requests traverse the live peer without a proxy transport failure.

    The status class is the whole oracle. A proxy-local failure to reach the peer
    is answered ``502``, so ``< 500`` separates "the peer answered" from "the
    proxy could not reach it"; the live service may accept the request or reject a
    test model name, and neither its generated output nor its particular
    client-error payload is a QA oracle.

    The body is deliberately **not** inspected. A relayed answer is the peer's own
    bytes, so any substring this helper looked for would be attributable to the
    proxy only if the proxy authored it — and it does not. A former
    ``"Cannot connect" not in body`` limb was dropped on that ground: the proxy's
    own failure bodies no longer embed the cause, so the limb could not fire from
    a transport failure, while a peer answer legitimately saying so would have
    tripped an oracle documented as the proxy's own connectivity. Whether an error
    body leaks the proxy's configuration is a different property and is witnessed
    at the cheaper tier that owns it — ``tests/test_upstream_answer_relay.py``'s
    ``TestLeakFreeErrorBodies`` — not duplicated here.
    """
    assert 200 <= status < 500, f"valid request failed with HTTP {status}: {body[:300]!r}"


class Proxy:
    """A running proxy subprocess with stdout/stderr captured to files.

    ``out_dir`` must live OUTSIDE ``dump_dir`` so captured logs never pollute
    the directory listings the gates assert on.
    """

    def __init__(
        self,
        dump_dir,
        out_dir,
        extra_args=(),
        counted_report=None,
        upstream=UPSTREAM,
        require_live_upstream=True,
        probe_report=None,
    ):
        """Start the real proxy subprocess.

        ``upstream`` / ``require_live_upstream`` exist for the one gate the
        ``qa-project`` §Upstream amendment exempts from the live remote
        upstream: the ``MS-004`` upstream-TLS gate names its own loopback peer
        and must not depend on ``hyperipper.local`` being reachable. Every
        other gate keeps the defaults, including the fail-closed preflight.
        ``probe_report`` selects the client-posture instrumented entry
        (:mod:`tests.qa.tls_probe_entry`); it is exclusive with
        ``counted_report``.
        """
        if counted_report is not None and probe_report is not None:
            raise ValueError("counted_report and probe_report are mutually exclusive")
        if require_live_upstream:
            require_upstream()
        self.dump_dir = os.path.abspath(str(dump_dir))
        self.upstream = upstream
        self.port = free_port()
        os.makedirs(str(out_dir), exist_ok=True)
        self.out_path = os.path.join(str(out_dir), "proxy.out")
        self.err_path = os.path.join(str(out_dir), "proxy.err")
        base_args = [
            "--host", "127.0.0.1",
            "--port", str(self.port),
            "--upstream", upstream,
            "--dump-dir", self.dump_dir,
            *extra_args,
        ]
        if counted_report is not None:
            cmd = [
                sys.executable, COUNTED_ENTRY,
                "--report", str(counted_report),
                "--dump-root", self.dump_dir,
                "--", *base_args,
            ]
        elif probe_report is not None:
            cmd = [
                sys.executable, TLS_PROBE_ENTRY,
                "--report", str(probe_report),
                "--", *base_args,
            ]
        else:
            cmd = [PROXY_BIN, *base_args]
        self._out_f = open(self.out_path, "wb")
        self._err_f = open(self.err_path, "wb")
        self.proc = subprocess.Popen(cmd, stdout=self._out_f, stderr=self._err_f)
        if not wait_tcp(self.port):
            self.stop(signal.SIGKILL)
            raise RuntimeError(
                f"proxy did not become ready on port {self.port}; stderr="
                f"{self.stderr()[:2000]!r}"
            )

    # -- driving ---------------------------------------------------------
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}/v1/messages"

    def post(self, body, timeout: float = 90.0):
        """POST a body (dict -> JSON bytes, or raw bytes). Returns (status, body).

        The opener is built with an empty ``ProxyHandler``, exactly as
        :func:`require_upstream` builds its probe: the oracle must judge the
        local proxy's relay of the declared upstream's answer, never an ambient
        ``http_proxy``'s answer for a request that never reached the gate
        (``S22``). The proxy subprocess is direct too (``trust_env=False``).
        """
        data = json.dumps(body).encode("utf-8") if isinstance(body, dict) else body
        req = urllib.request.Request(
            self.url(), data=data, headers={"Content-Type": "application/json"}
        )
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        try:
            with opener.open(req, timeout=timeout) as resp:
                return resp.status, resp.read()
        except urllib.error.HTTPError as e:
            return e.code, e.read()

    # -- observation -----------------------------------------------------
    def stderr(self) -> str:
        try:
            with open(self.err_path, "rb") as f:
                return f.read().decode("utf-8", errors="replace")
        except FileNotFoundError:
            return ""

    def session_dir(self, system_prompt: str) -> str:
        return os.path.join(self.dump_dir, session_id(system_prompt))

    def artifact_path(self, system_prompt: str, fmt: str) -> str:
        return os.path.join(self.session_dir(system_prompt), f"{fmt}.log")

    # -- lifecycle -------------------------------------------------------
    def stop(self, sig=signal.SIGTERM, timeout: float = 30.0):
        """Send a real signal, wait for exit, close captures. Returns returncode."""
        if self.proc.poll() is None:
            self.proc.send_signal(sig)
            try:
                self.proc.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait(timeout=10)
        rc = self.proc.returncode
        for f in (self._out_f, self._err_f):
            try:
                f.close()
            except OSError:
                pass
        return rc


def startup_refused(extra_args, dump_dir, out_dir, timeout: float = 30.0) -> tuple[int, str]:
    """Run the real binary and require it to refuse to start.

    The negative counterpart of :class:`Proxy`: for inputs the CLI must reject
    outright (an unknown selector value, a malformed option), the process never
    reaches a listening port. Returns ``(returncode, stderr)``; raises if the
    process started and stayed up, which is the refusal failing.

    ``extra_args`` is the scenario-specific remainder; the bound-port/upstream
    arguments this harness always fixes are supplied here so a refusal is about
    the argument under test and nothing else.
    """
    os.makedirs(str(out_dir), exist_ok=True)
    err_path = os.path.join(str(out_dir), "refused.err")
    out_path = os.path.join(str(out_dir), "refused.out")
    cmd = [
        PROXY_BIN,
        "--host", "127.0.0.1",
        "--port", str(free_port()),
        "--upstream", UPSTREAM,
        "--dump-dir", os.path.abspath(str(dump_dir)),
        *extra_args,
    ]
    with open(out_path, "wb") as out_f, open(err_path, "wb") as err_f:
        try:
            proc = subprocess.run(
                cmd, stdout=out_f, stderr=err_f, timeout=timeout
            )
        except subprocess.TimeoutExpired:
            raise AssertionError(
                f"expected a startup refusal for {list(extra_args)}, but the process ran on; "
                f"stderr={read_text(err_path)[-2000:]!r}"
            ) from None
    with open(err_path, encoding="utf-8", errors="replace") as f:
        return proc.returncode, f.read()
