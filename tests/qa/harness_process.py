"""Process driver for the buster-ripper QA gates: the real proxy subprocess and its timing helpers."""

from __future__ import annotations

import json
import os
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request

from tests.qa.harness import read_text, session_id


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


def _instrumented_entry(counted_report, probe_report, dump_dir) -> list[str] | None:
    """The argv prefix of the instrumented entry the caller selected; None for the plain binary."""
    if counted_report is not None:
        return [
            sys.executable, COUNTED_ENTRY,
            "--report", str(counted_report),
            "--dump-root", dump_dir,
        ]
    if probe_report is not None:
        return [sys.executable, TLS_PROBE_ENTRY, "--report", str(probe_report)]
    return None


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
        entry_prefix = _instrumented_entry(counted_report, probe_report, self.dump_dir)
        cmd = self._build_command(extra_args, entry_prefix)
        self._out_f = open(self.out_path, "wb")
        self._err_f = open(self.err_path, "wb")
        self.proc = subprocess.Popen(cmd, stdout=self._out_f, stderr=self._err_f)
        if not wait_tcp(self.port):
            self.stop(signal.SIGKILL)
            raise RuntimeError(
                f"proxy did not become ready on port {self.port}; stderr="
                f"{self.stderr()[:2000]!r}"
            )

    def _build_command(self, extra_args, entry_prefix) -> list[str]:
        """The proxy invocation, wrapped by the instrumented entry the caller selected."""
        base_args = [
            "--host", "127.0.0.1",
            "--port", str(self.port),
            "--upstream", self.upstream,
            "--dump-dir", self.dump_dir,
            *extra_args,
        ]
        if entry_prefix is None:
            return [PROXY_BIN, *base_args]
        return [*entry_prefix, "--", *base_args]

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

