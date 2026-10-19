"""Instrumented proxy entry for the upstream-TLS posture gates (TC-114 evidence).

Runs the real CLI in a real process. Only ``aiohttp``'s public client
constructors are wrapped, to answer the one question ``qa-plan.md`` assigns to
this tier — the plan's own words: *"exactly one `ClientSession` exists ... so
one connector governs every outbound call"* and its evidence source, *"one
connector assertion + the instrumented call sites"*. Nothing in the proxy is
patched or reordered; the posture itself is proven black-box by the loopback
peer's request log, and these counts only establish that a single client object
could have served everything the peer saw.

Usage::

    python tls_probe_entry.py --report PATH -- <proxy args>

At process exit a JSON report is written::

    {"connectors": [{"index", "id", "ssl_kwarg", "positionals", "kwargs"}],
     "sessions":   [{"index", "id", "connector_kwarg", "ssl_kwarg", "connector_id"}],
     "calls":      [{"method", "path", "session_id", "connector_id"}],
     "total_connectors": int, "total_sessions": int, "total_calls": int,
     "call_hook": "installed" | "unavailable"}
"""

from __future__ import annotations

import argparse
import json
import sys

connectors: list[dict] = []
sessions: list[dict] = []
calls: list[dict] = []
_ABSENT = "<absent>"


def _posture(kwargs: dict):
    """How the caller spelled the ssl posture, or a marker that it did not."""
    return repr(kwargs["ssl"]) if "ssl" in kwargs else _ABSENT


def _main() -> int:
    argv = sys.argv[1:]
    try:
        sep = argv.index("--")
    except ValueError:
        raise SystemExit("usage: tls_probe_entry.py --report PATH -- <proxy args>")
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", required=True)
    args = parser.parse_args(argv[:sep])
    proxy_argv = argv[sep + 1:]

    import aiohttp

    real_connector_init = aiohttp.TCPConnector.__init__
    real_session_init = aiohttp.ClientSession.__init__

    def connector_init(self, *init_args, **init_kwargs):
        real_connector_init(self, *init_args, **init_kwargs)
        connectors.append(
            {
                "index": len(connectors),
                "id": id(self),
                # The plan settles the construction as TCPConnector(ssl=...);
                # ssl is keyword-only there, and the positionals are recorded so
                # a posture spelled another way is still visible in the report.
                "ssl_kwarg": _posture(init_kwargs),
                "positionals": [repr(a)[:80] for a in init_args],
                "kwargs": sorted(init_kwargs),
            }
        )

    def session_init(self, *init_args, **init_kwargs):
        real_session_init(self, *init_args, **init_kwargs)
        connector = getattr(self, "connector", None)
        sessions.append(
            {
                "index": len(sessions),
                "id": id(self),
                "connector_kwarg": "connector" in init_kwargs,
                "ssl_kwarg": _posture(init_kwargs),
                "connector_id": id(connector) if connector is not None else None,
            }
        )

    aiohttp.TCPConnector.__init__ = connector_init
    aiohttp.ClientSession.__init__ = session_init

    call_hook = "unavailable"
    real_request = getattr(aiohttp.ClientSession, "_request", None)
    if real_request is not None:

        async def request(self, method, str_or_url, *a, **kw):
            connector = getattr(self, "connector", None)
            try:
                path = str(str_or_url)
            except Exception:  # noqa: BLE001 - diagnostics only
                path = "<unprintable>"
            calls.append(
                {
                    "method": str(method),
                    "path": path,
                    "session_id": id(self),
                    "connector_id": id(connector) if connector is not None else None,
                }
            )
            return await real_request(self, method, str_or_url, *a, **kw)

        aiohttp.ClientSession._request = request
        call_hook = "installed"

    exit_code = 0
    try:
        import buster_ripper.main as cli

        sys.argv = ["buster-ripper"] + proxy_argv
        try:
            cli.main()
        except SystemExit as e:
            exit_code = e.code if isinstance(e.code, int) else (0 if e.code is None else 1)
    finally:
        connector_ids = {c["id"] for c in connectors}
        report = {
            "connectors": connectors,
            "sessions": sessions,
            "calls": calls,
            "total_connectors": len(connectors),
            "total_sessions": len(sessions),
            "total_calls": len(calls),
            "call_hook": call_hook,
            # Derived once, here, so the gate's assertions read as facts:
            "connector_ids": sorted(connector_ids),
            "distinct_connector_ids_serving_calls": sorted(
                {c["connector_id"] for c in calls if c["connector_id"] is not None}
            ),
        }
        try:
            with open(args.report, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2)
        except OSError as exc:
            print(f"tls_probe_entry: could not write report: {exc}", file=sys.stderr)
    return exit_code


if __name__ == "__main__":
    sys.exit(_main())
