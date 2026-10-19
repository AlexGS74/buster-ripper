"""Instrumented proxy entry for the write-count gates (TC-105 evidence).

Runs the real CLI in a real process; only the file-open layer is wrapped to
count write-bearing operations against the dump root. The counted unit is the
one the plan settles: one write-bearing file operation on a dump artifact
(``write_text``, or ``open(..., "a"/"w")`` + ``write``). Directory creation is
not counted (once per session) and the SQLite stats path is outside the
measured surface (``--stats-db`` is never passed by the gates).

Usage::

    python counted_proxy_entry.py --report PATH --dump-root DIR -- <proxy args>

At process exit a JSON report is written::

    {"dump_root": ..., "ops": [{"path", "mode", "bytes"}, ...],
     "total_ops": int, "total_bytes": int}
"""

from __future__ import annotations

import argparse
import builtins
import io
import json
import os
import sys

_REAL_OPEN = io.open


class _CountedFile:
    """Pass-through file wrapper that accumulates written bytes for one op."""

    def __init__(self, inner, record):
        self._inner = inner
        self._record = record

    def write(self, data):
        if isinstance(data, str):
            try:
                n = len(data.encode(self._inner.encoding or "utf-8"))
            except (AttributeError, UnicodeEncodeError):
                n = len(data)
        else:
            n = len(data)
        self._record["bytes"] += n
        return self._inner.write(data)

    def writelines(self, lines):
        for line in lines:
            self.write(line)
        return None

    def __enter__(self):
        self._inner.__enter__()
        return self

    def __exit__(self, *exc_info):
        return self._inner.__exit__(*exc_info)

    def __iter__(self):
        return iter(self._inner)

    def __getattr__(self, name):
        return getattr(self._inner, name)


def main() -> None:
    argv = sys.argv[1:]
    try:
        sep = argv.index("--")
    except ValueError:
        raise SystemExit(
            "usage: counted_proxy_entry.py --report P --dump-root D -- <proxy args>"
        )
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", required=True)
    parser.add_argument("--dump-root", required=True)
    args = parser.parse_args(argv[:sep])
    proxy_argv = argv[sep + 1:]

    root = os.path.abspath(args.dump_root)
    ops = []

    def _under_root(path_str: str) -> bool:
        p = os.path.abspath(path_str)
        return p == root or p.startswith(root + os.sep)

    def counting_open(file, mode="r", *rest, **kwargs):
        inner = _REAL_OPEN(file, mode, *rest, **kwargs)
        if isinstance(mode, str) and "r" not in mode and (
            "w" in mode or "a" in mode or "x" in mode or "+" in mode
        ):
            try:
                path_str = os.fspath(file)
            except TypeError:
                path_str = None
            if path_str is not None and _under_root(path_str):
                record = {
                    "path": os.path.abspath(path_str),
                    "mode": mode,
                    "bytes": 0,
                }
                ops.append(record)
                return _CountedFile(inner, record)
        return inner

    io.open = counting_open
    builtins.open = counting_open
    try:
        import buster_ripper.main as cli

        sys.argv = ["buster-ripper"] + proxy_argv
        exit_code = 0
        try:
            cli.main()
        except SystemExit as e:
            exit_code = e.code if isinstance(e.code, int) else (0 if e.code is None else 1)
    finally:
        io.open = _REAL_OPEN
        builtins.open = _REAL_OPEN
        report = {
            "dump_root": root,
            "ops": ops,
            "total_ops": len(ops),
            "total_bytes": sum(o["bytes"] for o in ops),
        }
        with _REAL_OPEN(args.report, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
