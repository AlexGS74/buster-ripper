"""CR4 probe (package app): status matrix across the range for one tree.

Run as: PYTHONPATH=<tree> python probe_pkg.py <label> [axis]
  axis = default | stripdate-sglang
Read-only against the repo; boots the real create_app() on an in-process
TestServer, exactly as tests/conftest.py does.
"""
import asyncio
import json
import logging
import sys

from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from buster_ripper import backend, config
from buster_ripper.server import create_app

CAPTURED: list[str] = []


class _Grab(logging.Handler):
    def emit(self, record):
        try:
            CAPTURED.append(self.format(record))
        except Exception:
            pass


def frames(text: str) -> list[str]:
    """The in-project frames of a captured traceback, innermost last."""
    out = []
    for line in text.splitlines():
        s = line.strip()
        if s.startswith('File "') and ("buster_ripper" in s or "normalize.py" in s):
            out.append(s)
    return out


BODIES = [
    ('{"tools":[null]}', "tools non-mapping element"),
    ('{"messages":[null]}', "messages non-mapping element"),
    ('{"messages":["x"]}', "messages str element"),
    ('{"system":[null]}', "system non-mapping block"),
    ('{"messages":[{"content":[null]}]}', "content non-mapping block"),
    ('{"messages":[{"role":"user","content":"x"}]}', "well-formed control"),
    ('{"messages":[{"role":"user","content":[{"cache_control":{"type":"ephemeral"},"text":"Today\'s date is 2026-02-21.\\nhi"}]}]}',
     "date+cache_control well-formed"),
    ('[1,2]', "non-dict body"),
    ('not json', "unparseable body"),
]


async def main(label: str, axis: str) -> None:
    handler = _Grab()
    handler.setFormatter(logging.Formatter("%(message)s"))
    logging.getLogger("aiohttp.server").addHandler(handler)
    logging.getLogger("aiohttp.server").setLevel(logging.ERROR)

    app = create_app()
    async def _noop_detect(*a, **k):
        return None

    backend.detect_backend = _noop_detect  # type: ignore[assignment]
    backend.get_backend_state = lambda: ("sglang" if axis == "stripdate-sglang" else "vllm")  # type: ignore[assignment]
    config.STRIP_IMAGES = False
    config.STRIP_DATE = axis == "stripdate-sglang"
    config.UPSTREAM = "http://127.0.0.1:1"  # nothing listens: relay arm answers 502

    print(f"=== {label}  axis={axis}  STRIP_DATE={config.STRIP_DATE} backend={backend.get_backend_state()}")
    print(f"    package under test: {sys.modules['buster_ripper'].__file__}")
    async with TestServer(app) as server:
        async with TestClient(server) as client:
            for route in ("/v1/messages/count_tokens", "/v1/messages"):
                print(f"  --- {route}")
                for raw, note in BODIES:
                    CAPTURED.clear()
                    resp = await client.post(route, data=raw.encode(),
                                             headers={"Content-Type": "application/json"})
                    status = resp.status
                    await resp.read()
                    tb = [t for t in CAPTURED if "Traceback" in t]
                    exc = ""
                    site = ""
                    if tb:
                        last = tb[-1]
                        for line in last.splitlines():
                            if line.strip().startswith(("AttributeError", "TypeError",
                                                          "ValueError", "LookupError",
                                                          "UnicodeDecodeError", "json.")):
                                exc = line.strip()
                        f = frames(last)
                        if f:
                            site = f[-1].replace('File "', "").rstrip('",')
                    print(f"    {status:>3}  {raw[:58]:<58} {note:<28}"
                          f"{('  ' + exc) if exc else ''}")
                    if exc:
                        print(f"         escape site: {site}")
                    elif status >= 500:
                        print(f"         5xx without captured traceback: {CAPTURED[:1]}")


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "default"))
