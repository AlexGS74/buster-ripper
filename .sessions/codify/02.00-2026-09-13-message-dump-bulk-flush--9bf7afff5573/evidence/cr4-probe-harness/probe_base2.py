"""CR4 probe (base monolith), round 2: with a live upstream stub so a 500 is
attributable, and an ASGI wrapper that records the escaping exception.

Run: uv run --no-project --with fastapi --with httpx --with uvicorn --with click \
       python probe_base2.py <base-tree> [axis]
"""
import asyncio
import json
import sys
import threading
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from fastapi.testclient import TestClient

sys.path.insert(0, sys.argv[1])
import buster_ripper as m  # noqa: E402

AXIS = sys.argv[2] if len(sys.argv) > 2 else "default"

UPSTREAM_OK = json.dumps({
    "id": "msg_stub", "type": "message", "role": "assistant",
    "content": [{"type": "text", "text": "ok"}],
    "usage": {"input_tokens": 11, "output_tokens": 3},
}).encode()


class _Handler(BaseHTTPRequestHandler):
    def do_POST(self):  # noqa: N802
        length = int(self.headers.get("content-length", "0"))
        self.rfile.read(length)
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(UPSTREAM_OK)))
        self.end_headers()
        self.wfile.write(UPSTREAM_OK)

    def log_message(self, *a):
        pass


stub = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
threading.Thread(target=stub.serve_forever, daemon=True).start()
port = stub.server_address[1]

ESCAPED: dict[str, str] = {}


class _Record:
    """Pure ASGI wrapper: records the exception that reaches the server boundary."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        try:
            await self.app(scope, receive, send)
        except Exception:
            ESCAPED[scope.get("path", "?")] = traceback.format_exc()
            raise


BODIES = [
    ('{"tools":[null]}', "tools non-mapping element"),
    ('{"messages":[null]}', "messages non-mapping element"),
    ('{"messages":["x"]}', "messages str element"),
    ('{"system":[null]}', "system non-mapping block"),
    ('{"messages":[{"content":[null]}]}', "content non-mapping block"),
    ('{"messages":[{"role":"user","content":"x"}]}', "well-formed control"),
    ('[1,2]', "non-dict body"),
    ('not json', "unparseable body"),
]

print(f"=== base main {m.__file__} axis={axis_label if (axis_label:=AXIS) else ''} "
      f"STRIP_DATE={m.STRIP_DATE} upstream=stub:{port}")
m.UPSTREAM = f"http://127.0.0.1:{port}"
m.STRIP_DATE = AXIS == "stripdate-sglang"

print("  --- normalize() called directly on base bytes (pure function, no transport)")
for raw, note in BODIES:
    try:
        parsed = json.loads(raw)
    except Exception:
        parsed = {}
    try:
        m.normalize(dict(parsed) if isinstance(parsed, dict) else parsed)
        print(f"    ok       {raw[:56]:<56} {note}")
    except Exception as e:
        fr = traceback.extract_tb(e.__traceback__)[-1]
        print(f"    RAISES   {raw[:56]:<56} {note} -> {type(e).__name__} at {fr.filename.rsplit('/',1)[-1]}:{fr.lineno} in {fr.name}")

print("  --- HTTP status with a live upstream stub")
with TestClient(_Record(m.app), raise_server_exceptions=False, base_url="http://test") as client:
    for route in ("/v1/messages/count_tokens", "/v1/messages"):
        print(f"  --- {route}")
        for raw, note in BODIES:
            key = route + "|" + raw
            ESCAPED.clear()
            resp = client.post(route, content=raw.encode(),
                               headers={"Content-Type": "application/json"})
            extra = ""
            route = key.split("|")[0]
            if route in ESCAPED:
                tb = ESCAPED[route]
                sites = [ln.strip().replace('File "', "")
                         for ln in tb.splitlines() if ln.strip().startswith('File "') and "buster_ripper.py" in ln]
                ex = [ln.strip() for ln in tb.splitlines()
                      if ln.strip().startswith(("AttributeError", "TypeError", "ValueError", "json."))]
                extra = f"   escaped {ex[-1] if ex else '?'} @ {sites[-1].split('line ')[-1] if sites else '?'}"
                ESCAPED.clear()
            print(f"    {resp.status_code:>3}  {raw[:56]:<56} {note:<28}{extra}")
stub.shutdown()
