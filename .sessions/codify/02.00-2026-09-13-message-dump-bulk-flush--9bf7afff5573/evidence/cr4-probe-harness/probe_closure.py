"""CR4-F-01 closure verification on HEAD bytes (all simulation is in-memory;
no repo byte is written).

A  digest equality: session_id(body) after full normalize() == after the
   system-strip step alone, over a battery that includes the documented
   Claude Code shape (billing nonce at system position 0, cache_control on
   system and message blocks).
B  route-level effect of the narrowed call: re-run the count_tokens matrix with
   the route's `normalize` rebound to the system-strip step.
C  masking: is normalize.py:192 independently reachable today, and does it
   become reachable if :117 alone is hardened?
D  the alternative's blast radius: what a hardened normalize() does to
   /v1/messages, where base 500s.
"""
import asyncio
import copy
import json
import sys
import traceback
from contextlib import suppress

sys.path.insert(0, sys.argv[1])
from aiohttp import web  # noqa: E402
from aiohttp.test_utils import TestClient, TestServer  # noqa: E402

from buster_ripper import backend, config, normalize as nz, session as session_mod  # noqa: E402
from buster_ripper.normalize import normalize  # noqa: E402
from buster_ripper.routes import count_tokens as ct  # noqa: E402
from buster_ripper.server import create_app  # noqa: E402
from buster_ripper.session import session_id  # noqa: E402


def narrow(body):
    """The proposed narrowed call: only the step that writes system's text."""
    return nz._normalize_system(body)


NONCE = "x-anthropic-billing-header: cc_version=2.1.50.f15; cc_entrypoint=cli; cch=27acd;\n"
BATTERY = [
    ("documented client, string system, nonce",
     {"system": NONCE + "You are a Claude Code assistant.",
      "messages": [{"role": "user", "content": [{"type": "text", "text": "hi"}]}],
      "tools": [{"name": "b"}, {"name": "a"}],
      "max_tokens": 64}),
    ("documented client, block system, nonce + cache_control",
     {"system": [{"type": "text", "text": NONCE.rstrip() + "\nYou are helpful.",
                  "cache_control": {"type": "ephemeral"}},
                 {"type": "text", "text": "Static tail."}],
      "messages": [{"role": "user", "content": [
          {"type": "text", "text": "Today's date is 2026-02-21.\nhi",
           "cache_control": {"type": "ephemeral"}}]}],
      "tools": [{"name": "z"}, {"name": "a"}, {"name": "m"}]}),
    ("nonce-only system block (becomes empty after strip)",
     {"system": [{"type": "text", "text": NONCE.rstrip()}], "messages": []}),
    ("no system key", {"messages": [{"role": "user", "content": "x"}]}),
    ("system non-mapping block", {"system": [None], "messages": []}),
    ("system list of mixed", {"system": [None, {"text": "abc"}, "raw"], "messages": []}),
    ("tools reordered, system str without nonce",
     {"system": "plain", "tools": [{"name": "q"}, {"name": "p"}]}),
    ("cache_control on system blocks only",
     {"system": [{"type": "text", "text": "abc", "cache_control": {"type": "ephemeral"}}]}),
    ("images present (strip path)",
     {"system": "s", "messages": [{"role": "user", "content": [
         {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "AAAA"}}]}]}),
]

print(f"=== CR4-F-01 closure verification on {sys.argv[1]}")
print("--- A: digest equality, full normalize() vs system-strip step only")
for axis, strip_date, be in (("vllm/no-strip-date", False, "vllm"),
                             ("sglang/STRIP_DATE", True, "sglang")):
    config.STRIP_DATE = strip_date
    config.STRIP_IMAGES = True
    backend.get_backend_state = lambda be=be: be
    bad = []
    for label, body in BATTERY:
        for nonce_flip in (0, 1):
            a = copy.deepcopy(body)
            b = copy.deepcopy(body)
            if nonce_flip:
                c = a.get("system")
                if isinstance(c, str):
                    a["system"] = NONCE.replace("27acd", f"zzz{nonce_flip}") + c
                elif isinstance(c, list) and c and isinstance(c[0], dict) and "text" in c[0]:
                    c[0]["text"] = NONCE.replace("27acd", f"zzz{nonce_flip}") + c[0]["text"]
            try:
                normalize(a, session_id="")
            except Exception as e:
                bad.append((label, "FULL-RAISES", type(e).__name__, ""))
            try:
                narrow(b)
            except Exception as e:
                bad.append((label, "NARROW-RAISES", type(e).__name__, ""))
            da, db = session_id(a), session_id(b)
            if da != db:
                bad.append((label, nonce_flip, da, db))
        print(f"    [{axis:>18}] {label:<52} {'MATCH' if not any(x[0] == label for x in bad) else 'DIFFER'}")
    print(f"    [{axis:>18}] differing digests across {len(BATTERY)*2} cases: {len(bad)}")
    for x in bad:
        print(f"        DIFFER {x}")

config.STRIP_DATE = False
config.STRIP_IMAGES = False
backend.get_backend_state = lambda: "vllm"


async def probe(routes_bodies):
    app = create_app()

    async def _noop(*a, **k):
        return None

    backend.detect_backend = _noop
    config.UPSTREAM = "http://127.0.0.1:1"
    out = {}
    async with TestServer(app) as server:
        async with TestClient(server) as client:
            for route, raw in routes_bodies:
                r = await client.post(route, data=raw.encode(),
                                      headers={"Content-Type": "application/json"})
                out[(route, raw)] = r.status
                await r.read()
    return out


TRIO = ['{"tools":[null]}', '{"messages":[null]}', '{"messages":["x"]}']
EXTRA = ['{"system":[null]}', '{"messages":[{"content":[null]}]}', '[1,2]', 'not json']

print("--- B: count_tokens with the route's normalize rebound to the system-strip step")
orig_ct = ct.normalize
ct.normalize = lambda body, session_id="": (body, narrow(body))
matrix = asyncio.run(probe([("/v1/messages/count_tokens", b) for b in TRIO + EXTRA]))
for (route, raw), status in matrix.items():
    print(f"    {status:>3}  {raw}")
ct.normalize = orig_ct

print("--- C: is normalize.py:192 (_normalize_date) independently reachable today?")
for raw in TRIO:
    body = json.loads(raw)
    config.STRIP_DATE = True
    backend.get_backend_state = lambda: "sglang"
    try:
        normalize(body, session_id="")
        print(f"    {raw:<24} normalize() ok")
    except Exception as e:
        fr = traceback.extract_tb(e.__traceback__)[-1]
        print(f"    {raw:<24} raises {type(e).__name__} at {fr.filename.rsplit('/',1)[-1]}:{fr.lineno} ({fr.name})")
    config.STRIP_DATE = False

print("    with :117 alone hardened (simulated skip of non-mapping messages):")
orig_cc = nz._strip_cache_control_from_messages
nz._strip_cache_control_from_messages = lambda msgs: orig_cc(
    [m for m in msgs if isinstance(m, dict)])
config.STRIP_DATE = True
backend.get_backend_state = lambda: "sglang"
for raw in TRIO:
    body = json.loads(raw)
    try:
        normalize(body, session_id="")
        print(f"    {raw:<24} normalize() ok")
    except Exception as e:
        fr = traceback.extract_tb(e.__traceback__)[-1]
        print(f"    {raw:<24} raises {type(e).__name__} at {fr.filename.rsplit('/',1)[-1]}:{fr.lineno} ({fr.name})")
nz._strip_cache_control_from_messages = orig_cc
config.STRIP_DATE = False

print("--- D: blast radius of hardening normalize()'s two element loops, on /v1/messages")
orig_tools = nz._normalize_tools
orig_cc2 = nz._strip_cache_control_from_messages
orig_date = nz._normalize_date


def hard_tools(body):
    """The candidate repair: the tools loop stops assuming a mapping element."""
    tools = body.get("tools")
    if not isinstance(tools, list) or not tools:
        return orig_tools(body)
    name = lambda t: t.get("name", "") if isinstance(t, dict) else ""  # noqa: E731
    body["tools"] = sorted(tools, key=name)
    return ["tools reordered (hardened)"]


def hard_cc(msgs):
    """The candidate repair: the messages loop stops assuming a mapping element."""
    return orig_cc2([m for m in msgs if isinstance(m, dict)])


def hard_date(body):
    """And its successor: the date loop, same class, :192."""
    msgs = body.get("messages")
    if isinstance(msgs, list):
        body["messages"] = [m for m in msgs if isinstance(m, dict)]
    return orig_date(body)


matrix = asyncio.run(probe([("/v1/messages", b) for b in TRIO]))
print("    as landed (HEAD)   [base does the same: 500 at normalize :374/:402]")
for (route, raw), status in matrix.items():
    print(f"      {status:>3}  {raw}")
print("    with the two element loops hardened (simulated):")
nz._normalize_tools = hard_tools
nz._strip_cache_control_from_messages = hard_cc
matrix = asyncio.run(probe([("/v1/messages", b) for b in TRIO]))
for (route, raw), status in matrix.items():
    print(f"      {status:>3}  {raw}   <- diverges from base's 500 on the same body")
print("    with all three loops hardened (simulated):")
nz._normalize_date = hard_date
matrix = asyncio.run(probe([("/v1/messages", b) for b in TRIO]))
for (route, raw), status in matrix.items():
    print(f"      {status:>3}  {raw}")
nz._normalize_tools = orig_tools
nz._strip_cache_control_from_messages = orig_cc2
nz._normalize_date = orig_date
