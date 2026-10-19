VERDICT: HOLDS
PROMOTION: INELIGIBLE — the claim is exhaustive over the package; the establishing evidence is a directory-wide enumeration of every outbound request call site under `buster_ripper/` (broad directory), not one exact file's bytes. Two exact files do record the full argument lists of the only two out-of-routes call sites, and their fingerprints are below.

## Claim
No outbound HTTP request issued by a module outside `buster_ripper/routes/` passes a `headers` argument

## What it actually does
There are exactly six outbound HTTP request call sites in the package — four inside `buster_ripper/routes/` and two outside it. Neither of the two outside `routes/` passes `headers`:

- `buster_ripper/backend.py:54-56` — `session.get(f"{backend_url}/v1/models", timeout=aiohttp.ClientTimeout(total=5))`. URL + `timeout` only. The `/v1/models` backend probe goes out unauthenticated and unheadered.
- `buster_ripper/compaction.py:36-39` — `session.get(f"{config.UPSTREAM}/metrics", timeout=timeout)`. URL + `timeout` only. The Prometheus `/metrics` poll likewise carries no headers.

The four call sites that *do* pass `headers=headers` are all inside `routes/`: `routes/chat.py:128`, `routes/messages.py:133`, `routes/messages.py:234`, `routes/passthrough.py:33` — each forwarding headers built from the incoming `web.Request` (`chat.py:98` `_prepare_headers`, `messages.py:61-63`, `passthrough.py:14-16`). So header forwarding is confined to the proxy route handlers; the non-route modules' probes are bare GETs.

## Source
Exhaustive enumeration over the bounded source root `/Users/alexhubitski/Projects/buster-ripper/buster_ripper/` (all `.py`, `__pycache__` excluded):
- `grep -rn "session\.\(get\|post\|request\|put\|patch\|delete\|head\|options\)"` → the six sites above, no others.
- `grep -rn "headers"` across the package → every `headers=` argument in an outbound *request* lies under `routes/`; the matches outside `routes/` are only the `utils.py:18/27` helper definitions (which are never called) and a docstring at `dump.py:263` about diff file headers.
- `grep -rn "aiohttp"` outside `routes/` → `server.py`, `backend.py`, `compaction.py` only. `server.py:65-68` just constructs the package's single `aiohttp.ClientSession` (`timeout=`, `connector=` — no `headers=` default, so no request inherits headers either); it issues no request. `main.py:187` imports `aiohttp.web` for the server side.
- No other HTTP/network client exists in the package: no `httpx`/`requests`/`urllib`/`socket`/`subprocess`/`curl`, no `websockets`/`ws_connect`/`ClientRequest`/`urlopen`/`open_connection` (all greps empty). `session.py:36` `sqlite3.connect` is the only other `.connect(` and is not HTTP. `image.py`, `dump.py`, `normalize.py`, `config.py`, `__init__.py` perform no I/O to the network.
- Headers are keyword-only in aiohttp's client API, so a positional headers argument is not possible at the two sites; neither passes `**kwargs` either — both argument lists are read in full above.

Terminology note (per P9, no choice was needed to reach HOLDS): `web.Response`/`web.StreamResponse(headers=...)` at `chat.py:159-161`, `chat.py:191/200`, `messages.py:147/264`, `passthrough.py:48` are responses sent to *downstream* clients, not outbound requests, and in any case all live inside `routes/` — so neither reading of "outbound HTTP request" changes the verdict.

Checked independently of the two sibling claims about `utils.forward_headers` / `utils.response_headers`; this verdict rests on the request call-site enumeration, not on those results.

## Source Version
local working tree, `buster_ripper/` package (no version/revision consulted — out of scope for this role)

## Source Fingerprint
sha256:c2b71468af6b53efde753b0648d47eb6e06c8db4c738e08f7c01f4e4682d76ef  (buster_ripper/backend.py — full file read; call site at 54-56)
sha256:ac9d42143c98694a0b6672dd753afcef537fc5e03ec747b4bb3c1208e23535e7  (buster_ripper/compaction.py — full file read; call site at 36-39)
Note: these fingerprint the two out-of-routes request sites, not the package-wide enumeration the exhaustive half of the claim rests on; that is why promotion is ineligible.

## Invalidates On
- An edit adding `headers=` (or a `**kwargs` spread carrying it) to the probe at `buster_ripper/backend.py:54` or the poll at `buster_ripper/compaction.py:36`
- A new non-route module — or new code in `server.py`, `session.py`, `dump.py`, `image.py`, `normalize.py`, `config.py`, `main.py` — that issues an HTTP request with headers
- Relocation of a proxy handler out of `buster_ripper/routes/`, or a new outbound request path added to a non-route module (e.g. header passthrough moved into a shared client helper)
- A `headers=` default added to the `aiohttp.ClientSession(...)` constructor at `buster_ripper/server.py:66`, which would put headers on every outbound request without naming them at any call site

## Checked At
2026-10-05T18:08:51Z
