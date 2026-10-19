VERDICT: HOLDS
PROMOTION: INELIGIBLE — no single source locator: a universal over six call sites in six modules, established by exhaustive enumeration plus absence of any alternative HTTP mechanism in the package

## Claim
Every outbound HTTP call in the `buster_ripper` package goes through `app["session"]`

## What it actually does
The package has exactly six outbound HTTP call sites, all invoked on the session stored as `app["session"]` (built once in `server.py:66`): `backend.py:54` (`session.get`, session passed from `server.py:72` as `session=app["session"]`), `compaction.py:36` (`session.get`, session from `server.py:21` `app["session"]`), `routes/chat.py:125` (`session.post`, session from `request.app["session"]` at chat.py:37), `routes/messages.py:130` and `:231` (`session.post`, session from `request.app["session"]` at messages.py:127/228), and `routes/passthrough.py:29` (`session.request`, session from `request.app["session"]` at passthrough.py:25). Grep across `buster_ripper/` finds no `aiohttp.request(`, `httpx`, `requests`, `urllib`, `http.client`, raw socket, or websocket usage — no other outbound HTTP path exists. (The repo-root module `buster_ripper_legacy.py` does use its own `httpx.AsyncClient`s, but it is outside the `buster_ripper/` package.)

## Source
Exhaustive enumeration: buster_ripper/server.py:21,66,72; backend.py:54; compaction.py:36; routes/chat.py:37,125; routes/messages.py:127,130,228,231; routes/passthrough.py:25,29

## Source Version
Working-tree source of the buster-ripper repo (git metadata not consulted)

## Source Fingerprint
sha256:a9e49823ea456af9c0076781c4713c956745b21b026dfa768faf947e0b130592 (server.py); sha256:c2b71468af6b53efde753b0648d47eb6e06c8db4c738e08f7c01f4e4682d76ef (backend.py); sha256:ac9d42143c98694a0b6672dd753afcef537fc5e03ec747b4bb3c1208e23535e7 (compaction.py); sha256:16c7e99938b55e508b36c16092e7a89883baeff293385c3dfa3e9c161b413cae (routes/chat.py); sha256:721a2afe4d1ca35da9461c95f5c6dfe48dcf04ec4c4c0a606763cc29b0ca96da (routes/messages.py); sha256:072a58096586fcb3cf94c3cde37357bcee22abdffd8d89ae5952cfc4437494b2 (routes/passthrough.py)

## Invalidates On
Any new outbound HTTP call site under `buster_ripper/` that uses something other than the session stored as `app["session"]`
Introduction of `aiohttp.request(`, `httpx`, `requests`, `urllib`, `http.client`, a second `ClientSession`, or a websocket connection anywhere in the package
A call site switching to a locally-constructed client instead of `request.app["session"]` / the session threaded from it

## Checked At
2026-09-28
