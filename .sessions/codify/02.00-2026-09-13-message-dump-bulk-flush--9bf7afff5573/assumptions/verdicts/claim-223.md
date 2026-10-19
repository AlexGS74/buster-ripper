VERDICT: DOES-NOT-HOLD
PROMOTION: ELIGIBLE

## Claim
Every upstream request in `buster_ripper_legacy.py` builds its header set through `_forward_headers`

## What it actually does
It builds four of the file's five upstream requests. `_forward_headers` (`buster_ripper_legacy.py:621-627`)
supplies the header dict for every request a route handler sends upstream: `proxy_messages` streaming
(`:656` → `:669`), `proxy_messages` non-streaming (`:656` → `:704`), `chat_completions` (`:812` → `:847`),
and `passthrough` (`:901` → `:907`). The fifth is not a forwarded request at all — `_poll_kv_cache` opens
`httpx.AsyncClient(timeout=5)` and calls `client.get(f"{UPSTREAM}/metrics")` (`:213-214`) with **no
`headers=` argument**, so that request's header set is httpx's own defaults and never passes through
`_forward_headers` (and therefore gets no `host`/`content-length`/hop-by-hop filtering). Enumeration of
client-construction sites is complete: `grep -n "AsyncClient" buster_ripper_legacy.py` returns exactly
`:213, :664, :700, :843, :902`, and httpx is the only HTTP client the module imports (`:84`).
`count_tokens` (`:725`) makes no upstream request — `:757-758` states forwarding is pointless because vLLM
returns 404 — so it is not a counterexample, and the other `headers=` literals (`:696`, `:884`) are
outbound *responses* to the client, not requests to upstream.

Reading note (transparency, not a hedge): under a narrow reading of "upstream request" as "an inbound
client request forwarded upstream by a route handler", the claim holds. The verdict rests on the plain
reading — any request this module sends to `UPSTREAM` — which the file's own vocabulary supports, since
the poll's failure path is annotated `# upstream may be starting up` (`:226`) and its URL is built from
the `UPSTREAM` constant itself.

## Source
`buster_ripper_legacy.py` — falsifying site `:213-214` (`_poll_kv_cache`); supporting sites `:621-627`,
`:656`, `:669`, `:704`, `:812`, `:847`, `:901`, `:907`; whole file (1013 lines) read, production source
(claim subject is implementation, not tests)

## Source Version
Repo working tree file `buster_ripper_legacy.py`, 42129 bytes, mtime 2026-03-03 09:50:07; standalone
script header (`requires-python >=3.11`, deps fastapi/uvicorn/httpx/click), no package version string

## Source Fingerprint
sha256:7967563dc3ab4f66f0dc322c7945c5647b96c9f9a77fbafe3cd1d4780e8764bd

## Invalidates On
- Any edit to `buster_ripper_legacy.py` (fingerprint changes) — re-check before reuse
- `_poll_kv_cache` gains a `headers=` argument at `:214`, whether from `_forward_headers` or built inline
- A new `httpx.AsyncClient` request site is added to the module (the five-site enumeration becomes stale)
- `_forward_headers` is removed/renamed, or a route handler starts passing headers it did not build there

## Checked At
2026-10-05T18:18:03Z
