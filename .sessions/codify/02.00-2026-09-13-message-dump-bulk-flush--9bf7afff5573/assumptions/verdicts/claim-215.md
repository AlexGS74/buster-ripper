VERDICT: HOLDS
PROMOTION: ELIGIBLE

## Claim
Every response-side header filter under `buster_ripper/routes/` drops `content-length`

## What it actually does
The only response-side header filter reached from `routes/` is `utils.response_headers`, called at `messages.py:285` (non-streaming `/v1/messages`) and `passthrough.py:41` (catch-all). Its drop tuple is `("content-length", *_HOP_BY_HOP)` against `k.lower()`, so `content-length` is always removed from the relayed upstream headers — the docstring states the reason explicitly (aiohttp re-derives the length from the body actually sent, and `Response._start` only derives one when the header is absent). No other response path under `routes/` relays upstream headers at all: `chat.py:178-183`, `chat.py:207-211`/`216-219`, `messages.py:138-153` and `count_tokens.py:85` each construct a fresh header set (Content-Type / Cache-Control / X-Accel-Buffering / Connection) rather than filtering the upstream one, so none of them can pass an upstream `content-length` either.

The claim therefore holds under both readings of its scope: with "filter under `routes/`" read as *defined in files under `routes/`* it is vacuously true (no response-side filter is defined there), and read as *applied by `routes/` handlers* it is substantively true at both relay sites.

## Source
- `buster_ripper/utils.py:27-40` — `response_headers` drop tuple at line 39 and the explanatory docstring at lines 30-34
- `buster_ripper/routes/messages.py:285-290`, `buster_ripper/routes/passthrough.py:41-47` — the two response-side call sites
- `buster_ripper/routes/chat.py:174-220`, `buster_ripper/routes/messages.py:136-153`, `buster_ripper/routes/count_tokens.py:85` — exhaustive check of the remaining response-header constructions (no filtering, no upstream pass-through)

Note on the source boundary: the claim's subject is `buster_ripper/routes/` (the supplied source root, all four modules read in full); `buster_ripper/utils.py` was read because `response_headers` — the filter those routes call — is defined there, so the drop list itself lives one level below the stated root.

## Source Version
unavailable — git metadata is out of scope for this role; consulted bytes are fingerprinted instead.

## Source Fingerprint
- `buster_ripper/utils.py` — `sha256:76de98c1bd07e1cf3b2852d8dd64dd861160f54fe40d2781fce1ad66ae913c72`
- `buster_ripper/routes/messages.py` — `sha256:560a81be69d42552204400a43e2fc5ddec9a983a0b67f165fbe8d7ffee713709`
- `buster_ripper/routes/passthrough.py` — `sha256:c4eef101710b057f0efcf7e99cbe86c48d84f88db9610ec60a17a5fab1b54bbb`
- `buster_ripper/routes/chat.py` — `sha256:8da1d171930433fcec7f3657e3b668026dc78fe3657a6525f3df3c78c46388c5`
- `buster_ripper/routes/count_tokens.py` — `sha256:a69c9884cad5997e442a4eaa5c5557f13a589b227777b71fbe1c4167c38fa203`

## Invalidates On
- `content-length` removed from the drop tuple at `utils.py:39`, or `response_headers` given a parameter/mode that preserves it
- A new response-side filter introduced under `routes/` that relays upstream headers without going through `response_headers` (e.g. a streaming handler copying `resp.headers` into `StreamResponse`)
- Either call site (`messages.py:285`, `passthrough.py:41`) replaced by direct `resp.headers` forwarding
- `buster_ripper/utils.py` or any file under `buster_ripper/routes/` changing from the fingerprinted bytes above

## Checked At
2026-10-06T12:20:51Z
