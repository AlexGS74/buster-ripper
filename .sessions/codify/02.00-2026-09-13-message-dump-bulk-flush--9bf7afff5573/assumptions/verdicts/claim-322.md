VERDICT: HOLDS
PROMOTION: ELIGIBLE

## Claim
Every upstream request in `buster_ripper_legacy.py` that carries a client-derived header set builds that header set through `_forward_headers`

## What it actually does
The file has exactly five outbound upstream call sites, and every one of them that carries
request headers derived from the inbound client request gets those headers from
`_forward_headers(request)` — there is no second, ad-hoc route for assembling client headers.
`_forward_headers` is defined once (line 621) and called at exactly three sites (lines 656,
812, 901), and those three calls cover all four header-carrying upstream requests.

Complete enumeration of outbound upstream requests (from the source, not from tests):

| Line | Caller | Upstream call | `headers=` source |
|---|---|---|---|
| 214 | `_poll_kv_cache` | `client.get(f"{UPSTREAM}/metrics")` | none — no `headers` argument, client-derived or otherwise |
| 665 | `proxy_messages` (stream branch) | `client.stream("POST", f"{UPSTREAM}/v1/messages", ...)` | `headers` from line 656 `= _forward_headers(request)` |
| 701 | `proxy_messages` (non-stream branch) | `client.post(f"{UPSTREAM}/v1/messages", ...)` | same `headers` variable from line 656 |
| 844 | `chat_completions` | `client.post(f"{UPSTREAM}/v1/chat/completions", ...)` | `headers` from line 812 `= _forward_headers(request)` |
| 903 | `passthrough` | `client.request(method=..., url=f"{UPSTREAM}/{path}", ...)` | `headers` from line 901 `= _forward_headers(request)` |

`count_tokens` (line 726) issues no upstream request at all — it returns a synthesized
`Response` and the docstring states forwarding is pointless because vLLM 404s the endpoint.

Two non-request header helpers do not weaken the claim: line 696's literal
`{"Cache-Control": ..., "X-Accel-Buffering": ...}` is the downstream `StreamingResponse`
headers toward the client, and `_response_headers` (line 630, used at 718/882/891/913)
filters the *upstream response* headers on the way back to the client. Neither is an
upstream request header set.

## Caveat on the one impure case (recorded for re-check)
`chat_completions` builds its set through `_forward_headers` at line 812 and then edits two
entries in place before sending: `headers["content-length"] = str(len(body))` (line 834) and
`headers.pop("authorization", None)` when the value is an empty Bearer (lines 839-841). So
the set carried on line 844 is `_forward_headers`'s result plus/minus those two keys. The
claim's predicate is about *which function builds the set*, and that is `_forward_headers`
in all four cases — no request constructs its client headers by any other route. Read as
"constructed exclusively by `_forward_headers` with no further modification," this line would
fail; the wording used here ("builds ... through") is the construction-path reading, and the
impurity above is the full extent of what a stricter reading would turn on.

## Source
`/Users/alexhubitski/Projects/buster-ripper/buster_ripper_legacy.py` — production source
(the sole source root supplied for this check). Decisive lines: 214, 621-627, 656, 665-669,
701-704, 812, 834, 839-841, 843-847, 898-909.

## Source Version
local repo file, unversioned: 1013 lines, 42129 bytes (mtime 2026-03-03)

## Source Fingerprint
sha256:7967563dc3ab4f66f0dc322c7945c5647b96c9f9a77fbafe3cd1d4780e8764bd

## Invalidates On
- Any new outbound `httpx` call site in this file that passes a `headers=` value not traced to `_forward_headers(request)`
- Removal or renaming of `_forward_headers`, or an added second header-building helper used on an upstream request path
- `count_tokens` (or any currently-local-only endpoint) gaining an upstream forward that carries inbound client headers
- Byte change to `buster_ripper_legacy.py` such that the fingerprint above no longer matches

## Checked At
2026-10-05T21:07:42Z
