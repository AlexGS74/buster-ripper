VERDICT: HOLDS
PROMOTION: ELIGIBLE

## Claim
`buster_ripper_legacy.py`'s `_poll_kv_cache` metrics request is issued with no `headers` argument

## What it actually does
Inside `_poll_kv_cache` (defined at line 207), the metrics scrape is a two-line block:

```python
async with httpx.AsyncClient(timeout=5) as client:
    resp = await client.get(f"{UPSTREAM}/metrics")
```

The `client.get(...)` call passes exactly one argument — the URL string — and no keyword
arguments at all. There is no `headers=` on the request call, and there is also no `headers=`
on the `httpx.AsyncClient(...)` constructor (its only keyword is `timeout=5`), so no default
header set is attached at the client level either. The function body spans lines 207-227 and
contains no other outbound call, so this is the only request the function issues.

Worth stating so the verdict is not over-read: "no `headers` argument" is a statement about
the call site, not about the wire. httpx still populates its own defaults on the request
(`host`, `accept`, `accept-encoding`, `connection`, `user-agent`). What the claim asserts —
that the caller supplies no `headers` argument — is true.

## Source
`/Users/alexhubitski/Projects/buster-ripper/buster_ripper_legacy.py` — production source
(the sole source root supplied for this check). Decisive lines: 207-227, request at 213-214.
Corroborating: a whole-file scan for `headers` occurrences returns none at lines 213 or 214.

## Source Version
local repo file, unversioned: 1013 lines, 42129 bytes (mtime 2026-03-03)

## Source Fingerprint
sha256:7967563dc3ab4f66f0dc322c7945c5647b96c9f9a77fbafe3cd1d4780e8764bd

## Invalidates On
- A `headers=` keyword added to `client.get(...)` in `_poll_kv_cache`, or to the enclosing `httpx.AsyncClient(timeout=5)` constructor
- The metrics scrape moving to another function or a shared client that carries default headers
- Any new outbound request added inside `_poll_kv_cache` that is not the `/metrics` call
- Byte change to `buster_ripper_legacy.py` such that the fingerprint above no longer matches

## Checked At
2026-10-05T21:07:42Z
