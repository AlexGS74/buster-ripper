VERDICT: HOLDS
PROMOTION: ELIGIBLE

## Claim
`handle_messages` dispatches to `handle_streaming` or `handle_non_streaming` according to `body.get("stream", False)`

## What it actually does
Immediately after the dump/header setup, `handle_messages` binds
`stream = body.get("stream", False)` and branches on it: truthy returns
`await handle_streaming(request, body, headers, sid)`, falsy returns
`await handle_non_streaming(request, body, headers, sid)`. That assignment is the
only place the stream decision is read, and both callees are defined in the same
module (`handle_streaming` at line 74, `handle_non_streaming` at line 212).

Source form (`buster_ripper/routes/messages.py` lines 66–71):

```python
stream = body.get("stream", False)

if stream:
    return await handle_streaming(request, body, headers, sid)
else:
    return await handle_non_streaming(request, body, headers, sid)
```

Two details that do not contradict the claim as worded: the `body` being read is the
**normalized** body (line 36 rebinds `body` from `normalize(body, session_id="")`),
so anything normalization does to a `stream` key is what the branch sees; and the
branch tests truthiness, so a present-but-falsy `stream` (`false`, `0`, `""`, `null`)
takes the non-streaming arm, while a missing key takes it via the `False` default.

## Source
/Users/alexhubitski/Projects/buster-ripper/buster_ripper/routes/messages.py —
`handle_messages`, lines 66–71 (callees at lines 74 and 212; normalization rebind at line 36)

## Source Version
working-tree file content, no revision string consulted (repository state is out of
scope for this role); fingerprint below identifies the exact bytes

## Source Fingerprint
sha256:560a81be69d42552204400a43e2fc5ddec9a983a0b67f165fbe8d7ffee713709

## Invalidates On
- the `stream` value is read from somewhere other than `body.get("stream", False)` (e.g. from `request.headers`, a content-type sniff, or a different default)
- the `if stream:` branch is replaced by a dispatch table, a config flag, or backend-based routing that no longer selects between these two functions
- either callee is renamed/removed, or one arm gains an intermediate wrapper so `handle_messages` no longer returns that function's result directly
- `normalize()` starts rewriting or dropping the `stream` key in a way that makes the read site no longer the effective decision point

## Checked At
2026-10-06
