VERDICT: HOLDS
PROMOTION: ELIGIBLE

## Claim
`handle_messages` maps client `UnicodeDecodeError` from `request.json()` to a JSON 400 response

## What it actually does
The `await request.json()` call sits in a `try` whose only handler is
`except (json.JSONDecodeError, UnicodeDecodeError)`, and that handler returns
`web.json_response({"error": "Invalid JSON"}, status=400)` — a JSON-bodied response
with status 400. `UnicodeDecodeError` is named explicitly in the tuple, so any
`UnicodeDecodeError` escaping that awaited call is converted to a 400 rather than
propagating to aiohttp's default 500.

Source form (`buster_ripper/routes/messages.py` lines 28–31):

```python
try:
    body = await request.json()
except (json.JSONDecodeError, UnicodeDecodeError):
    return web.json_response({"error": "Invalid JSON"}, status=400)
```

Scope note: the verdict is about the handler's mapping, which the source establishes
directly. Whether aiohttp's `request.json()` actually raises `UnicodeDecodeError` on
some malformed-payload class is a library-side question and is **not** checked here —
no dependency source root or exact documentation locator was supplied for this bundle.
The mapping holds regardless of that answer, because the except clause names the type
unconditionally.

## Source
/Users/alexhubitski/Projects/buster-ripper/buster_ripper/routes/messages.py —
`handle_messages`, lines 28–31

## Source Version
working-tree file content, no revision string consulted (repository state is out of
scope for this role); fingerprint below identifies the exact bytes

## Source Fingerprint
sha256:560a81be69d42552204400a43e2fc5ddec9a983a0b67f165fbe8d7ffee713709

## Invalidates On
- `UnicodeDecodeError` is dropped from the except tuple, or the tuple becomes a bare `json.JSONDecodeError`
- the handler's return changes status away from 400, or changes from `web.json_response` to a non-JSON response type (plain text, `web.Response` without a JSON content type)
- the `request.json()` read moves out of this `try` block into another function or an unguarded call site
- a body-decoding path is added ahead of this block (e.g. manual `await request.read()` plus explicit decode) so `UnicodeDecodeError` can no longer reach this handler

## Checked At
2026-10-06
