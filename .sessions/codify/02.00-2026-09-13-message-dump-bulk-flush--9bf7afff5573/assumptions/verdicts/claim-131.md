VERDICT: HOLDS
PROMOTION: ELIGIBLE

## Claim
`handle_messages` maps client `json.JSONDecodeError` from `request.json()` to a JSON 400 response

## What it actually does
The `await request.json()` call sits in a `try` whose only handler is
`except (json.JSONDecodeError, UnicodeDecodeError)`, and that handler returns
`web.json_response({"error": "Invalid JSON"}, status=400)` — a JSON-bodied response
with status 400. `json.JSONDecodeError` is the first name in the tuple (module `json`
is imported at line 3, so this is the stdlib exception, not a re-exported stand-in),
so any `JSONDecodeError` escaping that awaited call becomes a 400 instead of
propagating to aiohttp's default 500.

Source form (`buster_ripper/routes/messages.py` lines 28–31):

```python
try:
    body = await request.json()
except (json.JSONDecodeError, UnicodeDecodeError):
    return web.json_response({"error": "Invalid JSON"}, status=400)
```

Scope note: the verdict is about the handler's mapping, which the source establishes
directly. Whether aiohttp's `request.json()` raises `json.JSONDecodeError` (as
opposed to an aiohttp-specific subclass or a `ContentTypeError`) is a library-side
question and is **not** checked here — no dependency source root or exact
documentation locator was supplied for this bundle. The mapping holds regardless of
that answer, because the except clause names the type unconditionally.

## Source
/Users/alexhubitski/Projects/buster-ripper/buster_ripper/routes/messages.py —
`handle_messages`, lines 28–31 (with `import json` at line 3)

## Source Version
working-tree file content, no revision string consulted (repository state is out of
scope for this role); fingerprint below identifies the exact bytes

## Source Fingerprint
sha256:560a81be69d42552204400a43e2fc5ddec9a983a0b67f165fbe8d7ffee713709

## Invalidates On
- `json.JSONDecodeError` is dropped from the except tuple, or the tuple is narrowed to `UnicodeDecodeError` alone
- the handler's return changes status away from 400, or changes from `web.json_response` to a non-JSON response type
- the `request.json()` read moves out of this `try` block into another function or an unguarded call site
- `import json` is removed or aliased so the name in the except tuple no longer resolves to stdlib `json.JSONDecodeError`

## Checked At
2026-10-06
