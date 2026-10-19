VERDICT: HOLDS
PROMOTION: ELIGIBLE

## Claim
`logging_middleware` awaits the handler without catching exceptions

## What it actually does
`logging_middleware` logs the request line and then returns `await handler(request)`
directly, with no `try`/`except` anywhere in its body. Any exception raised by the
downstream handler (or by a nested middleware further in) propagates out of
`logging_middleware` unchanged — the middleware neither converts it to a response
nor swallows it.

Corroborating negative check over the whole subject file: the only `try:`/`except`
constructs in `buster_ripper/server.py` are at lines 100–106
(`_cancel_background_task`) and 111–114 (`_final_dump_flush`). Both are teardown
helpers and are outside the middleware (lines 48–52), so no handler exception is
caught by this function.

The `@web.middleware` decorator at line 48 does not add exception handling around
the body; the awaited-handler call is the final statement of the function as
written in source.

Source-verified body (lines 48–52, verbatim):

```python
@web.middleware
async def logging_middleware(request: web.Request, handler) -> web.Response:
    """Log all requests."""
    log.info("%s %s", request.method, request.path)
    return await handler(request)
```

## Source
/Users/alexhubitski/Projects/buster-ripper/buster_ripper/server.py lines 48–52
(whole 138-line file read to confirm the absence of exception handling in this
function)

## Source Version
Repository source revision unavailable — resolving VCS metadata is outside this
role's boundary. The consulted bytes are fixed by the fingerprint below.

## Source Fingerprint
sha256:a9e49823ea456af9c0076781c4713c956745b21b026dfa768faf947e0b130592

## Invalidates On
- Any edit to `logging_middleware` that wraps `await handler(request)` in
  `try`/`except` (including a bare `except Exception:` that returns a response),
  or that changes the sha256 of `buster_ripper/server.py`.
- Replacement of the `@web.middleware` decorator with a wrapper/decorator that
  itself traps handler exceptions before this function's body is entered.
- Move of `logging_middleware` out of `buster_ripper/server.py` into another
  module, leaving this file's copy stale.

## Checked At
2026-10-05
