VERDICT: HOLDS
PROMOTION: ELIGIBLE

## Claim
`_stream_single_request`'s `try` block covers both its `response.prepare()` call and its chunk relay loop

## What it actually does
The single `try:` in `_stream_single_request` opens at line 129 and its only handler is `except aiohttp.ClientError` at line 179, so every statement between them is inside the guarded region — including `await response.prepare(request)` (line 154) and the whole relay loop `async for chunk in resp.content.iter_any():` with its `await response.write(chunk)` body (lines 156-174), plus the trailing `await response.write_eof()` / `return response` (lines 176-177). A raise from either the prepare call or any write inside the loop is therefore routed to the same `ClientError` arm. (The upstream error short-circuit at lines 136-142 and the `async with session.post(...)` context manager at line 130 are likewise inside the same `try`.)

## Source
/Users/alexhubitski/Projects/buster-ripper/buster_ripper/routes/messages.py — `_stream_single_request`, lines 121-181 (`try:` at 129; `response.prepare` at 154; relay loop 156-174; `except aiohttp.ClientError as e:` at 179)

## Source Version
local repository source, no version/revision string consulted (worktree file content only)

## Source Fingerprint
sha256:721a2afe4d1ca35da9461c95f5c6dfe48dcf04ec4c4c0a606763cc29b0ca96da

## Invalidates On
- Any edit that moves `await response.prepare(request)` out of the `try` block at messages.py:129-181 (e.g. preparing the response before entering the `try`, or adding an inner `try` that swallows before the outer handler)
- Any edit that moves the `async for chunk in resp.content.iter_any()` relay loop (messages.py:156-174) into a separate function or outside the guarded region
- Any edit that narrows the guard by inserting an earlier `except`/`finally` or restructuring the block so line 179 is no longer the handler for that span
- A rewrite of `_stream_single_request` whose line span no longer matches the fingerprint above

## Checked At
2026-10-05
