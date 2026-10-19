VERDICT: HOLDS
PROMOTION: ELIGIBLE

## Claim
The only `except` clause in `_stream_single_request` catches `aiohttp.ClientError`

## What it actually does
`_stream_single_request` occupies lines 121-181 and contains exactly one `except` clause: line 179, `except aiohttp.ClientError as e:`. It is a single-class catch (not a tuple), has no `else` arm, and there is no `except Exception`, no bare `except`, and no nested `try` anywhere in the function. The other `except` lines in the file all belong to different functions — 28 and 58 are in `handle_messages`, 204 and 266 are in the non-streaming path — so within the named function the claim is exhaustive and true. Practical consequence visible in the source: exceptions outside `aiohttp.ClientError` (e.g. `asyncio.TimeoutError`/`TimeoutError`, which is not a `ClientError` subclass, or a `TypeError` from the capture logic at lines 161-166) are not caught here and propagate to the caller.

## Source
/Users/alexhubitski/Projects/buster-ripper/buster_ripper/routes/messages.py — `_stream_single_request`, lines 121-181; sole handler at line 179. Confirmed exhaustive by a file-wide scan of `except` at lines 28, 58, 179, 204, 266 and `def` boundaries at lines 24, 73, 121, 184, 223.

## Source Version
local repository source, no version/revision string consulted (worktree file content only)

## Source Fingerprint
sha256:721a2afe4d1ca35da9461c95f5c6dfe48dcf04ec4c4c0a606763cc29b0ca96da

## Invalidates On
- Adding any further `except` arm (or a bare/`Exception` catch, or an `except (aiohttp.ClientError, X)` tuple) to `_stream_single_request` at messages.py:121-181
- Splitting the function so error handling moves into a helper or an outer caller while line 179 remains
- Renaming the caught class (e.g. catching `ClientOSError`/`ServerTimeoutError` instead of `aiohttp.ClientError`)
- Any edit to the file that changes the fingerprint above, since the exhaustiveness claim is over the whole function span

## Checked At
2026-10-05
