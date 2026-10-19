VERDICT: HOLDS
PROMOTION: ELIGIBLE

## Claim
`_stream_single_request`'s `ClientError` arm returns a newly constructed `web.json_response`

## What it actually does
The body of the `except aiohttp.ClientError as e:` arm (lines 179-181) is two statements: log the error, then `return web.json_response({"error": str(e)}, status=502)` at line 181. That call is written inline in the `return`, so a fresh response object is constructed by `web.json_response` on every exception — the arm does not re-raise, does not fall through, and does not return the `web.StreamResponse` built at lines 145-153 or the `web.Response` built at lines 138-142 (those are separate constructions on other paths). `web.json_response` is aiohttp's response factory, bound here through `from aiohttp import web` at line 7; the payload is a single-key dict `{"error": str(e)}` and the status is hardcoded 502.

## Source
/Users/alexhubitski/Projects/buster-ripper/buster_ripper/routes/messages.py — line 181 inside the `except aiohttp.ClientError` arm at line 179; import binding `from aiohttp import web` at line 7

## Source Version
local repository source, no version/revision string consulted (worktree file content only); aiohttp package version not consulted (outside supplied source root)

## Source Fingerprint
sha256:721a2afe4d1ca35da9461c95f5c6dfe48dcf04ec4c4c0a606763cc29b0ca96da

## Invalidates On
- Changing line 181 to return a pre-existing object (e.g. `return response`), a plain `web.Response(...)`, `web.text_response(...)`, or to `raise` instead of returning
- Changing the constructor from `web.json_response` to any other factory, or rebinding `web` so `json_response` is no longer aiohttp's response factory
- Any edit that alters the payload/status semantics claimed here (single `error` key, status 502), e.g. adding fields or deriving status from the exception
- Bumping or replacing the `aiohttp` dependency in a way that changes what `web.json_response` constructs (requires re-checking against that package's source, which is not part of this verdict's fingerprint)

## Checked At
2026-10-05
