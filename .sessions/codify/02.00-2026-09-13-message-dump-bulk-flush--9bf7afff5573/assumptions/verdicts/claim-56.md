VERDICT: HOLDS
PROMOTION: ELIGIBLE

## Claim
`chat.py:146` returns a typed 502 ("Invalid JSON from upstream") when the upstream response fails JSON parsing — the upstream-attributed arm

## What it actually does
`_make_chat_request` wraps the upstream call in `try/except`, and the handler
`except (json.JSONDecodeError, UnicodeDecodeError): return web.json_response({"error": "Invalid JSON from upstream"}, status=502)`
exists in production source (chat.py:158-159). The only statements inside that `try` that can raise
`JSONDecodeError`/`UnicodeDecodeError` are on the upstream-response path — `resp_data = await resp.json()`
(chat.py:152), which decodes the upstream body to text and then JSON-parses it — so a malformed or
undecodable upstream response body does land in this arm and produces a 502 whose body attributes the
fault to upstream. It is the upstream-attributed arm of exactly two "Invalid JSON" arms in this route:
the other is the client-attributed one, `return web.json_response({"error": "Invalid JSON"}, status=400)`
(chat.py:31), which covers the *request* body failing `await request.json()`. The claim's distinction is
real and matches the source.

Two things this arm does NOT cover, stated so the hold is not over-read:
- A response whose `Content-Type` is not JSON raises `aiohttp.ContentTypeError`, which is an
  `aiohttp.ClientError` and so is caught by the earlier handler (chat.py:155-157) — also a 502, but
  with `{"error": str(e)}`, not "Invalid JSON from upstream". Whether that is true depends on aiohttp's
  own `resp.json()` content-type enforcement, which is library behaviour outside the bounded source
  surface supplied for this bundle; the route's source alone settles only that this route does not
  inspect content type before calling `resp.json()`.
- `_apply_response_transforms` (chat.py:161) runs *after* the `try`, so a failure there never reaches
  this arm.

Anchor drift: the claim cites `chat.py:146`, which in the consulted bytes is `json=body,` inside
`session.post(...)`. The asserted return is at chat.py:159. The message string "Invalid JSON from
upstream" occurs exactly once in the file, so the claim was answerable by its named subject; the line
number is stale, not the assertion.

## Source
/Users/alexhubitski/Projects/buster-ripper/buster_ripper/routes/chat.py — chat.py:143-161 (`_make_chat_request` try/except), with chat.py:28-31 (the contrasting 400 arm)

## Source Version
Repo working source, 299 lines; no VCS revision consulted (Git metadata is out of scope for this role)

## Source Fingerprint
sha256:8da1d171930433fcec7f3657e3b668026dc78fe3657a6525f3df3c78c46388c5

## Invalidates On
- Any edit to the `except (json.JSONDecodeError, UnicodeDecodeError)` handler in `_make_chat_request` (status, error string, or the exception tuple)
- Any edit that moves `await resp.json()` or `_apply_response_transforms` across the `try` boundary, or adds another JSON-decoding statement inside the `try`
- Replacing aiohttp (or upgrading it such that `ClientResponse.json()` stops raising `json.JSONDecodeError` for malformed bodies), since the arm's reachability depends on which exception type a bad upstream body produces
- A second "Invalid JSON from upstream"-shaped arm being added elsewhere in the route, which would void the "the upstream-attributed arm" uniqueness reading

## Checked At
2026-10-06T12:16:35Z
