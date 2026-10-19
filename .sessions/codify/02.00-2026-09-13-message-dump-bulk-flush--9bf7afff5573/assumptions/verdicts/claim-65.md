VERDICT: DOES-NOT-HOLD
PROMOTION: ELIGIBLE

## Claim
`chat.py`'s `_make_chat_request` parses every upstream response with `await resp.json()` (`chat.py:131`) regardless of the response content type

## What it actually does
It does not parse *every* upstream response. `_make_chat_request` branches before the parse:

```python
if body.get("stream"):
    return await _relay_chat_stream(request, resp)   # chat.py:150-151
resp_data = await resp.json()                        # chat.py:152
```

When the request body asks for streaming, the function returns inside the `async with` block and the
upstream response is never JSON-parsed at all: `_relay_chat_stream` (chat.py:174-188) relays raw bytes
via `await resp.read()` (chat.py:177) and `resp.content.iter_any()` (chat.py:185). Its own docstring
states the intent — "Relay SSE chunks without decoding or applying JSON-only transforms". So `await
resp.json()` occurs exactly once in this file, on the non-stream branch.

The other half of the claim is accurate and worth keeping: on that non-stream branch the route calls
`resp.json()` with no content-type check of its own — it never reads `resp.headers["Content-Type"]`
before deciding to parse, and passes no `content_type=` argument. Any content-type gating that exists
is inside aiohttp's `ClientResponse.json()`, not in this route; that library behaviour is not settled
here because the bounded source surface for this bundle was `chat.py` alone (no installed aiohttp root
was supplied), and per protocol I did not go looking for one.

Because the row is a conjunction, one failing conjunct fails the row: the parse is unconditional *on
content type*, but conditional *on `body["stream"]`*.

Anchor drift: the claim cites `chat.py:131`, which in the consulted bytes is a blank line between
`_carries_token` and `_make_chat_request`. The actual `await resp.json()` is chat.py:152.

## Source
/Users/alexhubitski/Projects/buster-ripper/buster_ripper/routes/chat.py — chat.py:143-161 (`_make_chat_request`), chat.py:174-188 (`_relay_chat_stream`); sole occurrence of `resp.json` confirmed by `grep -n "resp.json" chat.py` → line 152 only

## Source Version
Repo working source, 299 lines; no VCS revision consulted (Git metadata is out of scope for this role)

## Source Fingerprint
sha256:8da1d171930433fcec7f3657e3b668026dc78fe3657a6525f3df3c78c46388c5

## Invalidates On
- Any edit that removes or reorders the `if body.get("stream"): return await _relay_chat_stream(...)` guard, or moves `await resp.json()` above it — that would make "every upstream response" true
- Any change that makes the stream decision depend on something other than `body.get("stream")` (e.g. an upstream content-type sniff), which changes what "every" ranges over
- Any added content-type inspection before `resp.json()`, which would fail the surviving half of this claim
- Replacing/upgrading aiohttp in a way that changes whether `ClientResponse.json()` enforces content type by default, which changes how much the "regardless of content type" half describes observable behaviour rather than just route code

## Checked At
2026-10-06T12:16:35Z
