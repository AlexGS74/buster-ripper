VERDICT: UNCLEAR
PROMOTION: INELIGIBLE — verdict is UNCLEAR; the claim hinges on two terms ("typed", and whether the cited line ranges are assertions or pointers) that resolve to HOLDS under one reading and DOES-NOT-HOLD under the other.

## Claim
`messages.py:28-29` and `chat.py:30-31` return a typed 400 (`{"error": "Invalid JSON"}`) when the client body fails JSON parsing

## What it actually does
What the two handlers do is unambiguous, and it matches the claim's substance. Both files have byte-identical parse boundaries:

- `messages.py` — `28: try:` / `29: body = await request.json()` / `30: except (json.JSONDecodeError, UnicodeDecodeError):` / `31: return web.json_response({"error": "Invalid JSON"}, status=400)`
- `chat.py` — `28: try:` / `29: body = await request.json()` / `30: except (json.JSONDecodeError, UnicodeDecodeError):` / `31: return web.json_response({"error": "Invalid JSON"}, status=400)`

So the 400 with body `{"error": "Invalid JSON"}` is returned at **line 31 in both files**, and the `except` that catches the parse failure is at **line 30 in both files**. The claim's ranges are `28-29` for messages and `30-31` for chat — two different slices of the same four-line block. They are one-sided: `30-31` is the except+return (the returning site, correct for chat), `28-29` is the try+parse (the *failing* site, not the returning site). No single consistent reading makes both ranges point at the return. Per P3 I will not answer the behavioural half while certifying the locational half, and per P9 I will not call the claim false over a slice that reads as a pointer.

"typed 400" is the second open term. Under "a 400 whose body is a structured JSON error document" → HOLDS: `web.json_response({"error": "Invalid JSON"}, status=400)`, and the handlers are type-annotated (`handle_messages -> web.StreamResponse`, `handle_chat -> web.Response`). Under "typed" as aiohttp's *typed error* — an `HTTPException` subclass raised rather than a status argument passed → DOES-NOT-HOLD in both files: neither raises anything here; the only `web.HTTPBadRequest` under the source root is `count_tokens.py:23`, and there it covers `UnicodeDecodeError` only (`json.JSONDecodeError` returns `{}` at `count_tokens.py:20-21`). I had to choose which meaning makes the claim false, which is the P9 test — so this is UNCLEAR, not DOES-NOT-HOLD.

Settled by re-extraction that (a) names one consistent locator per file — `messages.py:30-31` and `chat.py:30-31`, or "the `except` at line 30" in both — and (b) states what "typed" means (JSON-body error response vs. raised `web.HTTPBadRequest`). Both readings of "when the client body fails JSON parsing" agree the trigger is `json.JSONDecodeError` (plus `UnicodeDecodeError`), so the trigger condition itself is not in question.

## Source
`/Users/alexhubitski/Projects/buster-ripper/buster_ripper/routes/messages.py:28-31` and `/Users/alexhubitski/Projects/buster-ripper/buster_ripper/routes/chat.py:28-31`; corroborating negative search for `HTTPBadRequest`/`json_response` across the supplied `Source root` (only other hit: `count_tokens.py:23-24`).

## Source Version
Local worktree sources (unversioned checkout), Python 3.13, aiohttp `web` used as the response layer; aiohttp version not stated in the authorized surface.

## Source Fingerprint
- `messages.py` — `sha256:560a81be69d42552204400a43e2fc5ddec9a983a0b67f165fbe8d7ffee713709`
- `chat.py` — `sha256:8da1d171930433fcec7f3657e3b668026dc78fe3657a6525f3df3c78c46388c5`
- `count_tokens.py` — `sha256:a69c9884cad5997e442a4eaa5c5557f13a589b227777b71fbe1c4167c38fa203`

## Invalidates On
- Either handler's `except` clause at line 30 changing (narrowed to drop `UnicodeDecodeError`, or replaced by `except Exception`).
- Either line-31 return changing status, body key/value, or switching from `web.json_response(...)` to a raised `web.HTTPBadRequest` / `HTTPClientError`.
- Lines shifting so that `28-29` / `30-31` no longer straddle the try/except/return block in these files (any edit above line 28 in either file).
- `count_tokens.py:23` starting to raise `HTTPBadRequest` for `json.JSONDecodeError` too, which would remove the contrast that makes "typed" meaningful.

## Checked At
2026-10-06T12:16:38Z
