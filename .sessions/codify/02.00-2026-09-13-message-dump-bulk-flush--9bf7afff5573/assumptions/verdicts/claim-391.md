VERDICT: UNCLEAR
PROMOTION: INELIGIBLE — verdict is not HOLDS/DOES-NOT-HOLD; the ambiguity is in the claim's wording, not the source

## Claim
aiohttp's `web.Request` enforces its body-size limit from the value configured on the application serving the request

## What it actually does
The mechanism is settled: `Request` carries its own `self._client_max_size`
(`web_request.py:198`), set from the keyword-only `client_max_size` argument of
`BaseRequest.__init__` (`web_request.py:161`, default `1024**2`), and the
enforcement sites read only that instance attribute — `read()` raises
`HTTPRequestEntityTooLarge` when `body_size >= self._client_max_size`
(`web_request.py:675-680`) and `post()` uses `max_size = self._client_max_size`
for multipart (`web_request.py:722`, checks at 746 and 774). The value is injected
by the application that constructs the request: `Application._make_request` passes
`client_max_size=self._client_max_size` (`web_app.py:510`), and that method is
bound as the server's request factory by `Application._make_handler`
(`web_app.py:444`). So the limit does come from application configuration rather
than being hardcoded or global.

What is not settled is which application "the application serving the request"
names, and the two available readings give different answers:

- **Reading A — the application whose runner/handler serves the connection** (the
  app passed to `AppRunner`/`run_app`, i.e. the one that built the `Request`
  object via its `_make_request`): the claim **HOLDS**. The request's limit is
  exactly that app's `client_max_size`.
- **Reading B — the application `request.app` reports** (the innermost matched
  application, `match_info.current_app`, `web_request.py:891-896`): the claim
  **DOES NOT HOLD** for nested applications. `add_subapp` gives the sub-app its own
  `_client_max_size`, but routing happens in `Application._handle` *after*
  `_make_request` has already built the request, and `_handle` never rewrites
  `_client_max_size` (`web_app.py:527-569`). A sub-application's configured
  `client_max_size` is therefore inert for requests routed into it. For a flat,
  single application the two readings coincide, so the claim holds either way.

I had to pick between these meanings — which per P9 makes the word the finding,
not the behaviour. Do not read this as "the library might do something else": the
propagation app→request is real and unambiguous.

## Source
/Users/alexhubitski/Projects/buster-ripper/.venv/lib/python3.13/site-packages/aiohttp/web_request.py
(lines 161, 198, 285-286, 675-680, 722, 891-896) and
/Users/alexhubitski/Projects/buster-ripper/.venv/lib/python3.13/site-packages/aiohttp/web_app.py
(lines 444, 503-511, 527-569). Installed package ships no docs; verdict is grounded
in the installed source only.

## Source Version
aiohttp 3.13.3 (`aiohttp/__init__.py` line 1)

## Source Fingerprint
sha256:2aaacea7a01e581e5eead2ab1b9e4ba3b65bc2ae3d0f176b2a78ae5b6b76bb4e (web_request.py, full file bytes)
sha256:94653f68030dfa1df0cbe2d34c78ba49e287f3276dd46e75057702b29803e590 (web_app.py, full file bytes)

## Invalidates On
- Rewording the claim to name one application unambiguously settles it immediately
  as HOLDS (root/serving app) or DOES-NOT-HOLD (sub-app) against the source above —
  this is the repair, and it is an extraction fix, not a source gap
- A future aiohttp that applies the matched sub-application's `client_max_size` to
  the request, which would make both readings true
- Any file whose sha256 differs from the fingerprints above

## Checked At
2026-10-06T22:21:04Z
