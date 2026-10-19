VERDICT: HOLDS
PROMOTION: ELIGIBLE

## Claim
An aiohttp `web.Application` constructed without `client_max_size` limits a request body to 1024**2 bytes

## What it actually does
The parameter default in `Application.__init__` is `client_max_size: int =
1024**2`, so an app built without the argument stores `self._client_max_size ==
1048576`. `Application._make_request` always forwards it
(`client_max_size=self._client_max_size`), so every request that app serves
inherits 1 MiB and `BaseRequest.read()` raises `HTTPRequestEntityTooLarge` once the
accumulated body reaches it. Two precision notes, neither of which contradicts the
claim: the comparison is `body_size >= self._client_max_size`, so a body of exactly
1048576 bytes is already rejected (largest accepted body is 1048575 bytes; the 413
still reports `max_size=1048576`); and the limit is enforced by the buffered body
APIs — `read()`/`text()`/`json()` via `read()`, and `post()` via
`max_size = self._client_max_size` — while direct reads of `request.content` are
not bounded by it. `0` disables the check entirely (`if self._client_max_size:`),
which is not the default.

## Source
/Users/alexhubitski/Projects/buster-ripper/.venv/lib/python3.13/site-packages/aiohttp/web_app.py
— `client_max_size: int = 1024**2` at line 128, stored at line 175, forwarded at
line 510;
/Users/alexhubitski/Projects/buster-ripper/.venv/lib/python3.13/site-packages/aiohttp/web_request.py
— enforcement at lines 675-680 (`read()`) and 722/746/774 (`post()`).
Installed package ships no docs, so the verdict rests on the installed source
(`Request`'s own `client_max_size` default at `web_request.py:161` is likewise
`1024**2`).

## Source Version
aiohttp 3.13.3 (`aiohttp/__init__.py` line 1: `__version__ = "3.13.3"`)

## Source Fingerprint
sha256:94653f68030dfa1df0cbe2d34c78ba49e287f3276dd46e75057702b29803e590 (web_app.py, full file bytes)
sha256:2aaacea7a01e581e5eead2ab1b9e4ba3b65bc2ae3d0f176b2a78ae5b6b76bb4e (web_request.py, full file bytes)

## Invalidates On
- An aiohttp release that changes the `Application.__init__` default for
  `client_max_size` away from `1024**2`
- A release that changes the `>=` boundary to `>`, or moves enforcement off
  `BaseRequest.read()`/`post()` onto the parser or transport layer
- Any file whose sha256 differs from the fingerprints above

## Checked At
2026-10-06T22:21:04Z
