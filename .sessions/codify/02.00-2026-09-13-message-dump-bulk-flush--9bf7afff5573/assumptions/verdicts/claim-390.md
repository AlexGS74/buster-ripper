VERDICT: HOLDS
PROMOTION: ELIGIBLE

## Claim
`aiohttp.web.Application.__init__` accepts a `client_max_size` keyword argument

## What it actually does
`Application.__init__` declares `client_max_size: int = 1024**2` in its parameter
list, after a bare `*`, so it is a keyword-only argument. The value is stored on
the instance as `self._client_max_size` and nothing in `__init__` rejects,
renames, or deprecates it (unlike `router`, `loop`, and `debug`, which each emit
a `DeprecationWarning`).

## Source
/Users/alexhubitski/Projects/buster-ripper/.venv/lib/python3.13/site-packages/aiohttp/web_app.py
— `Application.__init__` signature at lines 121-131 (`*` at 123,
`client_max_size: int = 1024**2` at 128), assignment `self._client_max_size =
client_max_size` at line 175, and `"_client_max_size"` in `Application.ATTRS`
at line 116.

Docs are not shipped inside the installed package (no `.rst`/docs tree in
`aiohttp/`), so the installed source is the primary evidence here. Upstream
`web_advanced.html` was consulted for corroboration only and is not the basis of
this verdict.

## Source Version
aiohttp 3.13.3 (`aiohttp/__init__.py` line 1: `__version__ = "3.13.3"`)

## Source Fingerprint
sha256:94653f68030dfa1df0cbe2d34c78ba49e287f3276dd46e75057702b29803e590 (web_app.py, full file bytes)

## Invalidates On
- aiohttp upgraded to a release where `client_max_size` is removed from or renamed
  in `Application.__init__`'s signature
- A future release that makes `client_max_size` positional-acceptable (the `*`
  sentinel removed) or moves it behind a deprecated alias — changes the
  keyword-only character of the argument
- Verified against the installed 3.13.3 source; any file whose sha256 differs
  from the fingerprint above must be re-read before reuse

## Checked At
2026-10-06T22:21:04Z
