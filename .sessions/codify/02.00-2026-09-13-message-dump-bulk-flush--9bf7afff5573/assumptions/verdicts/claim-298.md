VERDICT: HOLDS
PROMOTION: ELIGIBLE

## Claim
`_cleanup_expired_images` reads `config.IMAGE_CACHE_TTL` at call time rather than binding it at import

## What it actually does
The module imports the config *module object* (`from . import config`, line 10), never the value, and
the only mention of `IMAGE_CACHE_TTL` in the file is inside the function body at line 56 as an
attribute lookup on that module object. There is no import-time snapshot — no module-level alias such
as `_TTL = config.IMAGE_CACHE_TTL`, and no default-argument capture (the signature is
`def _cleanup_expired_images(session_id: str) -> None`). So `config.IMAGE_CACHE_TTL` is resolved by
attribute lookup on each execution of the function, and a value changed in `config` after import is
seen by the next call.

## Source
/Users/alexhubitski/Projects/buster-ripper/buster_ripper/image.py — import at line 10
(`from . import config`); read at line 56 (`if now - ts > config.IMAGE_CACHE_TTL`); signature at
line 47. Full-file scan for `IMAGE_CACHE_TTL` returns line 56 only.

## Source Version
Repo working source, `buster_ripper/image.py` (no package version attached); fingerprinted below.

## Invalidates On
- Changing line 10 to `from .config import IMAGE_CACHE_TTL` (or any `as`-aliased value import), which
  freezes the value at import.
- Capturing the value at module import (`_TTL = config.IMAGE_CACHE_TTL`) or as a default argument
  (`def _cleanup_expired_images(session_id, ttl=config.IMAGE_CACHE_TTL)`) and comparing against that.
- Replacing the bare attribute read on line 56 with a cached/hoisted lookup outside the function body.

## Checked At
2026-10-05T18:52:21Z
