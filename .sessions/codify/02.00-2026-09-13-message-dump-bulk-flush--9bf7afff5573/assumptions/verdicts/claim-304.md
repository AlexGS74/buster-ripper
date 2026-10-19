VERDICT: HOLDS
PROMOTION: ELIGIBLE

## Claim
`_evict_by_lru` reads `config.IMAGE_CACHE_MAX_SIZE` at call time rather than binding it at import

## What it actually does
The module imports the config *module object* — `from . import config` (line 10) — and never
binds the constant to a module-level name, a default argument, or a closure cell. Both uses
inside `_evict_by_lru` are attribute lookups on that module object written in the function
body: `if len(cache) <= config.IMAGE_CACHE_MAX_SIZE:` (line 68) and
`excess = len(cache) - config.IMAGE_CACHE_MAX_SIZE` (line 73). Attribute resolution happens
when the function executes, so each call re-reads the current value of
`config.IMAGE_CACHE_MAX_SIZE`; a change to that attribute after `buster_ripper.image` was
imported is seen by the next call.

Counterfactual for the record: binding at import would look like
`from .config import IMAGE_CACHE_MAX_SIZE` or `def _evict_by_lru(session_id, limit=IMAGE_CACHE_MAX_SIZE)`.
Neither form appears in the consulted source.

## Source
/Users/alexhubitski/Projects/buster-ripper/buster_ripper/image.py — line 10 (`from . import config`),
lines 68 and 73 (attribute uses inside `_evict_by_lru`)

## Source Version
working-tree source, no revision string available in the bounded source

## Source Fingerprint
sha256:611df5acd7bf061242084de992a7d84102e0645a5c4b023e0d350841aa32dca7

## Invalidates On
- Line 10 changes to `from .config import IMAGE_CACHE_MAX_SIZE` (or any `from ... import` of
  the constant), freezing the value in the image module's namespace
- A module-level alias or default-argument/closure capture of the value is introduced and the
  lines 68/73 uses are switched to it
- Lines 68/73 stop referencing `config.` at call time

## Checked At
2026-10-05
