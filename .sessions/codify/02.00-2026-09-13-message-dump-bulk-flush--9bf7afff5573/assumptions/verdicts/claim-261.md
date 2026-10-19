VERDICT: HOLDS
PROMOTION: ELIGIBLE

## Claim
`handle_messages` calls `strip_and_cache_images` only when `config.STRIP_IMAGES` and
`has_images(body)` are both true

## What it actually does
`handle_messages` has exactly one call to `strip_and_cache_images` — line 41,
`image_handler.strip_and_cache_images(body, sid)` — and it is the body of the guard
`if config.STRIP_IMAGES and image_handler.has_images(body):` (line 40). Because the
predicate is a short-circuit `and`, the body runs only when `config.STRIP_IMAGES` is
truthy **and** `has_images(body)` returns truthy for the (already normalized) `body`;
either falsy operand skips the call. `has_images` is reached through the same
`image_handler` alias (`from .. import image as image_handler`, line 11), so the
claim's `has_images(body)` is the same call in the same operand position. The name
`strip_and_cache_images` occurs nowhere else in the file (293 lines checked).

## Source
`/Users/alexhubitski/Projects/buster-ripper/buster_ripper/routes/messages.py`
lines 26–71 (`handle_messages` body), specifically lines 40–41 and the import at
line 11; name occurrences verified over the full file.

## Source Version
Repo source file read from the supplied source root; no package version string was
consulted, and git metadata/revision state is out of scope for this role. The
fingerprint below pins the consulted bytes.

## Invalidates On
- Any edit changing the line-40 predicate (e.g. `or` instead of `and`, dropping the
  `config.STRIP_IMAGES` operand, or testing a different config field).
- Any edit adding another `strip_and_cache_images` call site in `handle_messages`, or
  moving this one out of the guarded block.
- Any change to `buster_ripper/routes/messages.py` whose bytes differ from the
  fingerprint above, or to `buster_ripper/image.py`'s `has_images`/`strip_and_cache_images`
  semantics if the claim is later read as covering truthiness of `has_images` results
  (`has_images` itself was not opened — it is outside the supplied source root).

## Checked At
2026-10-06
