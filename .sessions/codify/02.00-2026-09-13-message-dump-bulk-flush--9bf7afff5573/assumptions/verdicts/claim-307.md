VERDICT: HOLDS
PROMOTION: ELIGIBLE

## Claim
`get_cached_image` indexes `_image_cache` by session after calling `_cleanup_expired_images` for that session

## What it actually does
Inside `get_cached_image` (`buster_ripper/image.py:141-161`) the order of
operations is: a plain membership test `session_id not in _image_cache` at :151,
then `_cleanup_expired_images(session_id)` at :155, and only after that does the
function subscript the module cache with the session key —
`image_key not in _image_cache[session_id]` at :157 and
`source, _ = _image_cache[session_id][image_key]` at :160. Both session-indexed
reads therefore occur strictly after the per-session cleanup call, so the
function can never return an entry that cleanup had already expired for that
session. (`_cleanup_expired_images` itself re-reads `_image_cache[session_id]`
at :52, but that is inside the callee, not the indexing the claim attributes to
`get_cached_image`.)

## Source
/Users/alexhubitski/Projects/buster-ripper/buster_ripper/image.py —
`get_cached_image` body, lines 151-161 (call at :155; session subscripts at :157
and :160)

## Source Version
local working-tree file `buster_ripper/image.py`, 185 lines (no revision string
recorded; fingerprint below is the stability anchor)

## Source Fingerprint
sha256:611df5acd7bf061242084de992a7d84102e0645a5c4b023e0d350841aa32dca7

## Invalidates On
- Reordering `get_cached_image` so a `_image_cache[session_id]...` subscript
  executes before the `_cleanup_expired_images(session_id)` call (e.g. hoisting
  `cache = _image_cache[session_id]` above the cleanup)
- Removing or renaming the `_cleanup_expired_images` call, or moving expiry
  filtering into the lookup expression instead of a preceding call
- Changing `_cleanup_expired_images` to take a dict rather than a `session_id`,
  which removes the "for that session" argument the claim names
- Editing `buster_ripper/image.py` at all invalidates the fingerprint and
  requires re-reading before reuse

## Checked At
2026-10-05T18:55:46Z
