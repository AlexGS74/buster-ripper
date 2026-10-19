VERDICT: HOLDS
PROMOTION: ELIGIBLE

## Claim
`_evict_by_lru` limits its work to one session's mapping

## What it actually does
The function takes a single `session_id`, returns immediately if that id is absent from
`_image_cache` (lines 64-65), and then binds `cache = _image_cache[session_id]` (line 67).
Every later operation is on that one inner dict — `len(cache)`, `cache.items()`, the sort, and
`del cache[key]` (lines 68-75). The outer `_image_cache` is never iterated and its session keys
are never added or removed, so no other session is read, sized, or evicted, and the size cap is
enforced independently per session (the global cache can therefore hold up to
`IMAGE_CACHE_MAX_SIZE * number_of_sessions` entries — a consequence of the scoping, not a
separate claim).

Contrast within the same module, which shows the cross-session form when it exists:
`cleanup_expired_images()` loops `for session_id in list(_image_cache.keys())` (lines 176-179),
and `clear_session_cache()` deletes the outer key itself (lines 182-185). `_evict_by_lru`
resembles neither.

## Source
/Users/alexhubitski/Projects/buster-ripper/buster_ripper/image.py — `_evict_by_lru`, lines 62-75
(session scoping at lines 64-67; mutation confined to `cache` at lines 68-75); cross-session
comparison at lines 176-179 and 182-185

## Source Version
working-tree source, no revision string available in the bounded source

## Source Fingerprint
sha256:611df5acd7bf061242084de992a7d84102e0645a5c4b023e0d350841aa32dca7

## Invalidates On
- Lines 62-75 gain a loop over `_image_cache`/`_image_cache.keys()` or `.items()`, or a call
  that mutates another session's dict or an outer key
- The signature stops carrying one `session_id` (e.g. a global eviction pass), or `del
  _image_cache[session_id]` / a reassignment of `_image_cache` is introduced in the function

## Checked At
2026-10-05
