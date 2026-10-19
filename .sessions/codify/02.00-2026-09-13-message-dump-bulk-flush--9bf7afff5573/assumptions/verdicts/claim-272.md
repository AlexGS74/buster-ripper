VERDICT: HOLDS
PROMOTION: ELIGIBLE

## Claim
`_replace_image_with_placeholder` derives its cache key from the session mapping's current entry count

## What it actually does
Line 81 of `buster_ripper/image.py` reads `image_key = f"img_{len(_image_cache[session_id])}"`: the key is the literal prefix `img_` concatenated with `len()` of that session's mapping (`_image_cache[session_id]`, the `image_key -> (source_dict, timestamp)` dict) — i.e. the number of entries the session mapping holds at the moment the key is computed. The count is taken before the insert on line 84, so the first image in a session yields `img_0`; because `_evict_by_lru` (line 87) deletes entries, the count is not monotonic across calls and a later key can repeat an earlier one and overwrite that entry.

## Source
/Users/alexhubitski/Projects/buster-ripper/buster_ripper/image.py — `_replace_image_with_placeholder`, lines 78-93 (key derivation at line 81); cache shape declared in the module-level comment and `_image_cache` initializer, lines 13-14

## Source Version
local source file `buster_ripper/image.py` (no package version string in module); repository revision unavailable — git metadata is out of scope for this role

## Source Fingerprint
sha256:611df5acd7bf061242084de992a7d84102e0645a5c4b023e0d350841aa32dca7

## Invalidates On
- any change to the `image_key` expression on line 81 (different prefix, different `len()` argument, or hashing/content-derived keying)
- any change to `_image_cache`'s structure so `_image_cache[session_id]` is no longer the per-session mapping keyed by image key (lines 13-14)
- insertion, eviction, or expiry reordering that changes whether the count is read before or after the entry is written (lines 84-87, `_evict_by_lru` lines 62-75)

## Checked At
2026-10-05T18:40:39Z
