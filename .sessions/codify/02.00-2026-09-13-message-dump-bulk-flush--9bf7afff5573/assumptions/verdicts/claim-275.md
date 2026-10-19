VERDICT: HOLDS
PROMOTION: ELIGIBLE

## Claim
`_replace_image_with_placeholder` calls `_evict_by_lru` after inserting the entry

## What it actually does
The body sequences insert-then-evict: line 84 writes `_image_cache[session_id][image_key] = (source, time.time())`, and only after that does line 87 call `_evict_by_lru(session_id)` (comment on line 86: "Evict oldest entries if over limit"), before the placeholder block is returned on lines 90-93. So eviction is evaluated against the size that includes the just-inserted entry — `_evict_by_lru` (lines 62-75) returns early while `len(cache) <= config.IMAGE_CACHE_MAX_SIZE` and otherwise deletes the oldest-by-timestamp `len(cache) - IMAGE_CACHE_MAX_SIZE` keys.

## Source
/Users/alexhubitski/Projects/buster-ripper/buster_ripper/image.py — `_replace_image_with_placeholder` lines 78-93 (insert line 84, call line 87); `_evict_by_lru` lines 62-75, which reads the limit as `config.IMAGE_CACHE_MAX_SIZE` at lines 68 and 73 (the setting itself lives in the `config` module imported at line 10, outside this claim's bounded source)

## Source Version
local source file `buster_ripper/image.py` (no package version string in module); repository revision unavailable — git metadata is out of scope for this role

## Source Fingerprint
sha256:611df5acd7bf061242084de992a7d84102e0645a5c4b023e0d350841aa32dca7

## Invalidates On
- reordering lines 84 and 87 so `_evict_by_lru` runs before the insert, or moving eviction out of this function into a caller
- removing the `_evict_by_lru(session_id)` call on line 87 or replacing it with a different eviction/cleanup routine
- a signature or name change to `_evict_by_lru` (lines 62-75) that makes the line 87 call resolve elsewhere

## Checked At
2026-10-05T18:40:39Z
