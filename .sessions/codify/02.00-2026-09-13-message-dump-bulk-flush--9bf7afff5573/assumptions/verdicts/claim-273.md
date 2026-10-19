VERDICT: UNCLEAR
PROMOTION: INELIGIBLE — verdict is UNCLEAR, not HOLDS/DOES-NOT-HOLD (eligibility requirement); facts themselves are fingerprinted

## Claim
`_replace_image_with_placeholder` stores the block's `source` value as the cache entry's payload

## What it actually does
`buster_ripper/image.py` line 80 reads `source = block.get("source", {})` and line 84 writes `_image_cache[session_id][image_key] = (source, time.time())`. The block's `source` value is therefore always stored in the cache entry, but the value the key maps to is a 2-tuple `(source_dict, timestamp)` — the module's own state comment (lines 13-14) describes entries that way, and both `_cleanup_expired_images` (line 55) and `_evict_by_lru` (line 72) index element `[1]` for the timestamp, while `get_cached_image` (line 160) unpacks `source, _`. Whether the claim is true depends on what "the cache entry's payload" names, and the source supports both readings, so no definition-free verdict is available (P9):
- Reading A — "payload" = the substantive data the cache holds for the key, as distinct from its bookkeeping stamp: the claim HOLDS, the source dict is exactly that data.
- Reading B — "payload" = the whole value stored under the key: the claim does not HOLD, since the stored value is the `(source, timestamp)` pair, not the source dict; additionally, when a block has no `source` key the stored first element is `{}` (the line 80 default), not any source value from the block.

## Source
/Users/alexhubitski/Projects/buster-ripper/buster_ripper/image.py — `_replace_image_with_placeholder` lines 78-93 (lines 80 and 84); corroborating entry-shape uses at lines 13-14, 55, 72, 160

## Source Version
local source file `buster_ripper/image.py` (no package version string in module); repository revision unavailable — git metadata is out of scope for this role

## Source Fingerprint
sha256:611df5acd7bf061242084de992a7d84102e0645a5c4b023e0d350841aa32dca7

## Invalidates On
- a rewrite of the claim naming the stored value explicitly (e.g. "the cache entry's value is a `(source, timestamp)` tuple whose first element is the block's `source`") would settle it to HOLDS under either reading
- any change to line 84 so the entry value is the bare source dict, or so it stores something other than `block["source"]` as element 0
- any change to the `_image_cache` value shape (lines 13-14, 55, 72, 160) that removes the paired timestamp

## Checked At
2026-10-05T18:40:39Z
