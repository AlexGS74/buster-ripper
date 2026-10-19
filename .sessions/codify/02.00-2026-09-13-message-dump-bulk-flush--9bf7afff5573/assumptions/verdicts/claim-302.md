VERDICT: HOLDS
PROMOTION: ELIGIBLE

## Claim
`_evict_by_lru` removes entries until the session mapping holds at most `config.IMAGE_CACHE_MAX_SIZE` entries

## What it actually does
After the early return for `len(cache) <= config.IMAGE_CACHE_MAX_SIZE`, the function computes
`excess = len(cache) - config.IMAGE_CACHE_MAX_SIZE` and deletes exactly that many keys
(`for key, _ in sorted_items[:excess]: del cache[key]`), so the post-condition for any
non-negative limit is `len(cache) == config.IMAGE_CACHE_MAX_SIZE`, which satisfies
"at most `IMAGE_CACHE_MAX_SIZE` entries". Mechanically it is a single computed deletion
batch rather than a loop that re-tests the size each iteration; the resulting state is the
same, and it is the same slice of oldest-first items either way.

Boundary note (does not change the verdict, but bounds the guarantee): the limit's value is
not defined in the consulted source. For any `IMAGE_CACHE_MAX_SIZE >= 0` the post-condition
holds. If the value were negative, `excess > len(cache)`, `sorted_items[:excess]` deletes
every entry, and the mapping ends empty (0 entries) — still "at most", but only vacuously
since no negative size is representable as an entry count.

## Source
/Users/alexhubitski/Projects/buster-ripper/buster_ripper/image.py — `_evict_by_lru`, lines 62-75
(size test line 68; `excess` line 73; deletion loop lines 74-75)

## Source Version
working-tree source, no revision string available in the bounded source

## Source Fingerprint
sha256:611df5acd7bf061242084de992a7d84102e0645a5c4b023e0d350841aa32dca7

## Invalidates On
- Lines 68/73-75 of `buster_ripper/image.py` change the size test operand, the `excess`
  formula, or the deletion slice
- `IMAGE_CACHE_MAX_SIZE` is redefined as a non-numeric or negative value, or eviction is
  switched to a different limit constant

## Checked At
2026-10-05
