VERDICT: HOLDS
PROMOTION: ELIGIBLE

## Claim
`_evict_by_lru` orders eviction by the cache entry's second element

## What it actually does
The eviction order is set by `sorted_items = sorted(cache.items(), key=lambda x: x[1][1])`,
then the oldest-first slice `sorted_items[:excess]` is deleted. `cache` is the per-session
mapping whose values are the 2-tuples `(source_dict, timestamp)` (module comment line 13;
written as `(source, time.time())` at line 84), so for each `cache.items()` pair `x`, `x[1]`
is that entry tuple and `x[1][1]` is its second element — the timestamp. Sorting is ascending,
so the entries with the smallest second element are evicted first.

Terminology made explicit rather than chosen silently: "cache entry" is read as the value
stored under an `image_key`, which is how this module itself describes it (lines 13, 55, 84,
160 all destructure the entry as `(_, ts)` / `(source, _)`). If instead one called the
`(key, value)` pair produced by `.items()` the "entry", then the sort key would be the second
element *of* the entry's second element, not the entry's second element itself. The behaviour
being claimed — eviction ordered by the timestamp field — is the same either way; only the
label differs.

Related, not part of this claim: the second element is written once at cache-insert time
(line 84) and is never refreshed on a read (`get_cached_image` does not re-stamp), so the
ordering is oldest-cached-first, not last-accessed-first.

## Source
/Users/alexhubitski/Projects/buster-ripper/buster_ripper/image.py — `_evict_by_lru` line 72
(`key=lambda x: x[1][1]`), entry shape at lines 13, 55, 84, 160

## Source Version
working-tree source, no revision string available in the bounded source

## Source Fingerprint
sha256:611df5acd7bf061242084de992a7d84102e0645a5c4b023e0d350841aa32dca7

## Invalidates On
- Line 72's key function changes index depth or field (e.g. `x[1][0]`, or sorting by `x[0]`)
- The cached value tuple is reordered or gains/loses a field, so the timestamp is no longer
  the second element (lines 13, 55, 84, 160)

## Checked At
2026-10-05
