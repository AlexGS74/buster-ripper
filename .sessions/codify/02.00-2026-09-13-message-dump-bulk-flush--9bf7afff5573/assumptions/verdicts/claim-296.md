VERDICT: HOLDS
PROMOTION: ELIGIBLE

## Claim
`_cleanup_expired_images` compares a cache entry's second element against `config.IMAGE_CACHE_TTL`

## What it actually does
Entries are 2-tuples `(source_dict, timestamp)` and the comprehension unpacks them as
`(_, ts)` — the first element is discarded, the second is `ts` — and evaluates
`now - ts > config.IMAGE_CACHE_TTL`. So the entry field that reaches the TTL comparison is index 1,
the timestamp; the exact operand is the derived age `now - ts`, not the raw timestamp value.
Right-hand side of the comparison is `config.IMAGE_CACHE_TTL` verbatim.

## Source
/Users/alexhubitski/Projects/buster-ripper/buster_ripper/image.py — `_cleanup_expired_images`
line 56 (`if now - ts > config.IMAGE_CACHE_TTL`) with the `key, (_, ts) in cache.items()` unpack on
line 55; entry shape documented at line 13 (`(source_dict, timestamp)`) and written at line 84
(`_image_cache[session_id][image_key] = (source, time.time())`).

## Source Version
Repo working source, `buster_ripper/image.py` (no package version attached); fingerprinted below.

## Source Fingerprint
sha256:611df5acd7bf061242084de992a7d84102e0645a5c4b023e0d350841aa32dca7

## Invalidates On
- Reordering or reshaping the cached entry tuple so the timestamp is no longer index 1 (e.g.
  `(timestamp, source)` or a 3-tuple/dataclass), which changes which element the comparison uses.
- Changing line 56 to compare against a different bound (`IMAGE_CACHE_MAX_AGE`, a constant, or a
  literal) instead of `config.IMAGE_CACHE_TTL`.
- Switching the comparison to store an absolute expiry (`ts + ttl < now`) — still index 1, but the
  "second element vs TTL" description no longer matches the operand.

## Checked At
2026-10-05T18:52:21Z
