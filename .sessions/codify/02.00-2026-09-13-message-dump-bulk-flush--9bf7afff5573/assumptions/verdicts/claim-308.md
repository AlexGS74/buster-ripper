VERDICT: HOLDS
PROMOTION: ELIGIBLE

## Claim
`get_cached_image` unpacks its cache entry into exactly two targets

## What it actually does
`get_cached_image` performs one unpacking assignment on the cache entry —
`source, _ = _image_cache[session_id][image_key]` at `buster_ripper/image.py:160`
— whose left-hand side has exactly two targets (`source` and the throwaway `_`),
and it returns only the first. This matches the entry shape the module writes:
`_image_cache[session_id][image_key] = (source, time.time())` at :84 and the
declared type `dict[str, dict[str, tuple[dict, float]]]` at :14, i.e. a
2-tuple `(source_dict, timestamp)`. No other unpacking of an entry occurs in
this function, so the two-target count is neither more nor fewer. (The sibling
`_cleanup_expired_images` also unpacks two targets — `for key, (_, ts) in ...`
at :55 — and `_evict_by_lru` reaches the timestamp positionally via `x[1][1]` at
:72; neither is part of `get_cached_image`.)

## Source
/Users/alexhubitski/Projects/buster-ripper/buster_ripper/image.py —
unpacking assignment at :160; corroborating write site at :84 and type
declaration at :14

## Source Version
local working-tree file `buster_ripper/image.py`, 185 lines (no revision string
recorded; fingerprint below is the stability anchor)

## Source Fingerprint
sha256:611df5acd7bf061242084de992a7d84102e0645a5c4b023e0d350841aa32dca7

## Invalidates On
- Changing the tuple written at :84 or the declared value type at :14 to a
  different arity (e.g. adding a third field such as media type or hit count)
- Rewriting :160 to index positions instead of unpacking
  (`[...] [0]`), to unpack a starred target (`source, *_`), or to bind a
  different number of names
- Replacing the dict-of-tuples cache with dataclass/dict entries so there is no
  tuple to unpack
- Editing `buster_ripper/image.py` at all invalidates the fingerprint and
  requires re-reading before reuse

## Checked At
2026-10-05T18:55:46Z
