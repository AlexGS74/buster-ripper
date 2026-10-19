VERDICT: HOLDS
PROMOTION: ELIGIBLE

## Claim
A second write under an existing image key replaces that key's stored entry

## What it actually does
The only write into the per-session image cache is `_image_cache[session_id][image_key] = (source, time.time())`
(`buster_ripper/image.py:84`), a plain `dict.__setitem__` on the `dict[str, tuple[dict, float]]` declared at
`buster_ripper/image.py:14`. Re-assigning an existing key discards the previously stored `(source, timestamp)`
tuple wholesale and binds the new one — there is no merge, append, or versioning of the value, and no guard
(`setdefault`, `if key in cache`) anywhere on the write path. A subsequent `get_cached_image(session_id,
image_key)` therefore returns the newly stored `source`, and `_evict_by_lru`/`_cleanup_expired_images` operate on
the replaced tuple's new timestamp, not the original one.

Precision on the term "entry", as checked here: "entry" = the value bound to that key. Under that reading the
claim holds. One property the claim does **not** assert, and which differs from a remove-then-insert, is order:
Python dict re-assignment keeps the key at its original insertion position rather than moving it to the end, so a
replaced entry keeps the sort position used by `_evict_by_lru` (`image.py:72`) while carrying the new timestamp.

Reaching a second write under an *existing* key is possible in this implementation: `image_key` is
`f"img_{len(_image_cache[session_id])}"` (`image.py:81`) — an index recomputed from current size — and
`_evict_by_lru` deletes entries after the insert (`image.py:87`), shrinking `len(...)`. A later write can
therefore recompute an index that still names a live key, which is exactly the case this claim covers.

## Source
/Users/alexhubitski/Projects/buster-ripper/buster_ripper/image.py — write site `image.py:84`; cache type
declaration `image.py:14`; read site `image.py:160-161`; key derivation `image.py:81`; eviction `image.py:72-75`.
Corroborating language documentation: The Python Language Reference, "Mapping types" — `d[key] = a` binds `key` to
`a` (https://docs.python.org/3/library/stdtypes.html#mapping-types-dict).

## Source Version
Local repo source. `buster_ripper/image.py`, 5350 bytes, 185 lines. Git revision deliberately not resolved
(out of scope for this role).

## Source Fingerprint
sha256:611df5acd7bf061242084de992a7d84102e0645a5c4b023e0d350841aa32dca7

## Invalidates On
- The write at `image.py:84` changes from a bare `dict.__setitem__` to `setdefault`, a conditional insert, a list
  append, or any merge/accumulating structure (e.g. `dict[str, list[tuple[dict, float]]]` at `image.py:14`).
- `_image_cache`'s value type is replaced by a non-mapping or a custom container whose `__setitem__` does not
  overwrite existing keys.
- A guard is added on the write path that detects a duplicate `image_key` and allocates a fresh key instead
  (e.g. uniqueness loop, hash- or counter-based keying replacing `img_{len(...)}` at `image.py:81`).

## Checked At
2026-10-05T18:41:30Z
