VERDICT: HOLDS
PROMOTION: ELIGIBLE

## Claim
`_replace_image_with_placeholder` stores the cache entry's value as a `(source, timestamp)` tuple whose first element is the block's `source` value

## What it actually does
In `buster_ripper/image.py`, `_replace_image_with_placeholder` reads `source = block.get("source", {})` (line 80) and writes `_image_cache[session_id][image_key] = (source, time.time())` (line 84) — a 2-tuple whose first element is the image block's own `source` value and whose second element is the `float` from `time.time()` at insert time. The only deviation from "the block's `source` value" verbatim is the `.get` default: when the block has no `source` key, the first element is `{}` rather than a missing key. The declared cache type on line 14 (`dict[str, dict[str, tuple[dict, float]]]`) and the destructure on line 160 (`source, _ = _image_cache[session_id][image_key]`) agree with that shape, as do the tuple-index readers `_cleanup_expired_images` (`for key, (_, ts) in cache.items()`, line 55) and `_evict_by_lru` (`key=lambda x: x[1][1]`, line 72).

Read from production source only. No test, fixture, or QA artifact was consulted, and nothing was executed.

## Source
`/Users/alexhubitski/Projects/buster-ripper/buster_ripper/image.py` — `_replace_image_with_placeholder`, lines 78-94 (corroborating: line 14 type comment, lines 55, 72, 160 tuple consumers)

## Source Version
unavailable — local worktree source at the supplied source root; no revision or package version consulted (git metadata is out of scope for this role). The consulted bytes are identified by the fingerprint below instead.

## Source Fingerprint
sha256:611df5acd7bf061242084de992a7d84102e0645a5c4b023e0d350841aa32dca7

## Invalidates On
- `_replace_image_with_placeholder` stores a value of different arity or different element order (e.g. plain dict entry, `(timestamp, source)`, or a record/object instead of a 2-tuple)
- the first element stops being the block's `source` value — e.g. it becomes the whole block, `block["source"]["data"]`, or a derived/encoded form
- the cache write moves out of `_replace_image_with_placeholder` to another function that establishes a different value shape
- the module-level `_image_cache` declaration (line 14) or its consumers (lines 55, 72, 160) are changed to a shape inconsistent with `(source, timestamp)`

## Checked At
2026-10-05T21:12:41Z
