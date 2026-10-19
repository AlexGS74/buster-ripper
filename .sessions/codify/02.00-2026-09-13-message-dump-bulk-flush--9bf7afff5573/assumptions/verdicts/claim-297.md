VERDICT: HOLDS
PROMOTION: ELIGIBLE

## Claim
`_cleanup_expired_images` treats an entry as expired only when its age strictly exceeds the configured TTL

## What it actually does
The expiry predicate is `now - ts > config.IMAGE_CACHE_TTL` with a strict `>` — no `>=`, no tolerance
term. An entry whose age equals the TTL exactly (`now - ts == IMAGE_CACHE_TTL`) is therefore NOT
collected into `expired_keys` and survives the pass; it becomes eligible only on a later call once the
age is strictly greater than the TTL. Age is measured against a fresh `time.time()` read per call.

## Source
/Users/alexhubitski/Projects/buster-ripper/buster_ripper/image.py — `_cleanup_expired_images`
lines 53-57: `now = time.time()` then
`expired_keys = [key for key, (_, ts) in cache.items() if now - ts > config.IMAGE_CACHE_TTL]`.

## Source Version
Repo working source, `buster_ripper/image.py` (no package version attached); fingerprinted below.

## Invalidates On
- Changing `>` to `>=` on line 56 (boundary age would then count as expired).
- Rounding/adjusting `now` or `ts`, or adding slack to the TTL expression, so strictness no longer
  maps to "age exceeds the configured TTL".
- Storing absolute expiry deadlines instead of a creation timestamp, which changes what "age" means
  in this predicate.

## Checked At
2026-10-05T18:52:21Z
