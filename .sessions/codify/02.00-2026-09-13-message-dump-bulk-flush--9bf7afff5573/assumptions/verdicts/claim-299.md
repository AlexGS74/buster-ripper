VERDICT: HOLDS
PROMOTION: ELIGIBLE

## Claim
`cleanup_expired_images` calls `_cleanup_expired_images` once per session and performs no other operation

## What it actually does
`cleanup_expired_images()` is a zero-arg function whose entire body is a two-statement
loop: `for session_id in list(_image_cache.keys()): _cleanup_expired_images(session_id)`
(image.py lines 176-179). The loop body contains exactly that one call, and each dict key
is visited exactly once, so the helper is invoked once per session — where the sessions
addressed are the keys currently present in the module-level `_image_cache` (a session
with no cache entry receives no call; the docstring's "all sessions" means all cached
sessions). No other call, assignment, return value, or mutation happens in the function:
`list(...)` only snapshots the keys, and `_cleanup_expired_images` mutates the inner
per-session dict, not `_image_cache` itself, so the snapshot is not required for
correctness — it is still not an additional operation of the caller.

A grep of the whole source for the name confirms a single `def cleanup_expired_images`
(line 176) with no reassignment or aliasing, and its only in-body reference to the helper
is line 179. (The other helper call site, line 155, is inside `get_cached_image`, not
inside this function.)

## Source
/Users/alexhubitski/Projects/buster-ripper/buster_ripper/image.py — lines 176-179
(definition and body of `cleanup_expired_images`); corroborating name occurrences at
lines 47, 155, 179 via case-sensitive grep of the same file.

## Source Version
supplied source root was the single file `buster_ripper/image.py`; no package version
string consulted (185-line file read in full).

## Source Fingerprint
sha256:611df5acd7bf061242084de992a7d84102e0645a5c4b023e0d350841aa32dca7

## Invalidates On
- The body of `cleanup_expired_images` changes to add any statement beyond the single
  per-session helper call (e.g. also calling `_evict_by_lru`, `clear_session_cache`,
  touching `config`, or returning a count).
- The loop's iterable changes so a session key can be visited more or less than once
  (e.g. iterating a list built from `_image_cache` with duplicates, nesting, or a
  `while`/recursive pass).
- `_image_cache` is no longer the module-level `dict` keyed by `session_id` (line 14),
  which changes what "per session" means for this function.
- A second definition, monkeypatch, or re-export of `cleanup_expired_images` appears
  outside this file that supersedes the one at line 176 (the supplied source root is
  this file only, so that could not be checked here).

## Checked At
2026-10-05T18:52:30Z
