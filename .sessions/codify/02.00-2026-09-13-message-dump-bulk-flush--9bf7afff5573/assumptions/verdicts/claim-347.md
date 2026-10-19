VERDICT: HOLDS
PROMOTION: ELIGIBLE

## Claim
`clear_session_cache` removes the named session's entry from the `_image_cache` mapping

## What it actually does
`clear_session_cache(session_id)` is defined at `buster_ripper/image.py:182-185` and its
entire body is a membership-guarded `del _image_cache[session_id]`. `_image_cache` is the
module-level `dict[str, dict[str, tuple[dict, float]]]` declared at line 14 and keyed by
`session_id`, so the deleted item is exactly the top-level entry for the named session
(that session's whole image-key map goes with it). When the session is not present, the
`if` guard makes the call a no-op rather than an error; in both cases the named session's
entry is absent from the mapping after the call returns. No other definition of
`clear_session_cache` or any other `_image_cache` binding exists in the source read, and
no rebinding/reassignment of `_image_cache` happens outside the per-session entry writes
(lines 84, 122) — the only container-level deletion is this one.

## Source
/Users/alexhubitski/Projects/buster-ripper/buster_ripper/image.py — production source,
lines 182-185 (`clear_session_cache`), with lines 13-14 (`_image_cache` declaration and
its session-id-keyed shape) as the binding the deletion targets. Whole file (185 lines)
read; this is a claim about the repo, so the implementation module is the ground. No test,
fixture, or snapshot was consulted, and none was needed.

## Source Version
Supplied source root is the file itself (no revision locator provided, and Git metadata is
out of scope for this role). Consulted bytes are the working file
`buster_ripper/image.py`, size 5350 bytes, mtime 2026-03-04.

## Source Fingerprint
sha256:611df5acd7bf061242084de992a7d84102e0645a5c4b023e0d350841aa32dca7

## Invalidates On
- The body of `clear_session_cache` changes from `del _image_cache[session_id]` to any
  other effect (e.g. clearing nested keys only, dropping entries by TTL, mutating a copy
  of the map, or delegating to another cache).
- `_image_cache` is renamed, re-typed so it is no longer a session-id-keyed mapping, or a
  second shadowing cache/store becomes the one that `clear_session_cache` touches.
- A second definition of `clear_session_cache` is introduced that takes effect at the call
  sites (later definition in this module, or a re-export/override elsewhere).
- The file bytes no longer match the fingerprint above.

## Checked At
2026-10-05T21:35:49Z
