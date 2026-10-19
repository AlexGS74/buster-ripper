VERDICT: HOLDS
PROMOTION: ELIGIBLE

## Claim
`_cleanup_expired_images` removes no entry from the top-level `_image_cache` mapping

## What it actually does
The function binds `cache = _image_cache[session_id]` (the inner per-session dict) and the only
deletion it performs is `del cache[key]` for expired image keys. There is no `del _image_cache[...]`,
no `pop`/`clear`/rebinding of `_image_cache`, so a session key is never removed from the outer
mapping — even when every image inside that session has expired and the inner dict becomes empty
(empty inner dicts stay keyed in `_image_cache`). Removal from the top-level mapping happens only in
the separate `clear_session_cache`, not here.

## Source
/Users/alexhubitski/Projects/buster-ripper/buster_ripper/image.py — `_cleanup_expired_images`
(lines 47-59); outer mapping declared at line 14; contrast `clear_session_cache` (lines 182-185).

## Source Version
Repo working source, `buster_ripper/image.py` (no package version attached); fingerprinted below.

## Source Fingerprint
sha256:611df5acd7bf061242084de992a7d84102e0645a5c4b023e0d350841aa32dca7

## Invalidates On
- Any edit that adds a top-level deletion to `_cleanup_expired_images` (`del _image_cache[session_id]`,
  `.pop`, `.clear`, or rebinding `_image_cache`).
- Any change that makes `_image_cache` a flat single-level structure instead of
  session_id -> {image_key -> (source, ts)}.
- Replacement of the module-level dict with a cache object that prunes its own keys.

## Checked At
2026-10-05T18:52:21Z
