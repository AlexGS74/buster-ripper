VERDICT: HOLDS
PROMOTION: ELIGIBLE

## Claim
`_replace_image_with_placeholder` returns a `type` text block whose `text` embeds the cache key

## What it actually does
Lines 90-93 return the literal dict `{"type": "text", "text": f"[image:{image_key}]"}` — a content block tagged `type: "text"` whose `text` field interpolates the same `image_key` created on line 81 (`img_<count>`) inside the `[image:...]` marker, so the returned block carries the cache key it was just written under. The block is a fresh dict; the original image block's other fields (`type: "image"`, `source`, etc.) are not carried into the return value.

## Source
/Users/alexhubitski/Projects/buster-ripper/buster_ripper/image.py — `_replace_image_with_placeholder`, lines 78-93 (key at line 81, return at lines 90-93); the placeholder is consumed downstream via `get_cached_image` at lines 141-161

## Source Version
local source file `buster_ripper/image.py` (no package version string in module); repository revision unavailable — git metadata is out of scope for this role

## Source Fingerprint
sha256:611df5acd7bf061242084de992a7d84102e0645a5c4b023e0d350841aa32dca7

## Invalidates On
- any change to the returned dict on lines 90-93 — a different `type` value, a different field name than `text`, or a `text` string that omits `image_key`
- any change to how `image_key` is built (line 81) that stops the key appearing verbatim in `text`
- introduction of a separate identifier in the placeholder rather than the cache key used for the store on line 84

## Checked At
2026-10-05T18:40:39Z
