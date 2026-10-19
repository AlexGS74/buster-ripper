VERDICT: HOLDS
PROMOTION: ELIGIBLE

## Claim
`strip_and_cache_images` leaves messages whose role is not `user` untouched

## What it actually does
The first statement inside the loop is `if msg.get("role") != "user": continue`, so any message whose role is not exactly `"user"` — including `"assistant"`, `"system"`, `"tool"`, or a message with no `role` key (`None != "user"`) — bypasses every mutation in the function. The only assignment the loop performs is `msg["content"] = _process_user_content(content, session_id)`, which is reached only past that guard; nothing else in `strip_and_cache_images` writes into `body`. Caching work (`_replace_image_with_placeholder`, which writes to `_image_cache`) is likewise reachable only from that user-role branch. The message dicts for non-user roles are neither replaced nor modified; the returned value is the same `body` object passed in, with non-user messages unchanged.

(Corroborating scope note, established from the source and not from any test: the in-place mutation means a caller holding the request dict sees only user-message `content` rewritten.)

## Source
/Users/alexhubitski/Projects/buster-ripper/buster_ripper/image.py — `strip_and_cache_images`, lines 128-138 (guard at 129-130; sole mutation at 136; `return body` at 138)

## Source Version
unavailable — in-repo module, no package version; no git metadata read per this role's git-metadata boundary. Stability carried by the byte fingerprint.

## Source Fingerprint
sha256:611df5acd7bf061242084de992a7d84102e0645a5c4b023e0d350841aa32dca7 (whole file, 185 lines / 5350 bytes, all consulted)

## Invalidates On
- The role guard at lines 129-130 is removed, inverted, widened (e.g. `in {"user", "tool"}`), or replaced by index/position-based handling, so non-user messages are visited
- A second mutation of `msg` or `body` is added to the loop outside the user-role branch (e.g. normalizing `id`, `name`, or stripping fields for all roles)
- `_process_user_content`/`_process_content_block` is changed to mutate a shared or referenced object reachable from non-user messages

## Checked At
2026-10-05T18:38:16Z
