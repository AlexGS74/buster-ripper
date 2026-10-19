VERDICT: HOLDS
PROMOTION: ELIGIBLE

## Claim
`_render_body`'s output carries the request's conversation as one `[message i / role]` section per message

## What it actually does
It enumerates `body.get("messages") or []` and appends exactly one `[message {i} / {role}]` section for each message, rendering its content as text. The separate `system` field is rendered in `[system]`, not as a message section; structured content may lose its original representation.

## Source
`buster_ripper/dump.py:67-80` (`_render_body`)

## Source Version
unavailable

## Source Fingerprint
sha256:c8f093c89b61c355ac182feb070cd90b84924ca0d187332117d76b5c766ad9b9

## Invalidates On
Change to `buster_ripper/dump.py`'s `_render_body` message loop or its input-field handling.

## Checked At
2026-09-27
