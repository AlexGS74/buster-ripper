VERDICT: HOLDS
PROMOTION: ELIGIBLE

## Claim
`_render_body` emits only `[tools]`, `[system]` and `[message i / role]` sections — every other top-level request-body key is absent from its output

## What it actually does
The renderer explicitly reads only `tools`, `system`, and `messages` and constructs sections only for these fields; it does not render other top-level keys as fields. Their names or values can nevertheless appear incidentally inside a tool definition or message text, so “absent” describes field rendering, not a guarantee that arbitrary strings never occur.

## Source
`buster_ripper/dump.py:49-80` (`_render_body`)

## Source Version
unavailable

## Source Fingerprint
sha256:c8f093c89b61c355ac182feb070cd90b84924ca0d187332117d76b5c766ad9b9

## Invalidates On
Change to `buster_ripper/dump.py`'s `_render_body` implementation or its input-field handling.

## Checked At
2026-09-27
