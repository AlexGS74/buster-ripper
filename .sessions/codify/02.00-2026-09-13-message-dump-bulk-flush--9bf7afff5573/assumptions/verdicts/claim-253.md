VERDICT: HOLDS
PROMOTION: ELIGIBLE

## Claim
`has_images` inspects only the first user message it meets when scanning `messages` from the end

## What it actually does
`has_images` iterates `reversed(messages)`, skipping every message whose `role` is not `"user"`.
On the first message that *is* a user message it evaluates `_has_image_in_blocks(content)` (only
when `content` is a `list`) and then executes an unconditional `break`, so the loop never reaches
any earlier user message. If that most-recent user message contains no image block, the function
returns `False` even when an older user message in the same body does contain one.

Relevant source (buster_ripper/image.py:30-44):

```python
messages = body.get("messages", [])
if not messages:
    return False

# Check last user message for images
for msg in reversed(messages):
    if msg.get("role") != "user":
        continue

    content = msg.get("content")
    if isinstance(content, list) and _has_image_in_blocks(content):
        return True
    break

return False
```

The `break` sits after the `if`, not inside it, so it fires whether or not an image was found —
that is what makes the scan stop at one user message rather than continuing to the next one back.

## Source
/Users/alexhubitski/Projects/buster-ripper/buster_ripper/image.py — `has_images`, lines 25-44
(`break` at line 42). Production source only; no tests consulted (P10).

## Source Version
worktree source, no version stamp in file; fingerprinted below (revision-independent basis)

## Source Fingerprint
sha256:611df5acd7bf061242084de992a7d84102e0645a5c4b023e0d350841aa32dca7

## Invalidates On
- The `break` on line 42 is moved inside the `if` (or removed), so the loop keeps scanning
  older user messages after a non-image user message.
- `has_images` is rewritten to iterate `messages` forward, or to inspect all user messages /
  all messages rather than the last user message.
- The role filter changes (e.g. accepting `assistant` too, or not skipping non-user messages),
  which changes what "first user message it meets" means.
- `_image_cache`-based detection replaces message scanning as the source of truth for images.

## Checked At
2026-10-05
