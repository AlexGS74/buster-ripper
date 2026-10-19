VERDICT: HOLDS
PROMOTION: ELIGIBLE

## Claim
`_has_image_in_blocks` reports an image only for a mapping block whose `type` is `image`

## What it actually does
The function returns `True` on the first element of `content` that satisfies both
`isinstance(block, dict)` and `block.get("type") == "image"`; every other element — a plain
string content part, a dict whose `type` is anything else, or a dict with no `type` key at all —
is ignored. So a positive report requires a `dict` (hence a mapping) block with `type` exactly
equal to the case-sensitive string `"image"`. Nothing else in the block is consulted: presence of
a `source` dict, `base64` data, or a media type has no effect, and OpenAI-style `"image_url"` is
not treated as an image.

Relevant source (buster_ripper/image.py:17-22):

```python
def _has_image_in_blocks(content: list) -> bool:
    """Check if content list contains an image block."""
    for block in content:
        if isinstance(block, dict) and block.get("type") == "image":
            return True
    return False
```

Reading note (why not UNCLEAR under P9): the claim is restrictive ("only for"), i.e. it states a
necessary condition, and that condition is plainly met. It does *not* claim sufficiency for every
mapping kind — a mapping that is not a `dict` instance (e.g. `types.MappingProxyType`, or a
custom `collections.abc.Mapping`) with `type == "image"` would *not* be reported, because the
guard is `isinstance(block, dict)`, not an ABC check. That edge is beyond what the claim asserts,
and naming it here rather than converting a true claim into a false one.

## Source
/Users/alexhubitski/Projects/buster-ripper/buster_ripper/image.py — `_has_image_in_blocks`,
lines 17-22; same predicate duplicated in `_process_content_block` line 98. Production source
only; no tests consulted (P10).

## Source Version
worktree source, no version stamp in file; fingerprinted below (revision-independent basis)

## Invalidates On
- The `type` comparison changes (e.g. accepting `"image_url"`, `"input_image"`, a set of types,
  or a case-insensitive/falsy check).
- The guard changes from `isinstance(block, dict)` to a `collections.abc.Mapping` check or to
  duck-typed `block.get`, which makes non-dict mappings report images too.
- Detection is widened to structural keys (`"source" in block`) or narrowed/short-circuited
  differently, so a `type == "image"` dict no longer implies a positive report.

## Checked At
2026-10-05
