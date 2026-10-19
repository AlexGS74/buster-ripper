VERDICT: HOLDS
PROMOTION: ELIGIBLE

## Claim
`buster_ripper_legacy.py`'s `_forward_headers` drops `host`, `content-length` and every member of its `_HOP_BY_HOP`

## What it actually does
`_forward_headers` builds its upstream dict with a single comprehension whose exclusion
tuple is `("host", "content-length", *_HOP_BY_HOP)` — the star-unpacking expands all eight
members of the module-level frozenset (`connection`, `keep-alive`, `proxy-authenticate`,
`proxy-authorization`, `te`, `trailers`, `transfer-encoding`, `upgrade`) into the same
rejection set as `host` and `content-length`. Every header name the claim lists is therefore
dropped, and there is no other code path in the function — the comprehension is its entire
body. The docstring ("dropping hop-by-hop and host") under-states the behaviour, since
`content-length` is dropped too; the code, not the docstring, is what the claim is about.
Comparison is `k.lower() not in (...)`, so the drop is case-insensitive on the incoming name.

## Source
/Users/alexhubitski/Projects/buster-ripper/buster_ripper_legacy.py — lines 621-627
(`_forward_headers` definition, including the `*_HOP_BY_HOP` unpack) and lines 178-189
(the `_HOP_BY_HOP` frozenset literal whose eight members were enumerated above).
Exactly one definition of `_forward_headers` exists in the file.

## Source Version
unavailable — no version/revision marker in the source file; identity is carried by the
byte fingerprint below rather than a version string.

## Source Fingerprint
sha256:7967563dc3ab4f66f0dc322c7945c5647b96c9f9a77fbafe3cd1d4780e8764bd

## Invalidates On
- Any edit to the exclusion tuple at line 626 (removing `"host"` or `"content-length"`, or
  dropping the `*_HOP_BY_HOP` unpack in favour of an explicit subset).
- Any edit to the `_HOP_BY_HOP` members at lines 178-189 that adds, renames, or removes an
  entry — the claim is about *every* member, so the set's contents are part of what was
  verified, not just its name.
- Replacing the star-unpack with a two-stage filter (e.g. dropping hop-by-hop only in a
  caller) so that `_forward_headers` itself no longer performs all the drops.
- Changing the comparison from `k.lower()` to case-sensitive `k`, which would narrow which
  incoming spellings actually get dropped.

## Checked At
2026-10-05
