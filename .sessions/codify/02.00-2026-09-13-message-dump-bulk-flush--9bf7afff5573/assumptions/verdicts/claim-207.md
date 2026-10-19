VERDICT: HOLDS
PROMOTION: ELIGIBLE

## Claim
`utils._HOP_BY_HOP` is a frozenset naming `connection`, `keep-alive`, `proxy-authenticate`, `proxy-authorization`, `te`, `trailers`, `transfer-encoding` and `upgrade`

## What it actually does
Module-level `_HOP_BY_HOP = frozenset([...])` over a list literal of exactly eight
lowercase string members, in source order: `connection`, `keep-alive`,
`proxy-authenticate`, `proxy-authorization`, `te`, `trailers`,
`transfer-encoding`, `upgrade`. Type is `frozenset` (constructed by the
`frozenset(...)` call, not a set literal or tuple), it is immutable, and the
membership matches the claim one-for-one with no extra and no missing name.

Checked against the definition itself (lines 12-15), independently of the two
call sites that use it. Every claimed name is present and every present name is
claimed; there are no duplicates in the list literal, so the eight entries
correspond to eight distinct members. All members are lowercase, which is what
makes the `k.lower() in _HOP_BY_HOP` style comparison at the call sites work.

## Source
/Users/alexhubitski/Projects/buster-ripper/buster_ripper/utils.py — `_HOP_BY_HOP`, lines 12-15

## Source Version
Local repo source; no package version. File is 4277 bytes, SHA-256
`dd60987f7ef0741e55712e0469e854500a80318dedc70c130645ccb51bbc0d00`.

## Source Fingerprint
sha256:dd60987f7ef0741e55712e0469e854500a80318dedc70c130645ccb51bbc0d00

## Invalidates On
- Any name added to or removed from the list literal on lines 13-14 (membership no longer matches the eight claimed names)
- The `frozenset(...)` wrapper replaced by a `set` literal, tuple, or list (type claim fails)
- Any member re-cased to mixed case (breaks the lowercase-membership assumption at the call sites)
- The constant renamed, moved to another module, or re-exported from `utils`'s imports

## Checked At
2026-10-05
