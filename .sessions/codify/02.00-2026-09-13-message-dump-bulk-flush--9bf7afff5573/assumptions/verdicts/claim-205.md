VERDICT: HOLDS
PROMOTION: ELIGIBLE

## Claim
`utils.forward_headers` drops `host`, `content-length` and every name in `utils._HOP_BY_HOP`

## What it actually does
`forward_headers` returns a dict comprehension that keeps a header only when
`k.lower()` is not in the literal tuple
`("host", "content-length", "connection", "transfer-encoding", *_HOP_BY_HOP)`.
The splat expands every member of `_HOP_BY_HOP` into that tuple, and `"host"` and
`"content-length"` are listed explicitly, so the drop set is exactly
`{"host", "content-length"} ∪ _HOP_BY_HOP` (`"connection"` and
`"transfer-encoding"` are named twice — once literally, once via the splat —
which is redundant but does not change the set). Keys are compared after
`.lower()`, so the drop is case-insensitive on the incoming header name.

Verified by enumeration against the constant's contents read independently from
its own definition (see claim-207 for that check): `connection`, `keep-alive`,
`proxy-authenticate`, `proxy-authorization`, `te`, `trailers`,
`transfer-encoding`, `upgrade` — all eight appear in the exclusion tuple either
literally or via `*_HOP_BY_HOP`, and `host` / `content-length` appear literally.

Note on scope: the claim asserts these names ARE dropped. It does not assert
nothing else is dropped, and in fact nothing else is — the tuple contains no
further names.

## Source
/Users/alexhubitski/Projects/buster-ripper/buster_ripper/utils.py — `forward_headers`, lines 18-24 (defn) and `_HOP_BY_HOP`, lines 12-15

## Source Version
Local repo source; no package version. File is 4277 bytes, SHA-256
`dd60987f7ef0741e55712e0469e854500a80318dedc70c130645ccb51bbc0d00`.

## Source Fingerprint
sha256:dd60987f7ef0741e55712e0469e854500a80318dedc70c130645ccb51bbc0d00

## Invalidates On
- Line 23's exclusion tuple is edited to remove or rename any of `"host"`, `"content-length"`, or the `*_HOP_BY_HOP` splat
- `_HOP_BY_HOP` (lines 12-15) changes membership, changing what `forward_headers` drops via the splat
- `forward_headers` stops lowercasing the key before comparison (drop becomes case-sensitive)
- The header-collection type changes so `.items()` no longer yields plain `str` keys

## Checked At
2026-10-05
