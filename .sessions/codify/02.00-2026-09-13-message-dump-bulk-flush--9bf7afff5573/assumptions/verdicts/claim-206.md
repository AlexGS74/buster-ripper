VERDICT: DOES-NOT-HOLD
PROMOTION: ELIGIBLE

## Claim
`utils.response_headers` drops exactly the names in `utils._HOP_BY_HOP` and no others

## What it actually does
It drops the eight names in `_HOP_BY_HOP` **plus `content-length`** — the exclusion
set is not exactly `_HOP_BY_HOP`. The dict comprehension at `utils.py:36-40` filters
on the tuple `("content-length", *_HOP_BY_HOP)`, so the set of dropped names is
`_HOP_BY_HOP ∪ {"content-length"}`, and the docstring at `utils.py:28-34` states the
same intent ("dropping hop-by-hop and content-length").

Concretely, the names `response_headers` omits are:
`connection`, `keep-alive`, `proxy-authenticate`, `proxy-authorization`, `te`,
`trailers`, `transfer-encoding`, `upgrade` (the `_HOP_BY_HOP` frozenset, `utils.py:12-15`),
and additionally `content-length`. Matching is case-insensitive on incoming keys
(`k.lower()`), and no name outside that nine-name set is dropped.

`MALFORMED per P3` did not apply: the claim is a single assertion about one function,
answered wholly from one expression. `P9` does not apply either — no reading of
"exactly … and no others" makes the claim true, since `content-length` is unambiguously
a name that is not in `_HOP_BY_HOP` and is nonetheless dropped. This is not a claim only
a run could settle (`P1`): the filter predicate is written out in the source.

Note that `_HOP_BY_HOP` is also used by `forward_headers` (`utils.py:23`), whose exclusion
set is larger again (`host`, `content-length`, `connection`, `transfer-encoding` plus the
frozenset). That sibling does not change this verdict; the claim names `response_headers`.

## Source
`/Users/alexhubitski/Projects/buster-ripper/buster_ripper/utils.py` — `response_headers`,
lines 27-40 (exclusion tuple at line 39); `_HOP_BY_HOP` definition, lines 12-15.

## Source Version
Local repository source; supplied as `Source root` for this dispatch. No package version
applies (first-party module). Whole 133-line file read.

## Source Fingerprint
sha256:76de98c1bd07e1cf3b2852d8dd64dd861160f54fe40d2781fce1ad66ae913c72

## Invalidates On
- Line 39's exclusion tuple changes (e.g. `content-length` removed, or any further name added).
- The `_HOP_BY_HOP` frozenset at lines 12-15 changes membership.
- A second definition or reassignment of `response_headers` appears in `buster_ripper/utils.py`
  (currently defined once, at line 27).
- `utils.py` bytes change from the fingerprint above.

## Checked At
2026-10-06T12:15:52Z
