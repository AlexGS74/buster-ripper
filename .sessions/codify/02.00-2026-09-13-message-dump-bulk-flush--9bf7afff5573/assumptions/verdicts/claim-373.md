VERDICT: HOLDS
PROMOTION: ELIGIBLE

## Claim
`utils.response_headers` drops `content-length` in addition to every name in `_HOP_BY_HOP`

## What it actually does
It does. `response_headers` returns a dict comprehension over `upstream_headers.items()`
whose predicate is `k.lower() not in ("content-length", *_HOP_BY_HOP)`
(`utils.py:36-40`). The `*_HOP_BY_HOP` unpacking expands the frozenset into the exclusion
tuple, so every name in `_HOP_BY_HOP` is excluded and `content-length` is excluded alongside
it — that is precisely "content-length in addition to every name in `_HOP_BY_HOP`".

Members expanded from `_HOP_BY_HOP` (`utils.py:12-15`, a `frozenset` of eight literals):
`connection`, `keep-alive`, `proxy-authenticate`, `proxy-authorization`, `te`, `trailers`,
`transfer-encoding`, `upgrade`. The full dropped set is those eight plus `content-length`,
matched case-insensitively via `k.lower()`; every literal in both the tuple and the frozenset
is already lowercase, so the lowering only affects incoming keys. Nothing in the file
reassigns `response_headers` or rebinds `_HOP_BY_HOP` (each is bound exactly once), and the
docstring at lines 28-34 gives the reason for the extra name — aiohttp re-derives the length
from the body actually sent, so a lying upstream `Content-Length` must not be relayed.

Checked independently of the other claim in this bundle (`P4`): this verdict is derived from
the predicate at line 39 on its own, not by negating that one. Both are settled by the same
single expression, and they are consistent with each other.

## Source
`/Users/alexhubitski/Projects/buster-ripper/buster_ripper/utils.py` — `response_headers`,
lines 27-40 (exclusion tuple at line 39); `_HOP_BY_HOP` definition, lines 12-15.

## Source Version
Local repository source; supplied as `Source root` for this dispatch. No package version
applies (first-party module). Whole 133-line file read.

## Source Fingerprint
sha256:76de98c1bd07e1cf3b2852d8dd64dd861160f54fe40d2781fce1ad66ae913c72

## Invalidates On
- Line 39's exclusion tuple stops including the literal `"content-length"`.
- The `_HOP_BY_HOP` frozenset at lines 12-15 changes membership, changing what "every name
  in `_HOP_BY_HOP`" covers.
- The predicate stops comparing `k.lower()` (e.g. exact-case matching), changing which keys
  are actually dropped.
- `utils.py` bytes change from the fingerprint above.

## Checked At
2026-10-06T12:15:52Z
