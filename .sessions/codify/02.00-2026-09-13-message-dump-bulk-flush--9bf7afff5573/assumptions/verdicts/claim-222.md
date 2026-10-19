VERDICT: HOLDS
PROMOTION: ELIGIBLE

## Claim
`buster_ripper_legacy.py`'s `_response_headers` forwards a `content-length` header it receives

## What it actually does
The function body is one dict comprehension whose only rule is
`k.lower() not in _HOP_BY_HOP`, and `content-length` is not a member of that frozenset —
its eight members are `connection`, `keep-alive`, `proxy-authenticate`,
`proxy-authorization`, `te`, `trailers`, `transfer-encoding`, `upgrade`. So a
`content-length` present in `upstream_headers` is copied into the returned dict unchanged,
with the upstream's value. This is a genuine asymmetry with the request direction:
`_forward_headers` names `"content-length"` explicitly in its exclusion tuple (line 626),
whereas `_response_headers` never mentions it (its docstring says only "Strip hop-by-hop
headers from upstream response"). Whether a forwarded value matches the body actually
served is a run-time question outside this role; the claim as written — that the header is
forwarded rather than dropped — is settled by the source.

## Source
/Users/alexhubitski/Projects/buster-ripper/buster_ripper_legacy.py — lines 630-636
(`_response_headers` definition), lines 178-189 (`_HOP_BY_HOP` members, to confirm
`content-length` is absent), and line 626 contrasted as the request-side rule. Exactly one
definition of `_response_headers` exists in the file (line 630). Corroborated, not
established, by call sites: line 882-883 wraps the result in `dict(...)` and then sets
`resp_headers["content-length"]` itself, while lines 718, 891 and 913 pass
`_response_headers(resp.headers)` straight through with no such overwrite.

## Source Version
unavailable — no version/revision marker in the source file; identity is carried by the
byte fingerprint below rather than a version string.

## Source Fingerprint
sha256:7967563dc3ab4f66f0dc322c7945c5647b96c9f9a77fbafe3cd1d4780e8764bd

## Invalidates On
- Adding `"content-length"` to the `_HOP_BY_HOP` frozenset at lines 178-189 (would drop it
  in the response direction, and simultaneously change what `_forward_headers` rejects).
- Adding a second exclusion term to the line 635 test, or an explicit
  `resp_headers.pop("content-length", None)` in the function body.
- Changing the line 635 comparison from `k.lower()` to case-sensitive `k`, which would let
  non-lowercase spellings through while dropping none additionally — narrow but real.
- Replacing `httpx.Headers` with a headers object whose `.items()` already omits
  `content-length`, since then the function would never receive one to forward.

## Checked At
2026-10-05
