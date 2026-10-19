VERDICT: HOLDS
PROMOTION: ELIGIBLE

## Claim
`buster_ripper_legacy.py`'s `_forward_headers` applies no rule to an `Authorization` header

## What it actually does
The function body is one dict comprehension whose only rule is the exclusion test
`k.lower() not in ("host", "content-length", *_HOP_BY_HOP)`; `authorization` is not among
the rejected names, so an incoming `Authorization` header is copied through unchanged —
name spelling and value both. No rewriting, validation, casing change, or conditional
removal exists anywhere in the function. `proxy-authorization` (line 183) is a member of
`_HOP_BY_HOP`, but it is a distinct header field name, not a rule about `Authorization`.

For the reader's benefit, and as distinct from this verdict's subject: one *caller* does
apply a rule after `_forward_headers` has already returned — lines 838-841 pop
`headers["authorization"]` when the value is empty/invalid ("Model-agnostic: strip
empty/invalid Authorization header"). That mutates the returned dict in the route handler,
not inside the function the claim names, so it does not contradict the claim as written.
The claim is scoped to `_forward_headers`, and that scoping is unambiguous: exactly one
definition of `_forward_headers` exists in the file (line 621) and its body is lines 623-627.

## Source
/Users/alexhubitski/Projects/buster-ripper/buster_ripper_legacy.py — lines 621-627
(`_forward_headers`), lines 178-189 (`_HOP_BY_HOP` members, to confirm no
`authorization` entry), lines 812 and 838-841 (the separate caller-level handling, checked
to confirm it is downstream of the function rather than part of it).

## Source Version
unavailable — no version/revision marker in the source file; identity is carried by the
byte fingerprint below rather than a version string.

## Source Fingerprint
sha256:7967563dc3ab4f66f0dc322c7945c5647b96c9f9a77fbafe3cd1d4780e8764bd

## Invalidates On
- Adding `"authorization"` (or any name it would match) to the exclusion tuple at line 626.
- Adding any second statement to the `_forward_headers` body that inspects, rewrites, or
  conditionally removes an `Authorization` value.
- Hoisting the caller-side strip at lines 838-841 into `_forward_headers`, which would make
  the function apply a rule where it currently applies none.
- Replacing the plain dict return with a helper that normalizes auth (e.g. re-writing
  `Bearer`/`x-api-key` schemes) before returning.

## Checked At
2026-10-05
