VERDICT: HOLDS
PROMOTION: ELIGIBLE

## Claim
`session_id` derives its value from the first twelve characters of a `hashlib.sha256` hexdigest

## What it actually does
`session_id(body)` returns `hashlib.sha256(system.encode()).hexdigest()[:12]` — the full
SHA-256 hex digest of the request's system text (a `str`, or the space-joined `text` fields
when `system` is a list of dicts), sliced to its first 12 characters. Nothing else contributes
to the returned value: no prefix/suffix, no truncation at a different length, and no other
digest algorithm.

## Source
/Users/alexhubitski/Projects/buster-ripper/buster_ripper/session.py — `def session_id(body: dict) -> str`, lines 93–98 (return expression on line 98); `import hashlib` on line 4.

## Source Version
unavailable — repo production source read as working bytes; no package version or VCS
revision consulted (git metadata is out of scope for this role). Stability is carried by the
fingerprint below.

## Source Fingerprint
sha256:67a66b8aa733b31e982be4922350a7996ebdddb0fe5d391a6129a1cf342a9dff

## Invalidates On
- Any change to the return expression in `session_id` (slice length other than `[:12]`, a
  different `hashlib` constructor, digest not in hex form, or a transform applied to the
  digest string).
- A change to which `body` field is hashed, or to the list-form normalization on lines 96–97,
  which alters the input to the digest though not the 12-char-hexdigest shape of the claim.
- Relocation or redefinition of `session_id` (the claim was settled only within the supplied
  source root; a same-named symbol elsewhere was not searched for).

## Checked At
2026-10-05
