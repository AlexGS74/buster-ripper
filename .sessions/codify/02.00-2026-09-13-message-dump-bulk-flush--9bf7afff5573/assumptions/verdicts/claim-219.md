VERDICT: HOLDS
PROMOTION: ELIGIBLE

## Claim
`buster_ripper_legacy.py`'s `_HOP_BY_HOP` names the same eight header values as `buster_ripper/utils.py`'s `_HOP_BY_HOP`

## What it actually does
Both modules define `_HOP_BY_HOP` as a `frozenset` of exactly eight identical, lowercase HTTP header names — `connection`, `keep-alive`, `proxy-authenticate`, `proxy-authorization`, `te`, `trailers`, `transfer-encoding`, `upgrade` — so the two constants are set-equal. The legacy definition (`buster_ripper_legacy.py:178-189`) spells the eight values one per line; the package definition (`buster_ripper/utils.py:12-15`) packs them onto three lines, which is a formatting difference only.

Verified by enumerating both literals member-by-member (count = 8 in each, no extra or missing member in either direction), not by reading the comment above either one. Neither file defines a second `_HOP_BY_HOP`; a repo-wide case-sensitive grep for the identifier returns these two definitions and uses only.

One adjacent fact that is NOT part of this claim and does not change the verdict: `buster_ripper/utils.py:23` repeats `"connection"` and `"transfer-encoding"` literally in `forward_headers`'s exclusion tuple even though both are already members of `_HOP_BY_HOP`, and legacy `buster_ripper_legacy.py:626` does the same for `"connection"`/`"transfer-encoding"`. That is redundancy at the use sites, not a divergence in the constants.

## Source
- `/Users/alexhubitski/Projects/buster-ripper/buster_ripper/utils.py` lines 12-15 (package constant)
- `/Users/alexhubitski/Projects/buster-ripper/buster_ripper_legacy.py` lines 178-189 (legacy constant)

Grounded in production source under the supplied source root. `tests/test_header_parity.py` also asserts this equality and corroborates it, but per P10 it established nothing here; the verdict is unchanged if that test file is removed.

## Source Version
unavailable — no version/revision metadata consulted (Git metadata is out of scope for this role); the two files' exact bytes are fingerprinted below instead.

## Source Fingerprint
- `sha256:76de98c1bd07e1cf3b2852d8dd64dd861160f54fe40d2781fce1ad66ae913c72` — `buster_ripper/utils.py`
- `sha256:7967563dc3ab4f66f0dc322c7945c5647b96c9f9a77fbafe3cd1d4780e8764bd` — `buster_ripper_legacy.py`

## Invalidates On
- Either fingerprint changes: a header name added to, removed from, or renamed in either `_HOP_BY_HOP` literal breaks set-equality.
- Either constant is replaced by an import from a shared module — the claim ("legacy's names the same eight as utils's") then describes one definition, not two.
- Case or whitespace normalization is introduced in either literal (e.g. `Connection` instead of `connection`), making the sets unequal as written even though the header names "match".
- `buster_ripper_legacy.py` is deleted or stops being a source of truth for hop-by-hop filtering.

## Checked At
2026-10-06T12:15:55Z
