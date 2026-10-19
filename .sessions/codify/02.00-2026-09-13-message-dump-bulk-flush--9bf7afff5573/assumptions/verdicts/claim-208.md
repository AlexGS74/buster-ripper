VERDICT: DOES-NOT-HOLD
PROMOTION: INELIGIBLE — establishing source is fetched documentation reached through a proxied rendering; the consulted bytes (RFC 9110 §7.6.1 as returned by the fetch, sha256 below) are not exactly recoverable from the canonical locator, so re-verification could not reproduce them.

## Claim
The eight names in `utils._HOP_BY_HOP` are the hop-by-hop header fields RFC 9110 §7.6.1 defines

## What it actually does
RFC 9110 §7.6.1 is titled "Connection" and defines **no set of hop-by-hop header fields at all** — the term appears exactly once in the whole of RFC 9110 (the declarative hop-by-hop/end-to-end distinction the `Connection` field expresses). The only field enumeration in §7.6.1 is a five-item, explicitly non-exhaustive list of fields intermediaries `SHOULD` remove or replace before forwarding — `Proxy-Connection`, `Keep-Alive`, `TE`, `Transfer-Encoding`, `Upgrade` — introduced by "This includes but is not limited to". `utils._HOP_BY_HOP` (8 members) shares 4 names with that list (`keep-alive`, `te`, `transfer-encoding`, `upgrade`), omits `Proxy-Connection`, and adds `connection`, `proxy-authenticate`, `proxy-authorization`, `trailers`, none of which §7.6.1 lists. Independently, the sixth entry `trailers` is not a field name RFC 9110 defines: §6.6.2 defines the field `Trailer`, while `trailers` is a keyword *value* of `TE` (§6.5.1 / §10.1.4).

Two secondary checks, same source: RFC 7230 §6.1 (the predecessor `Connection` section, now obsoleted) likewise contains no hop-by-hop enumeration — "hop-by-hop" occurs twice in that document, neither in a list — so the eight-member set does not come from either `Connection` section. And RFC 9110's hop-by-hop model is declarative (field names carried in the `Connection` value), not a fixed registry, so no section of RFC 9110 defines a closed eight-name set.

## Source
`buster_ripper/utils.py:12-15` (production source: `_HOP_BY_HOP = frozenset([...])`) and RFC 9110 §7.6.1 "Connection", https://www.rfc-editor.org/rfc/rfc9110.txt (fetched rendering lines 2573-2637; §6.6.2 at line 2338, TE "trailers" keyword at line 2250). Cross-checked against RFC 7230 §6.1, https://www.rfc-editor.org/rfc/rfc7230.txt. Section identity independently corroborated by the IANA HTTP Field Name Registry (`Connection` → RFC 9110 §7.6.1).

## Source Version
RFC 9110 (June 2022, STD 97); RFC 7230 (June 2014, obsoleted). Repo file at working-tree state, no version identifier consulted.

## Source Fingerprint
`buster_ripper/utils.py` — sha256:dd60987f7ef0741e55712e0469e854500a80318dedc70c130645ccb51bbc0d00
Fetched RFC 9110 rendering (advisory only, not byte-recoverable from the canonical URL) — full doc sha256:ad3b38b7806783d5066714f7ac9aadcba8cec1605a400c7380173737a8adf902; §7.6.1 region sha256:64f3e45b82e9c5324f42f253f8120ffa21b04293c860c47baca879d7265ce5f4

## Invalidates On
- RFC 9110 is updated or errata'd such that §7.6.1 gains an exhaustive hop-by-hop field list (an update would be a new RFC, not a change to these bytes).
- A different spec (e.g. RFC 9112 "HTTP/1.1", or the IANA Field Name Registry) is taken as the referent for "the hop-by-hop header fields" instead of RFC 9110 §7.6.1.
- `_HOP_BY_HOP` in `buster_ripper/utils.py` changes membership or spelling — detectable by fingerprint mismatch on the sha256 above.

## Checked At
2026-10-05T18:10:24Z
