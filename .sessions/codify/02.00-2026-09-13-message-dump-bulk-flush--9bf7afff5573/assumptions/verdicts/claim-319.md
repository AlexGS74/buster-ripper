VERDICT: HOLDS
PROMOTION: ELIGIBLE

## Claim
`buster_ripper/session.py`'s `_stats` mapping is never pruned by expiry or by size

## What it actually does
The module-level `_stats: dict[str, SessionStats]` (session.py:14) is only ever added to. Every write site across the production package is an insertion or a bulk union — `_stats[sid] = stats` in `update_session_stats` (session.py:79) and `session._stats.update(session.db_load(session._db_conn))` at startup (main.py:158) — and every other reference is a read (`_stats.get` in session.py:76, `routes/count_tokens.py:45`, `main.py:160`). There is no `del`, `pop`, `clear`, rebinding, TTL check, size cap, or LRU eviction targeting `_stats` anywhere; entries accumulate for the process lifetime, keyed by a 12-hex-char hash of each distinct system prompt. The repo does contain expiry- and size-based pruning (`image.py` `_cleanup_expired_images` / `_evict_by_lru`), but it applies exclusively to `_image_cache`, not to `_stats`. The backing SQLite table `session_stats` likewise has no DELETE path in the source.

## Source
- Primary: `/Users/alexhubitski/Projects/buster-ripper/buster_ripper/session.py` (full read, lines 1–98)
- Exhaustive-membership corroboration: package-scoped searches for `_stats` and for deletion/eviction primitives (`del`, `.pop(`, `.clear(`, `prune|evict|expir|ttl|LRU|maxsize`, case-insensitive) over `/Users/alexhubitski/Projects/buster-ripper/buster_ripper/**/*.py`, plus a wildcard-import check (no `import *` found) — all matches accounted for; none prune `_stats`.

## Source Version
Working-tree source at time of check (no revision metadata consulted, per git-metadata boundary)

## Source Fingerprint
sha256:67a66b8aa733b31e982be4922350a7996ebdddb0fe5d391a6129a1cf342a9dff (buster_ripper/session.py)

## Invalidates On
- Any edit to session.py changing the fingerprint that introduces a deletion, eviction, expiry, or size bound on `_stats`
- New code elsewhere in the package (or a wildcard/re-export of `_stats`) that `del`/`pop`/`clear`s or rebinds `session._stats`
- A change giving `SessionStats` records their own lifecycle pruning (e.g., session teardown that removes stats entries)

## Checked At
2026-10-05
