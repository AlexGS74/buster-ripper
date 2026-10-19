VERDICT: HOLDS
PROMOTION: ELIGIBLE

## Claim
`buster_ripper/dump.py`'s `_sessions` mapping is never pruned by expiry or by size

## What it actually does
Inside the subject source, `_sessions` only ever grows: it is declared at line 46 and appears at
exactly three other sites — `for state in _sessions.values()` (line 110, read-only iteration in
`flush_all`), `_sessions.get(sid)` (line 139) and `_sessions[sid] = state` (line 145, the only
write). There is no `del`, `pop`, `popitem`, `clear`, or reassignment of the mapping anywhere in
the file, so a session entry added by `_lookup_or_create_state` lives for the whole process
lifetime.

Neither pruning axis exists in the code, not merely the idiom:
- **expiry**: `SessionState` (lines 32-42) carries `turn`, `prev_text`, `session_dir`, `pending`,
  `pending_bytes`, `fmt` — no `last_seen`/`created_at`/age field, so nothing could compare a
  session against a TTL. `_now()` (line 246) reads wall clock only to format summary lines; its
  result is never stored on state. No TTL/expiry setting is referenced — the only `config` reads
  in the file are `DUMP_DIR`, `DUMP_FORMAT`, `DUMP_FORMATS`, `DUMP_FLUSH_INTERVAL` and
  `DUMP_FLUSH_BYTES`.
- **size**: no maximum-sessions constant, no `len(_sessions)` guard at the insertion site, and no
  bounded container (`OrderedDict`/`move_to_end`/`maxlen`) — the mapping is a plain `dict`.

The one `.clear()` in the file, `state.pending.clear()` at line 293, empties a session's buffered
record list inside `_flush_state` and resets `pending_bytes` to 0 (line 294). It drains the
buffer *within* an entry; it does not remove the entry from `_sessions`. `flush_all` similarly
visits every state and leaves all of them in place.

## Source
/Users/alexhubitski/Projects/buster-ripper/buster_ripper/dump.py — read in full (305 lines);
declaration line 46, insertion line 145, read-only sites lines 110 and 139; `SessionState`
lines 32-42; buffer-only drain lines 284-294.

Scope note: the supplied locator was this single file, so the verdict covers the subject source.
`_sessions` is module-private by name and no pruning exists where the mapping is owned; a
cross-module `dump._sessions` mutation would require the package source root, which was not
supplied and was not searched.

## Source Version
buster_ripper/dump.py, 305 lines, 11472 bytes (mtime 2026-09-27 18:32); no version stamp in source

## Source Fingerprint
sha256:07b0b076552f72e62c625caade1b01821b3237375900233812d14c0f275b3c2b

## Invalidates On
- A `del`/`pop`/`popitem` of `_sessions` keys, or any reassignment of `_sessions`, is added to dump.py
- `SessionState` gains a time/age field and a sweep, or dump.py gains a max-sessions bound or bounded container
- Lines 110/139/145 or the 32-42 dataclass change in the file whose bytes hash to the fingerprint above
- The mapping's ownership moves out of dump.py (e.g. state moves to `session.py`), making this file non-authoritative

## Checked At
2026-10-05T19:03:59Z
