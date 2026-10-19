VERDICT: HOLDS
PROMOTION: ELIGIBLE

## Claim
`_normalize_cache_control` removes `cache_control` keys from user message content blocks

## What it actually does
It does delete those keys: `_normalize_cache_control` (line 124) hands `body["messages"]` to
`_strip_cache_control_from_messages` (line 113), which iterates every message and calls
`_strip_cache_control_from_content(msg.get("content"))` (line 100), where `del block["cache_control"]`
runs for each `dict` content block that carries the key. Because line 113-121 applies no role
filter, user messages are covered. Scope worth stating so the verdict is not over- or under-read:
the strip is broader than user messages — it also removes `cache_control` from content blocks of
messages of any other role, from the message dicts themselves (line 118-120), and from system
blocks (lines 128-131); and it is narrower in form — non-list `content` (e.g. a plain string) is
a no-op returning 0, and non-dict blocks are skipped. Contrast `_normalize_date` at line 192,
which *does* gate on `msg.get("role") != "user"` — the cache-control path has no such gate.

## Source
/Users/alexhubitski/Projects/buster-ripper/buster_ripper/normalize.py — `_normalize_cache_control`
lines 124-138; `_strip_cache_control_from_messages` lines 113-121; `_strip_cache_control_from_content`
lines 100-110 (`del` at line 108).

## Source Version
unavailable — the bounded source (single module file) carries no version or revision string;
fingerprint below is the stability basis.

## Invalidates On
- Any edit adding a role filter to `_strip_cache_control_from_messages`, or one that skips
  `"user"` specifically, which would make the claim false as written.
- Any edit replacing `del block["cache_control"]` with a copy-based strip (the key would be
  absent from the returned blocks but the caller's blocks would be unchanged — the observable
  "removes" claim would then depend on where `body["messages"]` is reassigned, which it
  currently is not).
- Any edit that restricts `_normalize_cache_control` to system blocks only (dropping the
  messages branch at lines 133-136).

## Checked At
2026-10-05T18:37:49Z
