VERDICT: HOLDS
PROMOTION: INELIGIBLE — the verdict is an absence across the whole `buster_ripper/` package (12 files), not something one exact source locator establishes; the only bytes actually read (`buster_ripper/image.py`) cannot detect a caller later added to another module, so promotion keyed to that single fingerprint would understate staleness risk.

## Claim
`clear_session_cache` has no caller under `buster_ripper/`

## What it actually does
`clear_session_cache(session_id)` (`buster_ripper/image.py:182`) deletes a session's whole image-cache
entry from the module-level `_image_cache`. Within the named surface it is declared and never
invoked: a bounded sweep of `buster_ripper/` for the identifier returns exactly one hit — the `def`
line itself — so there are zero call sites, and no other spelling of that name (no
`_clear_session_cache`, no re-export) exists in the package.

Evidence:

| Surface | Occurrences of `clear_session_cache` | Call sites under the claim's scope |
|---|---|---|
| `buster_ripper/` | 1 — `image.py:182`, the definition | 0 |

Ruled out as hidden callers: no `__all__`, no star-import, no `getattr`/`globals()`/`vars()`/`dir()`/
`importlib` dispatch anywhere in `buster_ripper/` (bounded search returned zero hits);
`buster_ripper/__init__.py` contains only a module docstring. The package imports `image` in exactly
two places, both as the alias `image_handler`, and the complete set of attribute calls through that
alias is `has_images` and `strip_and_cache_images` (`normalize.py:212-213`,
`routes/messages.py:38-39`). Neither of image.py's two per-session/global entry points for dropping
cached images is reached from production code; the module's own deleter path in production is
TTL/LRU eviction (`_cleanup_expired_images`, `_evict_by_lru`), not explicit clearing.

Scope boundary, stated so the verdict is not over-read: the claim names only `buster_ripper/`, and
inside it nothing calls this function. Outside that scope, the function is invoked only from the
test suite (`tests/test_image.py:260`, `:266`). That is reported as a boundary observation, not as
evidence for or against the claim — per P10 the absence is grounded in production source alone. It
does carry a finding worth passing on: if any invariant "a session's cached images are cleared" is
expected to hold at runtime, that invariant currently lives in the proof rather than in the
implementation, because no production path establishes it.

## Source
`/Users/alexhubitski/Projects/buster-ripper/buster_ripper/image.py` (read in full, 185 lines;
definition at lines 182-185), with the package-wide absence established by a bounded ripgrep sweep
of `buster_ripper/` using `--hidden --no-ignore` and excluding `__pycache__`, `.git`, and `*.pyc`.
Nothing under `.sessions/` was read; session files matched by the incidental repo-wide sweep were
excluded as non-evidence per P2.

## Source Version
Working-tree file, no installed-package version applies; content-addressed below.

## Source Fingerprint
sha256:611df5acd7bf061242084de992a7d84102e0645a5c4b023e0d350841aa32dca7 (`buster_ripper/image.py`, 185 lines)

## Invalidates On
- Any new reference to the exact identifier `clear_session_cache` under `buster_ripper/` — a direct
  call, an `image_handler.clear_session_cache(...)` call, an alias/`__all__` export, or a dispatch
  string.
- Any change to `buster_ripper/image.py` lines 141-185 that makes `get_cached_image` or another
  production entry point clear a session's cache (i.e. the function becoming reachable through a
  renamed or inlined path).
- Introduction of name-based dispatch (`getattr`, scheduler/registry, plugin scan) anywhere in the
  package, which would make static identifier search insufficient to prove absence.
- Re-scoping the claim to include `tests/`, where the function is called twice — that variant of the
  proposition is false.

## Checked At
2026-10-05T18:56:35Z
