VERDICT: HOLDS
PROMOTION: INELIGIBLE — the verdict is an absence over two whole surfaces (`buster_ripper/`, `tests/`); no single exact source locator establishes it, and the only bytes actually read (`buster_ripper/image.py`) cannot detect a caller added to a different file, so a re-check keyed to that one fingerprint would be misleading.

## Claim
`cleanup_expired_images` has no caller under `buster_ripper/` or `tests/`

## What it actually does
The public, zero-argument `cleanup_expired_images()` (declared `buster_ripper/image.py:176`) is
referenced nowhere in either named surface — its `def` line is the only occurrence of that exact
identifier in the package, and the identifier does not occur at all under `tests/`. What *is*
called is the separate, differently-named private helper `_cleanup_expired_images(session_id)`
(`image.py:47`), from two sites: `image.py:155` inside `get_cached_image`, and `image.py:179`
inside `cleanup_expired_images` itself. So the outer janitor is dead code reachable only if
something imports and calls it; nothing under these surfaces does.

Evidence, per named surface:

| Surface | Occurrences of `cleanup_expired_images` | Call sites of the public name |
|---|---|---|
| `buster_ripper/` | 4, all in `image.py` — `:47` def of `_…`, `:155` call of `_…`, `:176` def of the public name, `:179` call of `_…` | 0 |
| `tests/` | 0 (not even the underscore-prefixed helper) | 0 |

Ruled out as hidden callers within `buster_ripper/`: there is no `__all__`, no star-import, no
`getattr`/`globals()`/`vars()`/`dir()`/`importlib` dispatch anywhere in the package (bounded
search returned zero hits), and `buster_ripper/__init__.py` is a docstring only. The module's two
production importers alias it as `image_handler`, and the complete set of attribute uses through
that alias is `has_images` and `strip_and_cache_images` (`normalize.py:212-213`,
`routes/messages.py:38-39`) — neither janitor.

Caveat for the re-checker (does not change the verdict, and is not the reason for it): the claim
name is a prefix of the private helper. Read as an exact identifier, as written, the claim holds.
Read as loose substring/prefix matching that folds `_cleanup_expired_images` into the same unit,
the claim would be false. Those are two distinct functions with distinct signatures
(`() -> None` vs `(session_id: str) -> None`), so the loose reading conflates units; if the
extractor's intent was the helper, this row needs re-extraction naming it explicitly.

## Source
`/Users/alexhubitski/Projects/buster-ripper/buster_ripper/image.py` (lines 47, 155, 176, 179 —
read in full, all 185 lines), with the absence half established by bounded ripgrep sweeps of both
named surfaces (`buster_ripper/`, `tests/`) using `--hidden --no-ignore` and excluding
`__pycache__`, `.git`, and `*.pyc`. `tests/` is a surface named by the claim itself, so its files
are the production surface for that half (P10 exception), not corroboration of a behaviour claim.
Nothing under `.sessions/` was read; session files that matched the repo-wide sweep were excluded
as non-evidence per P2.

## Source Version
Working-tree file, no installed-package version applies; content-addressed below.

## Source Fingerprint
sha256:611df5acd7bf061242084de992a7d84102e0645a5c4b023e0d350841aa32dca7 (`buster_ripper/image.py`, 185 lines)

## Invalidates On
- Any new reference to the exact identifier `cleanup_expired_images` (a call, an alias export, or
  a dispatch string) appearing under `buster_ripper/` or `tests/`.
- Any edit to `buster_ripper/image.py` around lines 150-185 that changes the public/private split
  — in particular if `cleanup_expired_images` gains parameters, is renamed, or is folded into
  `_cleanup_expired_images`.
- Introduction of name-based dispatch (`getattr`, `__all__` re-export, star-import, scheduler
  registry) anywhere in the package, which would make static identifier search insufficient.
- The claim being re-extracted to mean the underscore-prefixed helper `_cleanup_expired_images`
  (or the pair of them), which flips the answer to false.

## Checked At
2026-10-05T18:56:35Z
