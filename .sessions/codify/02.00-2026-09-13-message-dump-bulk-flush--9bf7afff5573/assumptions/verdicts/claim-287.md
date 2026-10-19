VERDICT: HOLDS
PROMOTION: ELIGIBLE

## Claim
`get_cached_image` has no caller under `buster_ripper/`

## What it actually does
Across the entire bounded source root `buster_ripper/` (16 `.py` files, all text), the
identifier `get_cached_image` occurs exactly once — in its own `def` at
`buster_ripper/image.py:141`. Nothing imports it, calls it, re-exports it, or even
mentions it in a string, comment, or docstring, so within the claimed scope it is
unreached production code. No name-based dispatch could hide a caller either: a search
of the same root for `getattr`, `importlib`, `globals()`, `eval(`, `__all__` and
`setattr` returned zero matches, and the `def` is undecorated (preceded only by blank
lines), so nothing registers it into a dispatch table.

## Source
Bounded exhaustive search of the supplied source root
`/Users/alexhubitski/Projects/buster-ripper/buster_ripper` for the literal
`get_cached_image` (`rg -u -n --hidden -g '!.git' -g '!__pycache__' -g '!*.pyc'`),
which returned the single hit `buster_ripper/image.py:141`; corroborating search of the
same root for dynamic-dispatch constructs (`getattr|importlib|globals\(\)|__all__|eval\(|setattr`)
returned no matches. Grounded in production source only — the source root contains no
test root, so no test artifact established or contradicted this (P10 n/a).

## Source Version
Working-tree source at fingerprint below; `source revision: unavailable`
(Git metadata is out of scope for this role — the bytes themselves are fingerprinted).

## Source Fingerprint
consulted file (`buster_ripper/image.py`):
`sha256:611df5acd7bf061242084de992a7d84102e0645a5c4b023e0d350841aa32dca7`

search surface (sha256 of every `.py` under `buster_ripper/`, sorted relative paths,
content-hash then final hash — this is what makes the "no caller" negative exhaustive):
`sha256:7bb0c540a0bb826c075f52c93bad95ed694e8bf019d1dd4f5680e526c9430ee8`

## Invalidates On
- Any new `get_cached_image(...)` call site, import, or reference added under `buster_ripper/`
- Change to any of the 16 fingerprinted `.py` files (surface hash above) — the negative
  is only as exhaustive as that manifest
- Introduction of `getattr`/`importlib`/`globals()`/`__all__` name-based dispatch in
  `buster_ripper/` that could invoke the function without the literal name appearing
- The claim's scope moving beyond `buster_ripper/` (e.g. callers in repo-root scripts,
  `tests/`, or another package) — outside the stated scope and outside the source root
  supplied to this check, therefore deliberately not searched here

## Checked At
2026-10-05
