VERDICT: HOLDS
PROMOTION: ELIGIBLE

## Claim
Python's built-in `str` type has no `get` attribute

## What it actually does
`str` exposes the common sequence operations plus a fixed set of 46 documented methods
(`center`, `count`, `encode`, … `zfill`); none of them is `get`. `get` exists in the same
document only as a `dict` method (Mapping type), so `somestr.get(...)` raises
`AttributeError` and `isinstance(x, str) and hasattr(x, "get")` never overlaps for plain
built-in `str`.

Evidence located in the CPython 3.11.9 standard-types document:
- `Text Sequence Type --- :class:str` label `.. _textseq:` at line 1468.
- `String Methods` at line 1562: "Strings implement all of the common
  `<typesseq-common>` sequence operations, along with the additional methods described below."
  The section declares 46 `.. method:: str.*` directives; zero match `get`.
- Only one `.. method:: get(key[, default])` directive in the file, at line 4520, nested
  under `Mapping Types --- :class:dict` (line 4370) — i.e. `dict.get`, not `str.get`.
- No `str.get` / `:meth:get` string-method reference anywhere in the file.

Corroboration (secondary, not establishing): the repo's own `pyproject.toml:4` pins
`requires-python = ">=3.11"`, matching the 3.11 documentation consulted.

## Source
https://raw.githubusercontent.com/python/cpython/v3.11.9/Doc/library/stdtypes.rst
(rendered equivalent: https://docs.python.org/3.11/library/stdtypes.html#textseq and
#string-methods) — fetched complete, 213883 bytes, inspected with grep/sed over the
`textseq`, `String Methods`, and `Mapping Types` sections.

## Source Version
CPython tag v3.11.9 documentation source (`Doc/library/stdtypes.rst`); project requires
`>=3.11`. `str.get` has never existed in any released CPython 3.x.

## Source Fingerprint
sha256:397bf1f91d0515c1bc26e5c52d64efd6079d0be0589945cac4af6eec5245e5c6

## Invalidates On
- Any CPython release whose `Doc/library/stdtypes.rst` adds a `str.get` method directive to
  the `String Methods` / `textseq` sections (verified by re-hashing that file at the new tag).
- Reading the claim as being about a `str` *subclass* or an object with a custom
  `__getattr__` rather than the built-in type — outside the claim's wording, so re-extract
  rather than reuse this evidence.
- Reusing this verdict for a different Python major line (e.g. 2.x, where `unicode`/`str`
  differ) — this fingerprint is pinned to v3.11.9.

## Checked At
2026-10-05T14:28:40-04:00
