VERDICT: HOLDS
PROMOTION: ELIGIBLE

## Claim
`LookupError` is not a subclass of `UnicodeDecodeError`

## What it actually does
`LookupError` is a base class hanging directly off `Exception`, and its only
documented subclasses are `IndexError` and `KeyError`: "The base class for the
exceptions that are raised when a key or index used on a mapping or sequence is
invalid: `IndexError`, `KeyError`. This can be raised directly by
`codecs.lookup()`." `UnicodeDecodeError` sits in a completely separate branch
(`Exception` → `ValueError` → `UnicodeError` → `UnicodeDecodeError`), so
`LookupError` is neither it nor derived from it — the two are unrelated by
subclassing, and the same page states "Two exception classes that are not related
via subclassing are never equivalent." A `except UnicodeDecodeError` clause
therefore does not catch a `LookupError` raised by a codec-name lookup.

## Source
https://docs.python.org/3.14/library/exceptions.html
- "Base classes" section, `exception LookupError` (plain-text extraction: line
  398-401 of the fetched HTML): `LookupError` documented as the base of
  `IndexError`, `KeyError` only.
- "Exception hierarchy" tree (fetched HTML lines ~1403-1500): `├── LookupError`
  at `Exception`'s indent level with children `│    ├── IndexError` and
  `│    └── KeyError`; separately `├── ValueError` → `│    └── UnicodeError` →
  `│         ├── UnicodeDecodeError`.
This claim was checked from that page alone; no other verdict in the bundle was
used.

## Source Version
Python 3.14.8 documentation (docs.python.org/3.14, page footer "Python 3.14.8
Documentation", last updated 2026-10-06 UTC).

## Source Fingerprint
sha256:afdce60fa94da72d1c097b472257c770c6bed41342509b34863b6da7833b745e
(exact bytes fetched from https://docs.python.org/3.14/library/exceptions.html,
144074 bytes)

## Invalidates On
- The "Exception hierarchy" tree re-parents `LookupError` (or `UnicodeError`) so
  the two branches merge — e.g. a proposal making lookup failures also
  `UnicodeError`s.
- A documented change to the `LookupError` entry adding a Unicode/decode subclass.
- Re-check against the exact doc page for the interpreter version in use
  (3.9-3.13 pages not hashed here); any hash change to the fetched page requires
  re-verification.

## Checked At
2026-10-06T22:24:59Z
