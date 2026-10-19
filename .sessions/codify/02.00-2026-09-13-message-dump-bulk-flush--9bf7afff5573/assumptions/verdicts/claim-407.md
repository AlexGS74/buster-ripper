VERDICT: HOLDS
PROMOTION: ELIGIBLE

## Claim
`UnicodeDecodeError` is not a subclass of `LookupError`

## What it actually does
`UnicodeDecodeError` inherits `UnicodeError`, which inherits `ValueError`:
"`UnicodeDecodeError` — Raised when a Unicode-related error occurs during
decoding. It is a subclass of `UnicodeError`." and "`UnicodeError` — Raised when
a Unicode-related encoding or decoding error occurs. It is a subclass of
`ValueError`." `LookupError` is a sibling branch off `Exception` whose only
documented subclasses are `IndexError` and `KeyError`, so `UnicodeDecodeError` is
not below it in the hierarchy. Consequence: `except LookupError` does not catch a
decoding failure, and `except ValueError` does not catch an invalid codec-name
lookup — the two need separate handlers (`UnicodeDecodeError` is reachable via
`ValueError`/`UnicodeError`, `LookupError` only via `LookupError`).

## Source
https://docs.python.org/3.14/library/exceptions.html
- "Concrete exceptions": `exception UnicodeDecodeError` ("subclass of
  `UnicodeError`", plain-text lines 955-957 of the fetched HTML) and `exception
  UnicodeError` ("subclass of `ValueError`", lines 904-906).
- "Exception hierarchy" tree: `UnicodeDecodeError` nested under `ValueError` →
  `UnicodeError`; `LookupError` listed at `Exception` level with only
  `IndexError`/`KeyError` beneath it.
Checked from this page on its own; it does not rely on claim-406's verdict.

## Source Version
Python 3.14.8 documentation (docs.python.org/3.14, page footer "Python 3.14.8
Documentation", last updated 2026-10-06 UTC).

## Source Fingerprint
sha256:afdce60fa94da72d1c097b472257c770c6bed41342509b34863b6da7833b745e
(exact bytes fetched from https://docs.python.org/3.14/library/exceptions.html,
144074 bytes)

## Invalidates On
- The hierarchy tree re-parents `UnicodeError`/`UnicodeDecodeError` under
  `LookupError` (or introduces multiple inheritance from it).
- A documented change to the `UnicodeDecodeError` or `UnicodeError` entry naming a
  different base class.
- Re-check against the exact doc page for the interpreter version actually in use
  (3.9-3.13 pages not hashed here); any hash change requires re-verification.

## Checked At
2026-10-06T22:24:59Z
