VERDICT: HOLDS
PROMOTION: ELIGIBLE

## Claim
Python raises `LookupError` when a codec name is looked up and no such codec exists

## What it actually does
`codecs.lookup(encoding)` is documented to scan the registry cache and then the
registered search functions, and "If no `CodecInfo` object is found, a
`LookupError` is raised." Every convenience function that resolves an encoding
name through `lookup()` — `getencoder`, `getdecoder`, `getincrementalencoder`,
`getincrementaldecoder`, `getreader`, `getwriter` — is documented to "raise a
`LookupError` in case the encoding cannot be found", and the built-in-exceptions
reference states that `LookupError` "can be raised directly by `codecs.lookup()`".
`codecs.lookup_error(name)` raises `LookupError` the same way for an unknown error
handler name.

## Source
https://raw.githubusercontent.com/python/cpython/v3.14.8/Doc/library/codecs.rst
— `.. function:: lookup(encoding, /)`, lines 66-74 (raise-on-not-found); lines
116-162 (`get*` helpers, "Raises a :exc:`LookupError` in case the encoding cannot
be found"); line 475 (`lookup_error`). Cross-checked in
https://raw.githubusercontent.com/python/cpython/v3.12.0/Doc/library/codecs.rst
(lines 58-65, identical wording) and in
https://docs.python.org/3.14/library/exceptions.html#exception-hierarchy /
`exception LookupError` ("This can be raised directly by `codecs.lookup()`").

## Source Version
CPython 3.14.8 documentation source (tag `v3.14.8`); wording also verified
unchanged at tag `v3.12.0`. Docs page rendered as "Python 3.14.8 Documentation".

## Source Fingerprint
sha256:a93ab5fd0291451dacdbc3f71a22fbbeba2807550e46b2e2f40734af6b5ae726
(codecs.rst @ v3.14.8, 83655 bytes)
sha256:ed8acfbc828839f7e657406f40c486cc6ddd5b06d26bcd2d8afca89dd5760687
(codecs.rst @ v3.12.0, corroboration)
sha256:afdce60fa94da72d1c097b472257c770c6bed41342509b34863b6da7833b745e
(docs.python.org/3.14/library/exceptions.html, corroboration)

## Invalidates On
- The `codecs.lookup` doc paragraph changes the not-found exception to something
  other than `LookupError` (e.g. `ValueError`/`UnicodeError`), or drops the
  explicit raise sentence.
- CPython changes codec-registry semantics so an unknown name returns a fallback
  identity/ASCII codec or is auto-registered instead of raising.
- The project's interpreter is pinned to a Python release whose
  `Doc/library/codecs.rst` differs from the `v3.12.0`/`v3.14.8` bytes above
  (hash mismatch on that version's file) — re-check against the exact tag.
- A custom codec search function swallows the miss and returns a `CodecInfo`
  (per-call behaviour; the documented registry contract is unchanged).

## Checked At
2026-10-06T22:24:59Z
