VERDICT: HOLDS
PROMOTION: ELIGIBLE — two exact locators (named below), both fingerprinted; no single page states both the default and the exact exception class

## Claim
Python's `bytes.decode()` uses strict error handling when its `errors` argument is omitted, so invalid bytes raise `UnicodeDecodeError`

## What it actually does
The documented signature is `bytes.decode(encoding='utf-8', errors='strict')` — omitting `errors` selects the `'strict'` handler — and that handler raises on a malformed byte sequence; for decoding specifically the exception raised is `UnicodeDecodeError` (a subclass of `UnicodeError`/`ValueError`). The library reference names the class only at the `UnicodeError` level for `bytes.decode`; the Unicode HOWTO names `UnicodeDecodeError` and demonstrates it on `bytes.decode` itself, so both halves of the claim are covered.

## Source
Primary (default value): https://docs.python.org/3/library/stdtypes.html#bytes.decode — "bytes.decode(encoding='utf-8', errors='strict') … errors controls how decoding errors are handled. If 'strict' (the default), a UnicodeError exception is raised."

Primary (exception class, invalid bytes): https://docs.python.org/3/howto/unicode.html#the-string-type — "'strict' (raise a UnicodeDecodeError exception)"; worked example: `>>> b'\x80abc'.decode("utf-8", "strict")` → `UnicodeDecodeError: 'utf-8' codec can't decode byte 0x80 in position 0: invalid start byte`.

Corroborating: https://docs.python.org/3/library/codecs.html#error-handlers — "'strict' Raise UnicodeError (or a subclass), this is the default."; and `codecs.decode` docs — "The default error handler is 'strict' meaning that decoding errors raise ValueError (or a more codec specific subclass, such as UnicodeDecodeError)".

Checked by reading fetched documentation only — the interpreter was not run (P1).

## Source Version
Python 3.14.8 documentation (docs.python.org/3), fetched 2026-10-06. Claim is not version-sensitive within Python 3: the `errors='strict'` default for `bytes.decode` is unchanged since 3.x; the only listed changes are 3.1 (keyword-argument support) and 3.9 (`errors` value validity checked in Development/debug mode).

## Source Fingerprint
sha256:bf149b1649601f39dfcbb435a1c1650e2607fe9d5f315fc93ea8432b28ebae6b (stdtypes.html)
sha256:85626570292e087e7da3e0df0c941abdf85675c14f825df2ee1d5160d0fc8716 (howto/unicode.html)
sha256:5f29770c85fc38bec0a19414d698d89c34d1ceb6bfa2c0b5421dfcbaf281b98d (codecs.html)

## Invalidates On
- The `bytes.decode` signature in `library/stdtypes.html` changes its documented `errors` default away from `'strict'`.
- The docs rename or re-scope the decoding error class (e.g. state a class other than `UnicodeDecodeError` for `bytes.decode`).
- Either fingerprinted page's consulted text changes on re-fetch, or the verdict is applied to a CPython whose own bundled docs differ from the 3.14.8 render.
- A target runtime that is not CPython/Python 3 standard-library `bytes` (the claim is about the stdlib type, not an installed third-party shim).

## Checked At
2026-10-06
