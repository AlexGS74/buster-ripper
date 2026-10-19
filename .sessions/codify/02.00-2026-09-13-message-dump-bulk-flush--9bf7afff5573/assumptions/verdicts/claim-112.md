VERDICT: HOLDS
PROMOTION: ELIGIBLE

## Claim
Encoding a Python string with UTF-8 produces a `bytes` value whose length counts its encoded bytes

## What it actually does
`str.encode('utf-8')` returns a `bytes` object. Bytes objects are sequences of individual bytes, and `len` returns a sequence's length, so `len(s.encode('utf-8'))` counts encoded bytes rather than string characters.

## Source
https://raw.githubusercontent.com/python/cpython/3.14/Doc/builtins/stdtypes.rst — `str.encode` (lines 1928–1933), common sequence operations `len(s)` (line 1039), and bytes objects (lines 3318–3321).

## Source Version
CPython documentation branch 3.14; exact revision unavailable.

## Source Fingerprint
sha256:5a6951b72ecc630a87f45a650fc11c186cfeb4251bedd1ea28f45bcb3116e8e1

## Invalidates On
Change to the referenced CPython `Doc/builtins/stdtypes.rst` bytes, particularly `str.encode`, bytes sequence semantics, or `len` sequence semantics.

## Checked At
2026-09-27
