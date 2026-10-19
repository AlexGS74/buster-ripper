VERDICT: HOLDS
PROMOTION: ELIGIBLE

## Claim
`difflib.unified_diff`'s hunk-header line contains a second `@@` after the opening one

## What it actually does
The hunk-header line is built as `f'{t.hunk}@@ -{file1_range} +{file2_range} @@{lineterm}{t.reset}'`. It opens with `@@`, then carries the two range descriptors, then a second literal `@@` before the line terminator (and, when color is enabled, the reset sequence). The documented example shows exactly `@@ -1,4 +1,4 @@`.

## Source
CPython 3.15 `Lib/difflib.py`, function `unified_diff`, hunk-header yield line, https://raw.githubusercontent.com/python/cpython/3.15/Lib/difflib.py ; corroborated by the documented example at https://docs.python.org/3/library/difflib.html (`@@ -1,4 +1,4 @@`).

## Source Version
CPython 3.15 branch source; Python 3.15.0 documentation

## Source Fingerprint
sha256:aaa3d1ccdb8d8f0d2e8c99fe1a320a983b770fad94779f1dd417cdd37c033159 (Lib/difflib.py); sha256:694e808b780a68f0481f36fde8ef38db17adf6b74373106fcbc49c22016506ff (docs page)

## Invalidates On
- The hunk-header format string in `unified_diff` drops or changes the trailing `@@`
- The hunk header is reimplemented to use a single `@@` or a different range syntax
- A different difflib implementation or major version replaces this behavior

## Checked At
2026-10-17
