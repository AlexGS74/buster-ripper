VERDICT: HOLDS
PROMOTION: ELIGIBLE

## Claim
`difflib.unified_diff` yields its `---`/`+++` file-header lines before any `@@` hunk header

## What it actually does
`unified_diff` tracks a `started` flag while iterating `SequenceMatcher(...).get_grouped_opcodes(n)`. On the first group only, before computing or yielding that group's hunk header, it yields `--- {fromfile}{fromdate}{lineterm}` and then `+++ {tofile}{todate}{lineterm}`. Only afterwards, in the same iteration, does it yield `@@ -{file1_range} +{file2_range} @@{lineterm}`. Thus the two file-header lines precede every `@@` hunk header; if there are no grouped opcodes at all, neither the header lines nor any `@@` line is yielded.

## Source
CPython 3.15 `Lib/difflib.py`, function `unified_diff` (lines 1101-1170), https://raw.githubusercontent.com/python/cpython/3.15/Lib/difflib.py ; corroborated by the documented example at https://docs.python.org/3/library/difflib.html ("--- before.py", "+++ after.py", then "@@ -1,4 +1,4 @@").

## Source Version
CPython 3.15 branch source; Python 3.15.0 documentation

## Source Fingerprint
sha256:aaa3d1ccdb8d8f0d2e8c99fe1a320a983b770fad94779f1dd417cdd37c033159 (Lib/difflib.py); sha256:694e808b780a68f0481f36fde8ef38db17adf6b74373106fcbc49c22016506ff (docs page)

## Invalidates On
- `unified_diff` stops emitting file-header lines before the first hunk header
- `unified_diff` stops emitting `---`/`+++` lines entirely or reorders them after `@@`
- The `started` flag / grouped-opcode loop in `Lib/difflib.py` is restructured
- A different difflib implementation or major version replaces this behavior

## Checked At
2026-10-17
