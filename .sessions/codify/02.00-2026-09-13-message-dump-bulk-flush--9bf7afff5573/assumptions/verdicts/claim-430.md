VERDICT: HOLDS
PROMOTION: ELIGIBLE

## Claim
`difflib.unified_diff`'s `n` argument sets the number of unchanged context lines around each hunk and defaults to 3

## What it actually does
`unified_diff` has signature `unified_diff(a, b, fromfile='', tofile='', fromfiledate='', tofiledate='', n=3, lineterm='\n', *, color=False)` and calls `SequenceMatcher(None, a, b).get_grouped_opcodes(n)`. `SequenceMatcher.get_grouped_opcodes(n=3)` trims leading and trailing equal runs to at most `n` lines and splits long equal runs so each hunk carries at most `n` unchanged context lines on each side. The documentation states: "The number of context lines is set by *n* which defaults to three."

## Source
https://raw.githubusercontent.com/python/cpython/3.11/Lib/difflib.py — `unified_diff` signature and call at lines 1095-1138; `get_grouped_opcodes(self, n=3)` at lines 547-592. Corroborated by https://docs.python.org/3/library/difflib.html#difflib.unified_diff

## Source Version
CPython 3.11 branch `Lib/difflib.py`; project declares `requires-python = ">=3.11"`. The same signature and grouping call are present in the CPython 3.15 branch (lines 1101-1153), so the behavior is unchanged across the supported range checked.

## Source Fingerprint
sha256:0c6afc23568d55b3e9ac914f9c5361e3033e778aa5b58d3cc82835fc5c638679 (CPython 3.11 `Lib/difflib.py`); cross-checked 3.15 `Lib/difflib.py` sha256:aaa3d1ccdb8d8f0d2e8c99fe1a320a983b770fad94779f1dd417cdd37c033159

## Invalidates On
- The `n` default in `unified_diff` or `SequenceMatcher.get_grouped_opcodes` changes from 3
- `unified_diff` stops passing `n` to `get_grouped_opcodes`, or `get_grouped_opcodes` stops using `n` as the context-line limit
- Project minimum Python moves below 3.11 or to a version whose `difflib.unified_diff` differs on this point

## Checked At
2026-10-17
