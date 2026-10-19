VERDICT: HOLDS
PROMOTION: ELIGIBLE

## Claim
`difflib.unified_diff` appends its `lineterm` value to each file-header and hunk-header line, not to content lines

## What it actually does
`unified_diff` yields the two file headers as `'--- {}{}{}'.format(fromfile, fromdate, lineterm)` and `'+++ {}{}{}'.format(tofile, todate, lineterm)`, and the hunk header as `'@@ -{} +{} @@{}'.format(file1_range, file2_range, lineterm)`, so `lineterm` terminates every control line. Content lines are yielded as `' ' + line`, `'-' + line`, or `'+' + line` with no `lineterm` appended; their line endings come only from the input strings themselves. This matches the documentation: "the diff control lines (those with ---, +++, or @@) are created with a trailing newline", and setting `lineterm=''` produces uniformly newline-free output.

## Source
https://raw.githubusercontent.com/python/cpython/3.11/Lib/difflib.py — `unified_diff`, lines 1095-1161 (header formatting at 1143-1149; content lines at 1152-1161). Corroborated by https://docs.python.org/3/library/difflib.html#difflib.unified_diff

## Source Version
CPython 3.11 branch `Lib/difflib.py`; project declares `requires-python = ">=3.11"`. The same control-line formatting is present in the CPython 3.15 branch (lines 1101-1176), so the behavior is unchanged across the supported range checked.

## Source Fingerprint
sha256:0c6afc23568d55b3e9ac914f9c5361e3033e778aa5b58d3cc82835fc5c638679 (CPython 3.11 `Lib/difflib.py`); cross-checked 3.15 `Lib/difflib.py` sha256:aaa3d1ccdb8d8f0d2e8c99fe1a320a983b770fad94779f1dd417cdd37c033159

## Invalidates On
- `unified_diff` stops appending `lineterm` to the `---`/`+++`/`@@` lines in `Lib/difflib.py`
- `unified_diff` starts appending `lineterm` to context/added/removed content lines
- Project minimum Python moves below 3.11 or to a version whose `difflib.unified_diff` differs on this point

## Checked At
2026-10-17
