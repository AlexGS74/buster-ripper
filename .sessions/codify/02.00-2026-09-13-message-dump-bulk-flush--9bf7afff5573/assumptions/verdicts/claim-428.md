VERDICT: HOLDS
PROMOTION: ELIGIBLE

## Claim
`difflib.unified_diff` prefixes its file-header lines with `---`/`+++` and its hunk-header lines with `@@`

## What it actually does
On the first group, `unified_diff` yields `'--- {fromfile}{fromdate}{lineterm}'` and `'+++ {tofile}{todate}{lineterm}'` for the two file headers, and for every group yields `'@@ -{range1} +{range2} @@{lineterm}'` for the hunk header. The documentation describes these as "the diff control lines (those with `---`, `+++`, or `@@`)" and its worked example shows `--- before.py`, `+++ after.py`, `@@ -1,4 +1,4 @@`.

## Source
https://docs.python.org/3/library/difflib.html (difflib.unified_diff section, worked example), corroborated by the implementation at https://raw.githubusercontent.com/python/cpython/3.13/Lib/difflib.py lines 1138-1149:
```
            yield '--- {}{}{}'.format(fromfile, fromdate, lineterm)
            yield '+++ {}{}{}'.format(tofile, todate, lineterm)
...
        yield '@@ -{} +{} @@{}'.format(file1_range, file2_range, lineterm)
```

## Source Version
Python 3.15.0 documentation; CPython 3.13 branch source (`Lib/difflib.py`), fetched 2026-10-17

## Source Fingerprint
sha256:48a2142e8b2fedece2c2c8b59376b5ee76599c6539dc94a308f9a90ffc5ab2c9 (docs page as fetched)
sha256:1593ae08143632f4f81fea3beeeb279d803f8a9be922b12fe79664d78bba879d (Lib/difflib.py as fetched)

## Invalidates On
- `unified_diff` stops yielding `---`/`+++` file-header lines or `@@` hunk-header lines
- The documented unified_diff example output changes its header prefixes
- Either consulted source file changes bytes (fingerprint mismatch)

## Checked At
2026-10-17T23:41:03Z
