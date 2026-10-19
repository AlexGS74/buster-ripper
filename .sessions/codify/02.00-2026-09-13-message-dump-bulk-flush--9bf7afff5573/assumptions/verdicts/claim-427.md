VERDICT: HOLDS
PROMOTION: ELIGIBLE

## Claim
`difflib.unified_diff` prefixes each unchanged context line with a single space

## What it actually does
For an `equal` opcode, `unified_diff` yields each line as `' ' + line` — exactly one leading space, no other marker. The documentation's worked example shows the unchanged line as ` guido` (leading space), and the implementation yields `' ' + line` in the `tag == 'equal'` branch.

## Source
https://docs.python.org/3/library/difflib.html (difflib.unified_diff section, worked example), corroborated by the implementation at https://raw.githubusercontent.com/python/cpython/3.13/Lib/difflib.py lines 1151-1155:
```
            if tag == 'equal':
                for line in a[i1:i2]:
                    yield ' ' + line
```

## Source Version
Python 3.15.0 documentation; CPython 3.13 branch source (`Lib/difflib.py`), fetched 2026-10-17

## Source Fingerprint
sha256:48a2142e8b2fedece2c2c8b59376b5ee76599c6539dc94a308f9a90ffc5ab2c9 (docs page as fetched)
sha256:1593ae08143632f4f81fea3beeeb279d803f8a9be922b12fe79664d78bba879d (Lib/difflib.py as fetched)

## Invalidates On
- `unified_diff` stops yielding `' ' + line` for `equal` opcodes
- The documented unified_diff example output changes the context-line prefix
- Either consulted source file changes bytes (fingerprint mismatch)

## Checked At
2026-10-17T23:41:03Z
