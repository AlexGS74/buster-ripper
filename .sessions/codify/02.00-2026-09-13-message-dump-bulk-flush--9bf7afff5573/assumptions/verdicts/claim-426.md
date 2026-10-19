VERDICT: HOLDS
PROMOTION: ELIGIBLE

## Claim
`difflib.unified_diff` prefixes each removed content line with `-` and each added content line with `+`

## What it actually does
`unified_diff` emits one-character prefixes on content lines: lines from `a` in a `replace`/`delete` opcode are yielded as `'-' + line`, and lines from `b` in a `replace`/`insert` opcode are yielded as `'+' + line`. The documentation's worked example shows exactly this (`-bacon`, `-eggs`, `-ham` then `+python`, `+eggy`, `+hamster`).

## Source
https://docs.python.org/3/library/difflib.html (difflib.unified_diff section, worked example), corroborated by the implementation at https://raw.githubusercontent.com/python/cpython/3.13/Lib/difflib.py lines 1151-1161:
```
            if tag in {'replace', 'delete'}:
                for line in a[i1:i2]:
                    yield '-' + line
            if tag in {'replace', 'insert'}:
                for line in b[j1:j2]:
                    yield '+' + line
```

## Source Version
Python 3.15.0 documentation; CPython 3.13 branch source (`Lib/difflib.py`), fetched 2026-10-17

## Source Fingerprint
sha256:48a2142e8b2fedece2c2c8b59376b5ee76599c6539dc94a308f9a90ffc5ab2c9 (docs page as fetched)
sha256:1593ae08143632f4f81fea3beeeb279d803f8a9be922b12fe79664d78bba879d (Lib/difflib.py as fetched)

## Invalidates On
- `unified_diff` stops yielding `'-' + line` / `'+' + line` for `replace`/`delete`/`insert` opcodes
- The documented unified_diff example output changes its content-line prefixes
- Either consulted source file changes bytes (fingerprint mismatch)

## Checked At
2026-10-17T23:41:03Z
