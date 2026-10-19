VERDICT: HOLDS
PROMOTION: ELIGIBLE

## Claim
`tests/test_dump_write_count.py` parametrizes exactly `TestOneWritePerFlush`, `TestFlushCountIsF`, `TestPairedMeasurement` and `TestFileCount` with `@pytest.mark.parametrize("fmt", COLLAPSED_FORMATS)`

## What it actually does
The file contains exactly four `@pytest.mark.parametrize("fmt", COLLAPSED_FORMATS)` decorators (lines 164, 189, 239, 262), one on the single test method of each of `TestOneWritePerFlush`, `TestFlushCountIsF`, `TestPairedMeasurement`, and `TestFileCount` (the decorator sits on each class's one test method — the standard idiom — not on the `class` statement itself). The file's only other test classes, `TestWriteUnit` and `TestDocumentedCount`, carry no parametrize. `COLLAPSED_FORMATS` is defined at line 29 as `["change-focused", "self-contained"]`, so each of the four classes runs two `fmt` variants.

## Source
/Users/alexhubitski/Projects/buster-ripper/tests/test_dump_write_count.py — lines 29, 141–275 (class/decorator layout confirmed by grep: decorators at 164, 189, 239, 262)

## Source Version
Repository working tree, uncommitted-file state as of check; no version string applicable

## Source Fingerprint
sha256:63b2c4c8e049f9462d4b44277cb1a3c387680da2820154a3e583d601c814822e

## Invalidates On
Any parametrize decorator added to or removed from any class in tests/test_dump_write_count.py
COLLAPSED_FORMATS renamed, removed, or its list membership changed (line 29)
A new test class added to the file carrying the same decorator

## Checked At
2026-09-28
