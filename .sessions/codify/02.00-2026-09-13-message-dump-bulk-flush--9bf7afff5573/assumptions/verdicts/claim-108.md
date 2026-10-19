VERDICT: HOLDS
PROMOTION: ELIGIBLE

## Claim
`tests/test_dump.py` reads `index.txt` with `read_text()`

## What it actually does
`TestDocumentedFormatPreserved.test_index_line_format_unchanged` calls `.read_text()` on `index.txt`, then splits the result into lines for assertions.

## Source
`tests/test_dump.py:156-173` (the claim concerns the test suite itself).

## Source Version
unavailable

## Source Fingerprint
sha256:0f32dc28712742a9b89536d6f52a202cbbfab3543a5cd41f4a3f783cc96d7ab9

## Invalidates On
Any change to `tests/test_dump.py` that removes or replaces the `index.txt` read.

## Checked At
2026-09-27
