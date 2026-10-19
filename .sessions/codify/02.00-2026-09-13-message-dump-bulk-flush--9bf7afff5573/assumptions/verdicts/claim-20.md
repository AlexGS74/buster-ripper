VERDICT: DOES-NOT-HOLD
PROMOTION: ELIGIBLE

## Claim
Nothing under `buster_ripper/` or `tests/` reads `index.txt`, `turn_000.json` or a `turn_*.diff`

## What it actually does
`tests/test_dump.py` reads all three: `turn_000.json`, `turn_001.diff` and `turn_002.diff` via `read_text()` in one test, and `index.txt` via `read_text()` in another. The claim's subject includes the test suite, so those test reads directly refute it (P10 exception).

## Source
`tests/test_dump.py:129-163`

## Source Version
unavailable

## Source Fingerprint
sha256:0f32dc28712742a9b89536d6f52a202cbbfab3543a5cd41f4a3f783cc96d7ab9

## Invalidates On
Change to `tests/test_dump.py` removing or altering these reads.

## Checked At
2026-09-27
