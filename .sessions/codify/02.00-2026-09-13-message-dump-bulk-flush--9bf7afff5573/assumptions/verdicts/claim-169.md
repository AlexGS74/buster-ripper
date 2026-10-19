VERDICT: HOLDS
PROMOTION: ELIGIBLE

## Claim
`tests/test_dump_records.py` reads the collapsed `<format>.log` artifact with `read_text()` in its readability and turn-0 tests

## What it actually does
The file contains exactly two `read_text()` calls, both reading `dump_env / sid / f"{fmt}.log"` where `fmt` comes from `COLLAPSED_FORMATS = ["change-focused", "self-contained"]` (line 21): line 212 in `TestArtifactReadability._flushed_artifact`, the shared helper invoked by both readability tests (`test_change_focused_reader_sees_the_change`, `test_self_contained_reader_compares_records`), and line 247 inside `TestArtifactTurnZero.test_body_and_rendering_recoverable_from_artifact`, parametrized over `COLLAPSED_FORMATS`. No other file-read mechanism (`open()`, `read_bytes`) appears in the file.

## Source
`tests/test_dump_records.py` lines 212 and 247 (with `COLLAPSED_FORMATS` at line 21). The claim's subject is the test suite itself, so the test file is the production surface here (P10 exception).

## Source Version
unavailable (git metadata not consulted per protocol boundary)

## Source Fingerprint
sha256:d3084c680899d8760732435d6cbad7c9ec9d1f1a159cf5d3142298d5123f1444

## Invalidates On
`_flushed_artifact` (line 212) or the turn-0 test (line 247) switches from `.read_text()` to another read mechanism (e.g. `open().read()` or `read_bytes().decode()`)
The readability or turn-0 tests are removed/renamed so they no longer read `DUMP_DIR/<sid>/<format>.log`
`COLLAPSED_FORMATS` changes so the `{fmt}.log` artifact read is no longer a collapsed-format log

## Checked At
2026-09-28T13:59:07Z
