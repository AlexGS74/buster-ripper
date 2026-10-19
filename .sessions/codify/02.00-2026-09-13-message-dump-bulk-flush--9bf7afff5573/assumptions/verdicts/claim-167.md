VERDICT: UNCLEAR
PROMOTION: INELIGIBLE — verdict is UNCLEAR; no single reading of "write-count tests" was settleable from the sources, and promotion requires HOLDS or DOES-NOT-HOLD

## Claim
`tests/test_dump_write_count.py` runs its write-count, paired-measurement and file-count tests once per shipped collapsed format

## What it actually does
Four of the module's six test classes are `@pytest.mark.parametrize("fmt", COLLAPSED_FORMATS)` with `COLLAPSED_FORMATS = ["change-focused", "self-contained"]` (line 29), and that set equals the two collapsed values of the production closed set `config.DUMP_FORMATS = ("documented", "change-focused", "self-contained")` (buster_ripper/config.py line 23, whose comment names them "the two collapsed values"): `TestOneWritePerFlush`, `TestFlushCountIsF`, `TestPairedMeasurement`, `TestFileCount`. The two remaining classes are NOT parametrized: `TestWriteUnit` runs once with `config.DUMP_FORMAT = "change-focused"` hardcoded (line 149), and `TestDocumentedCount` runs once on the documented format. The claim's truth turns entirely on which classes "its write-count ... tests" covers — a term choice, not an observed behaviour (P9). Reading (a): "write-count tests" = the collapsed-format write-count tests (`TestOneWritePerFlush`, `TestFlushCountIsF`), alongside `TestPairedMeasurement` and `TestFileCount` — then every test in the claim's scope runs once per shipped collapsed format and the claim HOLDS. Reading (b): "write-count tests" = every write-counting test in the module (the module docstring line 1 self-describes the whole file as "Instrumented write-count tests for the dump writers (T004)"), which includes `TestWriteUnit` (one run, one hardcoded format) and `TestDocumentedCount` (one run, no collapsed format) — then the claim DOES-NOT-HOLD.

## Source
tests/test_dump_write_count.py — COLLAPSED_FORMATS (line 29) and the parametrize decorators at lines 164, 189, 239, 262; non-parametrized classes at lines 141-158 (TestWriteUnit) and 217-228 (TestDocumentedCount). Cross-checked against buster_ripper/config.py line 23 for the shipped format set.

## Source Version
unavailable (working-tree repo source; git metadata out of scope for this role)

## Source Fingerprint
sha256:63b2c4c8e049f9462d4b44277cb1a3c387680da2820154a3e583d601c814822e (test file; config.py consulted for the shipped set: sha256:7e1eff53eb56eb96aba4ff9badd0d9d16f3b02f062e2973676dcf8d670f71dde)

## Invalidates On
- Re-extraction naming the exact test classes "write-count tests" covers — that alone settles reading (a) vs (b)
- Any change to COLLAPSED_FORMATS in the test file, to the four parametrize decorators, or to config.DUMP_FORMATS (a third shipped collapsed format, or dropping one, changes the answer under both readings)

## Checked At
2026-09-28
