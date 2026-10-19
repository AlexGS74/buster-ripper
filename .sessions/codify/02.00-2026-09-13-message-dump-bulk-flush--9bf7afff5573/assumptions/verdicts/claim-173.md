VERDICT: HOLDS
PROMOTION: ELIGIBLE

## Claim
`tests/test_dump_write_count.py`'s `TestDocumentedCount` executes one unparametrized run on the `"documented"` dump format

## What it actually does
`TestDocumentedCount` (lines 217–228) contains a single test method, `test_documented_run_counts_2n_writes`, with no `parametrize` decorator, so it collects exactly one run. The run executes under the `"documented"` format established by the shared `dump_env` fixture (line 129 sets `config.DUMP_FORMAT = "documented"`); the test body itself never overrides `DUMP_FORMAT`, so "documented" is the effective format throughout — noting for precision that the value comes from the fixture rather than an assignment inside the class body.

## Source
/Users/alexhubitski/Projects/buster-ripper/tests/test_dump_write_count.py — lines 217–228 (fixture default at line 129)

## Source Version
Repository working tree, uncommitted-file state as of check; no version string applicable

## Source Fingerprint
sha256:63b2c4c8e049f9462d4b44277cb1a3c387680da2820154a3e583d601c814822e

## Invalidates On
The `dump_env` fixture's `config.DUMP_FORMAT = "documented"` default (line 129) changed or removed
TestDocumentedCount gains a parametrize decorator or an explicit `config.DUMP_FORMAT` override

## Checked At
2026-09-28
