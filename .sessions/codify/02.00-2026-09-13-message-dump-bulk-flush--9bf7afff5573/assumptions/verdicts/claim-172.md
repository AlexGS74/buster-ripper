VERDICT: HOLDS
PROMOTION: ELIGIBLE

## Claim
`tests/test_dump_write_count.py`'s `TestWriteUnit` executes one unparametrized run that hardcodes `config.DUMP_FORMAT = "change-focused"`

## What it actually does
`TestWriteUnit` (lines 141–158) contains a single test method, `test_mkdir_and_outside_paths_are_not_counted`, with no `parametrize` decorator, so it collects exactly one run. That method assigns `config.DUMP_FORMAT = "change-focused"` directly at line 149 (overriding the `"documented"` default the `dump_env` fixture set at line 129) before dumping its turns.

## Source
/Users/alexhubitski/Projects/buster-ripper/tests/test_dump_write_count.py — lines 141–158 (assignment at line 149)

## Source Version
Repository working tree, uncommitted-file state as of check; no version string applicable

## Source Fingerprint
sha256:63b2c4c8e049f9462d4b44277cb1a3c387680da2820154a3e583d601c814822e

## Invalidates On
The `config.DUMP_FORMAT = "change-focused"` assignment in TestWriteUnit changed or removed
A parametrize decorator added to TestWriteUnit (making the run parametrized rather than single)

## Checked At
2026-09-28
