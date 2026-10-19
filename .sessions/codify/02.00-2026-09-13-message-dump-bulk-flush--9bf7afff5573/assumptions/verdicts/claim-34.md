VERDICT: DOES-NOT-HOLD
PROMOTION: ELIGIBLE

## Claim
`buster_ripper`'s modules read `config` attributes at call time rather than binding them at import, so values assigned before serving are visible everywhere

## What it actually does
Most request-path reads use `config.<NAME>` at call time, but `buster_ripper.main` evaluates `config.UPSTREAM`, `config.DUMP_FLUSH_INTERVAL`, and `config.DUMP_FLUSH_BYTES` as Click option defaults while defining the command at import time (lines 30–57). Assigning these attributes after importing `main` but before serving does not update those already-bound CLI defaults; the blanket “everywhere” claim is false.

## Source
`buster_ripper/main.py:30-57` (Click option decorators); `buster_ripper/main.py:104-105` (parsed upstream reassigned to config)

## Source Version
unavailable

## Source Fingerprint
sha256:cbe2802d474abe21ada5bb2e94c09cce15237ea53d526c6d31fed25207c950d1

## Invalidates On
Change to `buster_ripper/main.py` option defaults or how the CLI assigns configuration at startup.

## Checked At
2026-09-27
