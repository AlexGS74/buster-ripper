VERDICT: DOES-NOT-HOLD
PROMOTION: ELIGIBLE

## Claim
`serve_buster_ripper.sh` passes CLI options to the binary only through its `EXTRA_FLAGS` variable and forwards no positional arguments

## What it actually does
The script passes CLI options to the binary through two channels, not one: the always-on options `--upstream "${UPSTREAM}"`, `--host "${HOST}"`, and `--port "${PORT}"` are written directly in the `exec buster-ripper` line (lines 36–40), outside `EXTRA_FLAGS`; the `EXTRA_FLAGS` variable (lines 28–32) carries only the four conditional flags `--strip-date`, `--verbose`, `--dump-dir <dir>`, and `--preserved-thinking`. The second conjunct is accurate: the script references no positional parameters (`$1`/`$@` never appear), and the exec line consists solely of option flags, so no positional arguments are forwarded.

## Source
/Users/alexhubitski/Projects/buster-ripper/serve_buster_ripper.sh — exec invocation at lines 36–40; `EXTRA_FLAGS` construction at lines 28–32

## Source Version
unavailable (working-tree script file, no version metadata)

## Source Fingerprint
sha256:67d194bc95dcc42ed3d09ee0214e97fd754465a615ebd555c8d3280f170361a9

## Invalidates On
- `serve_buster_ripper.sh` exec line is changed to route `--upstream`/`--host`/`--port` through `EXTRA_FLAGS` (making it the sole option channel)
- `serve_buster_ripper.sh` adds positional-parameter forwarding (e.g. `"$@"` appended to the `exec buster-ripper` invocation)

## Checked At
2026-09-19
