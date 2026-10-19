VERDICT: HOLDS
PROMOTION: ELIGIBLE

## Claim
`serve_buster_ripper.sh` appends a `--dump-format` flag to `EXTRA_FLAGS` only when `DUMP_FORMAT` is non-empty

## What it actually does
Line 41 is `[[ -n "${DUMP_FORMAT}" ]] && EXTRA_FLAGS+=(--dump-format "${DUMP_FORMAT}")`.
The `-n` test is the sole gate on the append, so the pair `--dump-format <value>` is
added to the `EXTRA_FLAGS` array only when `DUMP_FORMAT` expands to a non-empty
string; when it is empty the guard short-circuits and nothing is appended. `--dump-format`
appears nowhere else in the script, and `EXTRA_FLAGS` is only initialized (line 36)
and appended to (lines 37-42) before being expanded into the `exec buster-ripper`
command at line 54.

## Source
/Users/alexhubitski/Projects/buster-ripper/serve_buster_ripper.sh:41
(`[[ -n "${DUMP_FORMAT}" ]] && EXTRA_FLAGS+=(--dump-format "${DUMP_FORMAT}")`);
whole file read to confirm no other `--dump-format`/`EXTRA_FLAGS` writer.

## Source Version
working-tree file, no version marker in the script itself

## Source Fingerprint
sha256:c58256fbec793c520720c9e86fee8740653939275b508f50b212ab60bd0908af

## Invalidates On
- Line 41's guard changes from `-n` to unconditional, a different test, or a different variable
- A second `--dump-format` append or an unconditional `EXTRA_FLAGS` write is added
- `EXTRA_FLAGS` stops being an array (a string append would re-split the flag/value pair)

## Checked At
2026-10-05
