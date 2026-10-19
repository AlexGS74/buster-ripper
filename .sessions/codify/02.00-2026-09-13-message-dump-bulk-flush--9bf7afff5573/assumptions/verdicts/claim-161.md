VERDICT: HOLDS
PROMOTION: ELIGIBLE

## Claim
`serve_buster_ripper.sh` appends a `--dump-dir` flag to `EXTRA_FLAGS` only when `DUMP_DIR` is non-empty

## What it actually does
Line 40 is `[[ -n "${DUMP_DIR}" ]] && EXTRA_FLAGS+=(--dump-dir "${DUMP_DIR}")`.
The `-n` (non-empty) test is the only gate on the append, so the pair
`--dump-dir <value>` reaches the `EXTRA_FLAGS` array only when `DUMP_DIR`
expands to a non-empty string. `--dump-dir` appears nowhere else in the script,
and `EXTRA_FLAGS` is only initialized (line 36) and appended to (lines 37-42)
before being expanded into the `exec buster-ripper` command at line 54.

## Source
/Users/alexhubitski/Projects/buster-ripper/serve_buster_ripper.sh:40
(`[[ -n "${DUMP_DIR}" ]] && EXTRA_FLAGS+=(--dump-dir "${DUMP_DIR}")`);
whole file read to confirm no other `--dump-dir`/`EXTRA_FLAGS` writer.

## Source Version
working-tree file, no version marker in the script itself

## Source Fingerprint
sha256:c58256fbec793c520720c9e86fee8740653939275b508f50b212ab60bd0908af

## Invalidates On
- Line 40's guard changes from `-n` to unconditional, a different test, or a different variable
- A second `--dump-dir` append or an unconditional `EXTRA_FLAGS` write is added
- Line 28's initialization changes (e.g. `${DUMP_DIR:-...}` instead of `${DUMP_DIR-...}`),
  which would alter when the `-n` guard can be false even though the guard itself is unchanged
- `EXTRA_FLAGS` stops being an array (a string append would re-split the flag/value pair)

## Checked At
2026-10-05
