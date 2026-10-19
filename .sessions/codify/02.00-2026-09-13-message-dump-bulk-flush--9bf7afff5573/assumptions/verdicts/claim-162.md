VERDICT: HOLDS
PROMOTION: ELIGIBLE

## Claim
`serve_buster_ripper.sh`'s `exec` line passes the always-on `--upstream`/`--host`/`--port` options and the assembled `EXTRA_FLAGS` to the `buster-ripper` binary

## What it actually does
The script's one and only `exec` statement (lines 50-54) invokes `buster-ripper` with
`--upstream "${UPSTREAM}"`, `--host "${HOST}"` and `--port "${PORT}"` written literally on
the exec line — unconditional, so "always-on" is accurate — and then appends the assembled
array via `${EXTRA_FLAGS[@]+"${EXTRA_FLAGS[@]}"}`. Every conditional flag in the script is
built into `EXTRA_FLAGS` (lines 36-42: `--strip-date`, `--verbose`, `--skip-tls-verify`,
`--dump-dir <value>`, `--dump-format <value>`, `--preserved-thinking`) and reaches the
binary only through that expansion, so the exec line carries both parts the claim names.

Verified independently of the other two claims in this bundle by reading the exec statement
and the array-building block directly; the grep for `exec` in the file returns exactly one
match (line 50), so there is no alternate exec path where this would differ.

## Source
`/Users/alexhubitski/Projects/buster-ripper/serve_buster_ripper.sh` lines 36-42 (EXTRA_FLAGS
assembly) and lines 50-54 (the `exec` statement).

## Source Version
Unavailable as a revision string — the locator supplied by the caller is the mutable worktree
file itself. Consulted bytes are identified by the fingerprint below; per this role's
boundary, Git metadata was not read to resolve a revision.

## Source Fingerprint
sha256:c58256fbec793c520720c9e86fee8740653939275b508f50b212ab60bd0908af

## Invalidates On
- The `exec` statement drops, renames, or comments out any of `--upstream`/`--host`/`--port`.
- Any of `--upstream`/`--host`/`--port` moves behind a conditional or into `EXTRA_FLAGS`, so
  it is no longer always-on.
- The `${EXTRA_FLAGS[@]+"${EXTRA_FLAGS[@]}"}` expansion is removed from the exec line, or the
  array is replaced by a differently-named variable or a plain string.
- A second `exec` (or non-`exec` invocation of the binary) is added that bypasses this line.
- The sha256 fingerprint of `serve_buster_ripper.sh` above changes in any way.

## Checked At
2026-10-05T04:38:29Z
