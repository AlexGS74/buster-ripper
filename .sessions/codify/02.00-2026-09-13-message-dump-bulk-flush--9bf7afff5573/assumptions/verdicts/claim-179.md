VERDICT: HOLDS
PROMOTION: ELIGIBLE

## Claim
`serve_buster_ripper.sh` initializes `SKIP_TLS_VERIFY` from the environment with default `0`

## What it actually does
Line 27 is `SKIP_TLS_VERIFY=${SKIP_TLS_VERIFY:-0}` — a shell parameter expansion that
takes the value already exported in the environment and falls back to the literal `0`
when the variable is unset or empty. The variable is not unconditionally reassigned,
sourced from a config file, or given any other default elsewhere in the script, so an
environment value other than `0` survives initialization.

## Source
/Users/alexhubitski/Projects/buster-ripper/serve_buster_ripper.sh:27 (whole file read;
54 lines). Checked against the mutable worktree source root supplied as
`Source root:` — no snapshot was provided. Confirmed exhaustively by a search scoped to
that one file: `SKIP_TLS_VERIFY` occurs only at line 16 (comment), line 27
(initialization) and line 39 (consumption).

## Source Version
Mutable worktree file — no VCS revision consulted (Git metadata is out of scope for
this role). Stability is carried by the byte fingerprint below.

## Source Fingerprint
sha256:c58256fbec793c520720c9e86fee8740653939275b508f50b212ab60bd0908af

## Invalidates On
- Any edit to line 27 that replaces `${SKIP_TLS_VERIFY:-0}` with a hard assignment, a
  different default (e.g. `1`), or `${SKIP_TLS_VERIFY-0}`/`${SKIP_TLS_VERIFY:=0}`
  semantics that change the unset/empty distinction.
- Introduction of another assignment to `SKIP_TLS_VERIFY` after line 27 (a later
  reinitialization would override the environment default).
- Addition of a config-file or CLI-parse stage that sets `SKIP_TLS_VERIFY` before line
  27 runs, or a change to the script's `set -u` status at line 2 that alters how unset
  environment variables are handled.
- Any change to the bytes of `serve_buster_ripper.sh` (the fingerprint no longer
  matches, so the check must be re-run).

## Checked At
2026-10-05
