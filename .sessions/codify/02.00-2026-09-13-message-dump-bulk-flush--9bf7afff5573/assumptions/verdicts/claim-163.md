VERDICT: HOLDS
PROMOTION: ELIGIBLE

## Claim
`serve_buster_ripper.sh` forwards no positional arguments to the exec'd binary

## What it actually does
The exec statement (lines 50-54) is made up entirely of long options and their
option-arguments: `--upstream "${UPSTREAM}"`, `--host "${HOST}"`, `--port "${PORT}"`, plus
whatever `EXTRA_FLAGS` contributes. Every `EXTRA_FLAGS` append (lines 37-42) is either a
bare `--flag` or a `--flag "<quoted value>"` pair, so no array element can become a loose
word — the block comment at lines 32-35 states that the array form exists precisely to stop
word-splitting from handing the binary "an extra positional word".

Separately, the script never touches its own positional parameters: a search of the file for
`"$@"`, `$*`, `$1`-`$9`, `${N}` and `shift` returns no matches (the only hit for that pattern
set is the `exec` keyword on line 50). So script-level arguments are neither accepted nor
passed through, and nothing in the file re-executes the binary by another route.

Term scope, recorded for the next re-check rather than as doubt: "positional argument" is used
here in its standard CLI sense — an argument not bound to a named option. The values following
`--upstream`/`--host`/`--port`/`--dump-dir`/`--dump-format` are option-arguments of options that
declare them, not positional words; no reading of the file produces a bare-word argument.

## Source
`/Users/alexhubitski/Projects/buster-ripper/serve_buster_ripper.sh` lines 32-42 (EXTRA_FLAGS
assembly and its rationale) and lines 50-54 (the `exec` statement).

## Source Version
Unavailable as a revision string — the locator supplied by the caller is the mutable worktree
file itself. Consulted bytes are identified by the fingerprint below; per this role's
boundary, Git metadata was not read to resolve a revision.

## Source Fingerprint
sha256:c58256fbec793c520720c9e86fee8740653939275b508f50b212ab60bd0908af

## Invalidates On
- The exec line gains `"$@"`, `$*`, `$1`, or any script positional parameter.
- An `EXTRA_FLAGS+=( ... )` append adds a value that is not a `--flag` and not directly
  preceded by the `--flag` it belongs to (e.g. `EXTRA_FLAGS+=("${DUMP_DIR}")`).
- A flag's value is assembled unquoted or into a plain string variable that can word-split
  into an extra bare word.
- Argument forwarding is introduced via `shift`, `getopts`, `set --`, or a wrapper array that
  reaches the exec line outside `EXTRA_FLAGS`.
- The sha256 fingerprint of `serve_buster_ripper.sh` above changes in any way.

## Checked At
2026-10-05T04:38:29Z
