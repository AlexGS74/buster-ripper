VERDICT: HOLDS
PROMOTION: ELIGIBLE

## Claim
`serve_buster_ripper.sh`'s exec line names the binary as the bare command `buster-ripper`

## What it actually does
Line 50 reads `exec buster-ripper \` — the command word is the bare token `buster-ripper`.
It carries no directory component (no `/`, no `./`, no absolute path), no variable or
command substitution, and no launcher prefix such as `uv run` or `python -m`, so resolution
is left to the shell's `PATH` lookup at exec time.

A search of the file for `exec` returns exactly this one match, so there is no second
invocation site naming the binary differently. The header comments at lines 7-9 describe
`uv tool install .` / `uv run buster-ripper --help` as prerequisites — consistent with a
PATH-resolvable console script named `buster-ripper`, but that is corroboration only; the
verdict rests on line 50.

Out of scope here by role boundary: whether that bare name actually resolves on a given
machine is state, not source — establishing it would require running something.

## Source
`/Users/alexhubitski/Projects/buster-ripper/serve_buster_ripper.sh` line 50 (the `exec`
statement's command word), with lines 1-9 read for context.

## Source Version
Unavailable as a revision string — the locator supplied by the caller is the mutable worktree
file itself. Consulted bytes are identified by the fingerprint below; per this role's
boundary, Git metadata was not read to resolve a revision.

## Source Fingerprint
sha256:c58256fbec793c520720c9e86fee8740653939275b508f50b212ab60bd0908af

## Invalidates On
- Line 50's command word becomes a path (`./buster-ripper`, `/usr/local/bin/buster-ripper`),
  a variable (`"${BIN}"`), or a substitution (`"$(command -v buster-ripper)"`).
- A launcher is prefixed to the exec (`uv run buster-ripper`, `python -m buster_ripper`,
  `env ... buster-ripper`), so the bare command is no longer what is named.
- The console-script name changes in the packaging metadata the install uses, so the bare
  token on the exec line is stale.
- An additional `exec`/invocation site is added that names the binary by other means.
- The sha256 fingerprint of `serve_buster_ripper.sh` above changes in any way.

## Checked At
2026-10-05T04:38:29Z
