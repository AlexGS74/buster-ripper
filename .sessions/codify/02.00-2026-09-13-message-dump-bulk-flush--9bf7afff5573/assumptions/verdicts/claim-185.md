VERDICT: HOLDS
PROMOTION: ELIGIBLE

## Claim
A prefixed assignment `VAR="" cmd` leaves `VAR` set-and-empty in `cmd`'s environment

## What it actually does
Both halves of the claim are stated in bash's manual, and nothing in it carves out
empty values. (1) `VAR=""` *sets* the variable: "A parameter is set if it has been
assigned a value. The null string is a valid value", and for an assignment of the
form `name=[value]`, "If value is not given, the variable is assigned the null
string" — so `VAR=""` and `VAR=` are the same thing: set, with an empty value.
(2) With a command name present, that set variable goes into the command's
environment: "Otherwise, the variables are added to the environment of the
executed command and do not affect the current shell environment", and
"the environment ... is ... a list of name-value pairs, of the form `name=value`".
The Environment node makes the same point for prefix assignments specifically:
"If any parameter assignment statements ... appear before a simple command, the
variable assignments are part of that command's environment for as long as it
executes. These assignment statements affect only the environment seen by that
command." The pair added therefore has `VAR` present with an empty value, not
absent — the docs only ever remove pairs via `unset`/`export -n`, never by
assigning the null string.

Scope note (does not change the verdict): if `cmd` names a shell *function* rather
than an executed command, the same node says "the variables are local to the
function and exported to that function's children" — still set-and-empty for
`cmd`, still not the caller's environment.

## Source
GNU Bash Reference Manual — `doc/bashref.texi` upstream source at
`https://raw.githubusercontent.com/mirror/bash/master/doc/bashref.texi`, nodes
"Environment" (Environment subsection: name-value pairs; prefix-assignment
paragraph; -k option; `$_`) and "Simple Commands" (Simple Command Expansion
enumerate + "If no command name results ... Otherwise, the variables are added to
the environment of the executed command ..."), plus the "Parameters" node for
"A parameter is set if it has been assigned a value. The null string is a valid
value." Corroborated by the packaged manual page `bash(1)` at
`https://manpages.debian.org/testing/bash/bash.1.en.html` — sections PARAMETERS
and SIMPLE COMMAND EXPANSION carry the same sentences. (www.gnu.org unreachable
from this environment; documentation-only claim, read from the manual source and
packaged manual page — no bash probe was run, and `mirror/bash` was confirmed to
be GNU Bash by `patchlevel.h`: "This file is part of GNU Bash".)

## Source Version
GNU Bash 5.3 (per `doc/version.texi`: `@set EDITION 5.3`, `@set VERSION 5.3`,
updated 18 May 2025; master-branch bytes as fetched). Corroborating manual page:
`bash(1)` from Debian package `bash 5.3-3+b1`.

## Source Fingerprint
sha256:f3d37d57a1061e24d266051de9bd47ffa43dc86584afea11576c535ad2be32d5 (full fetched `doc/bashref.texi`, 415804 bytes)
sha256:a827e4d7e765538cfbb239118accd7b2d45f610cd474c496e119eb57861fc5e5 (corroborating `bash(1)` man page HTML, 532248 bytes)

## Invalidates On
- The fetched `doc/bashref.texi` bytes change (digest above), or the Environment /
  Simple Command Expansion sentences quoted above are reworded.
- Any bash release that documents null-valued prefix assignments being omitted
  from the command environment (i.e. `VAR="" cmd` dropping the pair instead of
  adding `VAR=`).
- Reading the claim as being about a *special builtin* or about `VAR=""` with no
  command name at all, or under `set -o posix`, where the manual routes the
  assignment to the current shell environment instead — that is a different
  claim, and this verdict does not cover it.

## Checked At
2026-10-04
