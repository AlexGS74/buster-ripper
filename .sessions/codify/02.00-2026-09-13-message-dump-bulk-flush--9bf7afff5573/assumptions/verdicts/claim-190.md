VERDICT: HOLDS
PROMOTION: ELIGIBLE

## Claim
bash executes the first matching executable found in `$PATH`, so an earlier entry shadows an installed command

## What it actually does
For a command name containing no slash, and only after shell functions and
builtins have been ruled out, bash searches the elements of `$PATH` — in list
order, left to right — for "a directory containing an executable file by that
name", and executes the program that search resolves. A name is therefore
resolved to exactly one directory, so a matching executable in an earlier
`$PATH` element prevents the search from ever reaching a same-named executable
installed in a later element: the earlier entry shadows it. Matching is
executable-specific — a directory holding a non-executable file of that name is
not a match and the search proceeds — which is what makes the claim's "first
*matching executable*" wording right.

Two qualifiers from the same page, both consistent with the claim rather than
against it: (1) `$PATH` is consulted only after the function and builtin checks,
so a shell function or builtin of the same name shadows *every* `$PATH` entry,
not just later ones; and (2) bash "uses a hash table to remember the full
pathnames of executable files to avoid multiple PATH searches" and "performs a
full search of the directories in `$PATH` only if the command is not found in
the hash table" — so within one shell instance, a name resolved before `$PATH`
changed (or before a shadowing file was installed) keeps executing the remembered
path until the table is cleared (`hash -r` / `rehash`). Shadowing by an earlier
entry is thus order-governed but not guaranteed to take effect in an existing
shell at the moment the earlier file appears. The manual page states the ordered
search and the single resolution explicitly; "first match wins" is the direct
consequence of that search terminating at a match, not a separate sentence.

## Source
GNU Bash Reference Manual, §3.7.2 "Command Search and Execution" —
https://www.gnu.org/software/bash/manual/html_node/Command-Search-and-Execution.html
Consulted bytes cached locally at
`${TMPDIR}assumption-sources/Command-Search-and-Execution.html` (fingerprint
below). Key text: "If the name is neither a shell function nor a builtin, and
contains no slashes, Bash searches each element of $PATH for a directory
containing an executable file by that name. Bash uses a hash table to remember
the full pathnames of executable files to avoid multiple PATH searches … Bash
performs a full search of the directories in $PATH only if the command is not
found in the hash table." … "If the search is successful, or if the command name
contains one or more slashes, the shell executes the named program in a separate
execution environment."
Documentation-only claim: no source root or dependency root was supplied, so the
published manual page is the source. No project code, no test suite, and no
execution of bash or any probe was used (P1). A supplementary manual page
defining `$PATH` ordering verbatim was attempted
(`…/html_node/Definitions.html`, `…/html_node/Bash-Variables.html`) and the
fetches timed out; the verdict rests on the §3.7.2 page above alone.

## Source Version
Bash Reference Manual, Edition 5, documenting bash version 5.3 (version string
present in the fetched page). Section numbering `3.7.2` matches that edition.

## Source Fingerprint
sha256:8182c7cba5445d0a623b60860c8b8ebc65a09dce273a068cd0d1182f31e5c184

## Invalidates On
- The `Command-Search-and-Execution.html` page (Edition 5 / bash 5.3 text above)
  changes how `$PATH` is searched — e.g. a last-match or all-match resolution, or
  removal of the function/builtin-before-`$PATH` ordering.
- A bash release whose manual states that `$PATH` entries are resolved
  differently (a `PATHORDER`-style option, or hash-table behaviour that rechecks
  later entries before honouring an earlier one) would defeat the shadowing half.
- The claim being re-scoped to a shell that resolves commands differently, or to
  a name containing a slash (then `$PATH` is not consulted at all and the claim
  does not apply).
- Re-scoping to a wrapper that resolves the command itself (a launcher, `env`/`xargs`
  with an explicit path, or an absolute-path invocation), where bash's `$PATH`
  search is bypassed and earlier entries shadow nothing.

## Checked At
2026-10-05T01:51:01Z
