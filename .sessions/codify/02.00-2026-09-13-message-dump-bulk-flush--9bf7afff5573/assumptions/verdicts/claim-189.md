VERDICT: HOLDS
PROMOTION: ELIGIBLE

## Claim
An unquoted bash parameter expansion word-splits, and an empty value contributes zero words

## What it actually does
Both halves are the documented default rule. (a) The shell scans the results of parameter
expansion that did **not** occur within double quotes for word splitting, treating each
character of `$IFS` as a field terminator — so an unquoted `$var` holding `a b` becomes two
words, and its contents are then also eligible for pathname expansion ("bash scans each word
for the characters `*`, `?`, and `[` … if one of these characters appears, and is not quoted,
then the word is regarded as a pattern"). (b) An empty unquoted expansion yields no field at
all: "Unquoted implicit null arguments, resulting from the expansion of parameters that have
no values, are removed", and, since word splitting strips leading/trailing IFS whitespace and
"null fields result only from quoting", an empty result leaves zero words. That removal is not
an IFS artifact — the manual states it explicitly for the degenerate case: "If the value of
IFS is null, no word splitting occurs, but implicit null arguments … are still removed."

Documented scope limits (do not contradict the rule, they bound where it is applied): word
splitting is a step of the expansion list, so it is not performed in contexts the manual
exempts by construction — the right-hand side of an assignment ("Word splitting and pathname
expansion are not performed"), the words inside `[[ … ]]` ("do not undergo word splitting and
pathname expansion"), and arithmetic evaluation; and `set -f` suppresses only the later
pathname-expansion step, not the splitting. Explicitly quoted nulls (`""`, `''`) are the
contrast case: they are retained and passed as empty strings.

## Source
https://man7.org/linux/man-pages/man1/bash.1.html — man page sections EXPANSION → "Word
Splitting" ("The shell scans the results of parameter expansion, command substitution, and
arithmetic expansion that did not occur within double quotes for word splitting"; "Unquoted
implicit null arguments, resulting from the expansion of parameters that have no values, are
removed"; "null fields result only from quoting") and EXPANSION → "Pathname Expansion"; plus
PARAMETERS ("A parameter is set if it has been assigned a value. The null string is a valid
value."). Quoted text verified present in the fetched document. Same text is GNU Bash
Reference Manual §3.5.7 "Word Splitting"
(https://www.gnu.org/software/bash/manual/html_node/Word-Splitting.html — fetch timed out in
this run; the mirrored man page carries the identical wording).

## Source Version
GNU Bash 5.3 man page (page header "GNU Bash 5.3 / 2025 April 7"); man7 copy obtained from
upstream git://git.savannah.gnu.org/bash.git on 2026-08-03 (most recent upstream commit
2026-06-10). Not the locally installed shell — bash was not executed for this check.

## Source Fingerprint
sha256:e35fa2966c44a4e9ee519760160f1aba14c670451a463177968ed2c4710bf846 (exact fetched bytes,
https://man7.org/linux/man-pages/man1/bash.1.html)
sha256:b44556511b41581c65d6dc2505ca306472c6b7f4a75c1c056c1f155ffe1010cb (tag-stripped,
whitespace-normalized text of those same bytes, from which the quotations were read)

## Invalidates On
- A bash release that changes Word Splitting (e.g. an explicit null-argument retention change
  or a new shell option that suppresses splitting of unquoted expansions) — recheck the
  "Word Splitting" section of the Bash 5.4+ manual.
- A manual revision that rewords "Unquoted implicit null arguments … are removed" or moves the
  IFS-null sentence (fingerprint above changes).
- Reuse against a shell that is not GNU bash (dash/sh/busybox ash) or a bash running in posix
  mode with `--posix`, where null-argument handling is specified by POSIX rather than by this
  section.
- Reuse in a context the manual exempts (assignment RHS, `[[ ]]`, arithmetic), where the
  splitting step does not run at all.

## Checked At
2026-10-05T04:46:10Z
