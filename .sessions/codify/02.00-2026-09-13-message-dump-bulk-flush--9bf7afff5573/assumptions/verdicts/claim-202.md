VERDICT: HOLDS
PROMOTION: ELIGIBLE

## Claim
bash's "${array[@]}" expansion in double quotes contributes exactly one argument word per array element, whatever characters an element contains

## What it actually does
This is the documented defining property of `[@]` under double quotes. The Arrays section
states it directly: "If the word is double-quoted, `${name[*]}` expands to a single word with
the value of each array member separated by the first character of the IFS special variable,
and `${name[@]}` expands each element of name to a separate word." The EXPANSION section makes
the cardinality explicit — word splitting and pathname expansion are the only steps that can
increase word count, and "The only exceptions to this are the expansions of `"$@"` and
`"${name[@]}"`" — i.e. the one-word-per-element count comes from the expansion itself, not from
splitting the elements afterwards. The "whatever characters" half follows from three further
sentences: the elements are produced *inside* double quotes, and "The shell scans the results
of parameter expansion … that did not occur within double quotes for word splitting" (so
spaces/tabs/newlines in an element are never delimiters); pathname expansion treats a word as a
pattern only if `*`, `?`, or `[` appears "and is not quoted" (so glob characters in an element
stay literal); and quote removal strips only "unquoted occurrences of the characters \, ', and
" that did not result from one of the above expansions" (so a quote or backslash *stored in* an
element is not eaten). Each element becomes a separately quoted word, analogous to `"$@"` being
"equivalent to `"$1"` `"$2"` …".

Boundary cases the manual settles consistently with "one word per element": an empty array (or
no members) contributes **zero** words — "When there are no array members, `${name[@]}` expands
to nothing" — while a set-but-null element still contributes one empty word, because expansion
within double quotes "produces a null field, which is retained and passed to a command as an
empty string". Sparse indexed arrays expand their assigned members only. When the expansion sits
inside a larger word, the first and last elements are joined to the surrounding text ("the
expansion of the first parameter is joined with the beginning part of the expansion of the
original word"), which merges text but does not change the count of element-derived words.

## Source
https://man7.org/linux/man-pages/man1/bash.1.html — sections PARAMETERS → Arrays ("`${name[@]}`
expands each element of name to a separate word. When there are no array members, `${name[@]}`
expands to nothing"), PARAMETERS → Special Parameters (`@`: "`"$@"` is equivalent to `"$1"`
`"$2"` …"), EXPANSION ("Quote removal is always performed last. It removes quote characters
present in the original word, not ones resulting from one of the other expansions"; "The only
exceptions to this are the expansions of `"$@"` and `"${name[@]}"`"), EXPANSION → Word
Splitting, and EXPANSION → Pathname Expansion. Quoted text verified present in the fetched
document. Same text is GNU Bash Reference Manual §3.4/§3.5
(https://www.gnu.org/software/bash/manual/html_node/Shell-Parameter-Expansion.html — fetch timed
out in this run; the mirrored man page carries the identical wording).

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
- A bash release changing `[@]` semantics (e.g. a compat change to how empty members or
  no-members expansions are counted) — recheck PARAMETERS → Arrays in the 5.4+ manual.
- A manual revision rewording "expands each element of name to a separate word" or the
  quote-removal sentence (fingerprint above changes).
- Reuse where the construct is not `"${array[@]}"`: `${array[*]}` / `$*` (one joined word),
  unquoted `${array[@]}` (elements then undergo word splitting and pathname expansion), or
  `$array` / `${array[0]}` (single element only).
- Reuse against a non-bash shell (dash/sh/busybox ash) or POSIX-mode bash where array
  expansions are not available or differ, or a context the manual exempts from expansion
  (e.g. inside `[[ ]]`, where "the words … do not undergo word splitting and pathname
  expansion" and quoting behaves differently).

## Checked At
2026-10-05T04:46:10Z
