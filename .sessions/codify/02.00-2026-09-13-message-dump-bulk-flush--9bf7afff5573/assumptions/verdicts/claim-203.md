VERDICT: HOLDS
PROMOTION: ELIGIBLE

## Claim
bash's `${array[@]+word}` alternate-value expansion contributes zero words when `array` is unset or has no elements

## What it actually does
For the colon-less alternate-value form, bash tests only whether the parameter exists: if it
does not, nothing is substituted — so no word is emitted (quoted or unquoted). Bash defines an
array variable as set only when at least one subscript has been assigned a value, so an array
with no elements is not set for the purposes of that test, and `${name[@]}` with no members
"expands to nothing". Both stated cases — unset array, and array with no elements — therefore
contribute zero words; `word` appears only when the array has at least one assigned element
(the value of the array itself is never used by this operator).

## Source
https://raw.githubusercontent.com/mirror/bash/bash-5.3/doc/bashref.texi (GNU Bash Reference
Manual source, tag `bash-5.3`; official upstream tree mirrored at bminor/mirror "bash").
Three passages, all in that one file:

- @subsection Shell Parameter Expansion (byte offset 82361 ff.): "When not performing substring
  expansion, using the forms described below (e.g., `:-'), Bash tests for a parameter that is
  unset or null. Omitting the colon results in a test only for a parameter that is unset."
- Same node, `@item $@{@var{parameter}:+@var{word}@}`: "If @var{parameter} is null or unset,
  nothing is substituted, otherwise the expansion of @var{word} is substituted. The value of
  @var{parameter} is not used."
- @section Arrays, offset 313531 and 314676: "When there are no array members,
  `$@{name[@]@}` expands to nothing." / "An array variable is considered set if a subscript has
  been assigned a value. The null string is a valid value."

Derivation is two-sentence (existence test from the Parameter Expansion node + the Arrays
node's definition of array set-ness); no third-party source was needed and no probe was run.
Corroborating, not load-bearing: the same three sentences are present at tag `bash-5.2`
(byte offsets 285377 / 286501 and the same `:+` item).

## Source Version
bash-5.3 (`doc/bashref.texi`, 415,804 bytes, fetched 2026-10-05); cross-checked against the
bash-5.2 copy of the same file (374,486 bytes)

## Source Fingerprint
sha256:f3d37d57a1061e24d266051de9bd47ffa43dc86584afea11576c535ad2be32d5  (bash-5.3 doc/bashref.texi, the exact fetched bytes consulted)
sha256:32ced60bb645d47be7bfa4e5767edb2ea1b0c840d8968c17f3eb9df960b44863  (bash-5.2 doc/bashref.texi, corroboration)

## Invalidates On
- A bash release whose `doc/bashref.texi` no longer contains "An array variable is considered
  set if a subscript has been assigned a value" in the Arrays node, or drops "Omitting the
  colon results in a test only for a parameter that is unset" from Shell Parameter Expansion
- A documented change making the colon-less `+` operator test the *variable's existence* rather
  than the array's assigned-subscript set-ness (i.e. `arr=()` starting to substitute `word`)
- Applying the claim to bash < 5.2-era semantics on a specific installed shell (e.g. macOS
  /bin/bash 3.2), where the nounset interaction differs — see claim-204; re-check per version
- Non-bash shells (zsh, ksh, dash) — this verdict is bash-specific
- Any change in what the claim means by "no elements": an array whose last element was
  `unset`-ed still exists as a variable ("Unsetting the last element of an array variable does
  not unset the array"), which the manual does not tie to the `+` test explicitly

## Checked At
2026-10-05T05:11:24Z
