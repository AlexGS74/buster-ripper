VERDICT: HOLDS
PROMOTION: ELIGIBLE

## Claim
Under `set -u`, bash's `${array[@]+word}` expansion on an unset or empty `array` raises no nounset error

## What it actually does
`set -u` explicitly exempts array variables subscripted with `@` or `*` from the unbound-variable
check, so referencing `${array[@]}` in any expansion form — including the alternate-value `+`
form — is not an error whether the array is unset or declared with no elements. Independently,
the alternate-value operator never substitutes the parameter's value ("The value of `parameter`
is not used"; when the parameter is unset/null "nothing is substituted"), so there is no value
lookup for nounset to complain about: the shell performs an existence test and emits nothing.
An empty array is "not considered set" by the same manual (set requires an assigned subscript),
so it takes the identical no-error path as an unset array.

## Source
https://raw.githubusercontent.com/mirror/bash/bash-5.3/doc/bashref.texi — @subsection "The Set
Builtin", `@item -u` (byte offset 221154): "Treat unset variables and parameters other than the
special parameters @samp{@@} or @samp{*}, **or array variables subscripted with @samp{@@} or
@samp{*},** as an error when performing parameter expansion."

Established independently of claim-203 (different node, different sentence). Two further facts
from the same file support the empty-array half without being load-bearing: the `@item -u`
exemption is version-dated below, and the Shell Parameter Expansion `:+` item ("nothing is
substituted ... The value of @var{parameter} is not used", offset 86495 ff.).

Version boundary, from the upstream changelog (corroboration of when this became true, not the
basis of the verdict): https://raw.githubusercontent.com/mirror/bash/bash-4.4/CHANGES,
"3. New Features in Bash / a. Using ${a[@]} or ${a[*]} with an array without any assigned
elements when the nounset option is enabled no longer throws an unbound variable error."
The `-u` sentence's array clause is present in the bash-5.2 and bash-5.3 manuals and absent
from the bash-4.3, 4.4, 5.0 and 5.1 copies of `doc/bashref.texi` (checked at each tag; the
prose caught up with the 4.4 behaviour change).

## Source Version
bash-5.3 (`doc/bashref.texi`, 415,804 bytes); clause also present at tag `bash-5.2`
(374,486 bytes); absent at tags bash-4.3 / bash-4.4 / bash-5.0 / bash-5.1; behavioural change
recorded for bash-4.4 (`CHANGES`, 357,491 bytes)

## Source Fingerprint
sha256:f3d37d57a1061e24d266051de9bd47ffa43dc86584afea11576c535ad2be32d5  (bash-5.3 doc/bashref.texi, exact fetched bytes consulted)
sha256:32ced60bb645d47be7bfa4e5767edb2ea1b0c840d8968c17f3eb9df960b44863  (bash-5.2 doc/bashref.texi, same clause)
sha256:992ae56e662f50e206ae8ed492cf9b1d8c54d7bf2f307e60a4153cd1f7191a7b  (bash-4.4 CHANGES, version-boundary entry)

## Invalidates On
- A bash release whose `The Set Builtin` `-u` item drops the clause "or array variables
  subscripted with @ or *"
- Running on a shell predating bash-4.4 (e.g. macOS `/bin/bash` 3.2), where nounset did report
  unbound-variable errors for `@`-subscripted arrays; the manuals for 4.3 and earlier do not
  document an array exemption, so the verdict must be re-checked per installed version
- Any documented nounset change that applies the check to the *test* forms (`+`, `-`, `?`, `=`)
  rather than only to value substitution
- `set -u` semantics under `set -o posix`, or `nounset` interaction with associative arrays
  referenced without a subscript, if the target code shape differs from an `@`-subscripted array
- Claim being extended to a subscripted *scalar* `${array[key]+word}` on an unset plain variable
  — that is a different nounset path and is not covered here

## Checked At
2026-10-05T05:11:24Z
