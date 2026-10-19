VERDICT: HOLDS
PROMOTION: ELIGIBLE

## Claim
bash's `${VAR-word}` substitutes `word` only when `VAR` is unset

## What it actually does
The no-colon forms test only for *existence*: bash states that where the colon
forms test "unset or null", "Omitting the colon results in a test only for a
parameter that is unset ... if the colon is omitted, the operator tests only for
existence." So `${VAR-word}` substitutes `word` when `VAR` is unset, and
substitutes the value of `VAR` otherwise — including when that value is the empty
string. The manual's worked example for the default-value entry demonstrates all
three states side by side: `v=123` → `${v-unset}` prints `123`; `unset v` →
`${v-unset}` prints `unset`; `v=` (set and empty) → `${v-unset}` prints nothing,
while `${v:-unset-or-null}` at that same value prints the default. That last pair
is what makes the "only" in the claim true rather than merely implied: empty-but-set
does not trigger the no-colon form.

## Source
GNU Bash Reference Manual, node "Shell Parameter Expansion" — `doc/bashref.texi`
upstream source at
`https://raw.githubusercontent.com/mirror/bash/master/doc/bashref.texi`: the
paragraph "When not performing substring expansion, using the forms described
below (e.g., `:-`), Bash tests for a parameter that is unset or null. Omitting the
colon results in a test only for a parameter that is unset." plus the
`@item $@{parameter:-word@}` `@example` block that exercises `${v-unset}` against
`v=123`, unset `v`, and `v=`.
Corroborated by the installed-version manual page: `bash(1)`, section PARAMETER
EXPANSION, at
`https://manpages.debian.org/testing/bash/bash.1.en.html` — "When not performing
substring expansion, using the forms documented below (e.g., :-), bash tests for a
parameter that is unset or null. Omitting the colon tests only for a parameter
that is unset." (www.gnu.org unreachable from this environment; documentation-only
claims, so read from the manual source and the packaged manual page — no bash
probe was run.)

## Source Version
GNU Bash 5.3 (per `doc/version.texi`: `@set EDITION 5.3`, `@set VERSION 5.3`,
updated 18 May 2025; master-branch bytes as fetched). The prose rule is identical
in the `bash(1)` page from Debian package `bash 5.3-3+b1`; only the worked
`@example` block is new in the current manual.

## Source Fingerprint
sha256:f3d37d57a1061e24d266051de9bd47ffa43dc86584afea11576c535ad2be32d5 (full fetched `doc/bashref.texi`, 415804 bytes)
sha256:13669991f07ceabe7e30f8ee462af6d24176f02e8db1bd905a3f63dbd7b8554d (consulted excerpt, 802 bytes)
sha256:a827e4d7e765538cfbb239118accd7b2d45f610cd474c496e119eb57861fc5e5 (corroborating `bash(1)` man page HTML, 532248 bytes)

## Invalidates On
- The fetched `doc/bashref.texi` bytes change (fingerprint above), or the
  "Omitting the colon results in a test only for a parameter that is unset"
  sentence is reworded.
- A bash release that makes the no-colon forms also test for null (collapsing
  `-` onto `:-`), or that changes which forms the "when not performing substring
  expansion" qualifier covers.
- The mirror branch is mutable: re-verify against the digest above, or against
  `www.gnu.org/software/bash/manual/html_node/Shell-Parameter-Expansion.html`
  when reachable.

## Checked At
2026-10-04
