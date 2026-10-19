VERDICT: HOLDS
PROMOTION: ELIGIBLE

## Claim
bash's `${VAR:-word}` substitutes `word` when `VAR` is unset or empty

## What it actually does
`${parameter:-word}` is bash's "Use Default Values" form, and bash tests for a
parameter that is *unset or null*: if either holds, the expansion of `word` is
substituted; otherwise the value of `parameter` is substituted. Bash's own manual
documents the two trigger states separately and demonstrates both, including the
empty-string case (`v=` with `${v:-unset-or-null}` → `unset-or-null`) and the
unset case (`unset v` → `unset-or-null`). Bash defines "null" as the empty string
and calls it a valid value ("A parameter is set if it has been assigned a value.
The null string is a valid value."), so "empty" in the claim is exactly the null
case, not a third state.

## Source
GNU Bash Reference Manual, node "Shell Parameter Expansion" — `doc/bashref.texi`
upstream source at
`https://raw.githubusercontent.com/mirror/bash/master/doc/bashref.texi`
(consulted excerpt begins "When not performing substring expansion, using the
forms described below (e.g., `:-`), Bash tests for a parameter that is unset or
null." and the `@item $@{parameter:-word@}` entry with its `@example` block).
Corroborated by the installed-version manual page: `bash(1)`, section PARAMETER
EXPANSION, as published at
`https://manpages.debian.org/testing/bash/bash.1.en.html` —
"${parameter:-word} Use Default Values. If parameter is unset or null, the
expansion of word is substituted. Otherwise, the value of parameter is
substituted." (www.gnu.org was unreachable from this environment, so the upstream
manual was read from a source mirror plus the packaged manual page; no probe of
bash was run — these are documentation-only claims.)

## Source Version
GNU Bash 5.3 (per `doc/version.texi`: `@set EDITION 5.3`, `@set VERSION 5.3`,
updated 18 May 2025; master-branch bytes as fetched). Corroborating manual page:
`bash(1)` from Debian package `bash 5.3-3+b1`.

## Source Fingerprint
sha256:f3d37d57a1061e24d266051de9bd47ffa43dc86584afea11576c535ad2be32d5 (full fetched `doc/bashref.texi`, 415804 bytes)
sha256:13669991f07ceabe7e30f8ee462af6d24176f02e8db1bd905a3f63dbd7b8554d (consulted excerpt, 802 bytes)
sha256:a827e4d7e765538cfbb239118accd7b2d45f610cd474c496e119eb57861fc5e5 (corroborating `bash(1)` man page HTML, 532248 bytes)

## Invalidates On
- The fetched `doc/bashref.texi` bytes change (fingerprint above) or the
  "Use Default Values" entry is reworded away from "unset or null".
- A bash release that changes `:-` to test only for unset (i.e., collapses `:-`
  onto `-`), or that redefines "null" so an empty assignment no longer counts.
- The mirror branch is mutable: re-verify against the digest above, or against
  `www.gnu.org/software/bash/manual/html_node/Shell-Parameter-Expansion.html`
  when reachable.

## Checked At
2026-10-04
