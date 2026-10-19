VERDICT: HOLDS
PROMOTION: ELIGIBLE

## Claim
bash's `[[ -n s ]]` conditional expression is true only for a non-empty string

## What it actually does
Bash's `-n` unary conditional primary is true exactly when the length of its operand
is non-zero, and its companion `-z string` is true exactly when that length is zero —
so `-n` is false only for the empty string and true for every non-empty one.
The bare form `s` (operand with no operator) is documented in the same entry as
equivalent to `-n string`, and inside `[[ ... ]]` the words between the brackets do
not undergo word splitting or filename expansion, so a literal `s` is a one-character,
non-zero-length string and the expression as written evaluates true on that ground.
Nothing in the primary's truth depends on the operand's content beyond its length.

## Source
https://www.gnu.org/software/bash/manual/html_node/Bash-Conditional-Expressions.html
(section 6.4 "Bash Conditional Expressions"; primary listed as "`-n string` / `string`
— True if the length of string is non-zero.", with `-z string` immediately above it).
Consulted bytes: the HTML retrieved from that exact URL (13,125 bytes); the quoted
sentence was verified present in those bytes (it is split by `<em>` markup, so it must
be read with tags stripped).

## Source Version
The GNU Bash Reference Manual, Edition 5.3, last updated 18 May 2025, for Bash
version 5.3 (stated in the fetched document's header comment).

## Source Fingerprint
sha256:401cf8ecf01b3aeb29423a6c5fb6f971083fbb733d4e4020261a6b022e9709fd

## Invalidates On
- Any regeneration/edit of that manual node that changes the `-n`/`-z` wording (the
  fingerprint stops matching) — re-read before reuse.
- A Bash release that changes unary string-primary semantics, or that lets `[[ ]]`
  collapse an empty expansion into a different truth value for `-n`.
- Reuse of this evidence against a shell other than Bash (`sh`, `dash`, `ksh`, `zsh`),
  or against the `test`/`[` builtins, whose behaviour the manual says is determined by
  argument count and which are documented separately.

## Checked At
2026-10-05T01:47:58Z
