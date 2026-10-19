VERDICT: HOLDS
PROMOTION: ELIGIBLE

## Claim
In bash, a command left of `&&` that fails in an `&&` list does not trigger errexit under `set -e`

## What it actually does
The `-e` bullet of the `set` builtin says the shell exits immediately when a pipeline,
list, or compound command returns non-zero, but then carves out the cases where it does
not: "The shell does not exit if the command that fails is part of the command list
immediately following a while or until reserved word, part of the test in an if
statement, part of any command executed in a && or || list except the command following
the final && or ||, any command in a pipeline but the last (subject to the state of the
pipefail shell option), or if the command's return status is being inverted with !".
In `A && B` the only command "following the final `&&`" is `B`; `A` sits left of that
`&&` and is therefore inside the exemption, so a failing `A` short-circuits the list
without ending the shell — the list still returns `A`'s non-zero status, but errexit
does not fire on it. The same holds for every position except the last in a longer
list: in `A && B && C`, both `A` and `B` are exempt and only `C` can trigger the exit.

## Source
https://www.gnu.org/software/bash/manual/html_node/The-Set-Builtin.html
(section 4.3.1 "The Set Builtin", option `-e` / `errexit` bullet).
Consulted bytes: the HTML retrieved from that exact URL (18,020 bytes); the exemption
clause was verified present in those bytes (`&&`/`||` appear wrapped in `<code>`/`<samp>`
markup, so the sentence must be read with tags stripped).

## Source Version
The GNU Bash Reference Manual, Edition 5.3, last updated 18 May 2025, for Bash
version 5.3 (stated in the fetched document's header comment).

## Source Fingerprint
sha256:ca70b4480745a3310417bc974ef3a4c74a1cb335156557fc114948ec006fa9aa

## Invalidates On
- Any regeneration/edit of that manual node that changes the `-e` exemption clause
  (fingerprint stops matching) — re-read before reuse.
- A Bash release that narrows the exemption so a non-final command in an `&&`/`||` list
  can trigger errexit.
- Reuse where the "left of `&&`" command is itself the final command of an enclosing
  list, or where the list runs inside a compound command/function in a context where
  `-e` is already ignored — the manual's separate paragraphs ("If a compound command or
  shell function executes in a context where -e is ignored…" and "If a compound command
  other than a subshell returns a non-zero status because a command failed while -e was
  being ignored…") govern those, not this exemption.
- Reuse against a shell other than Bash, or against POSIX `sh` errexit wording.

## Checked At
2026-10-05T01:47:58Z
