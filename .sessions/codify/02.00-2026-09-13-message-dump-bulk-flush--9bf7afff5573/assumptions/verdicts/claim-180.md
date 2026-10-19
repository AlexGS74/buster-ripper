VERDICT: HOLDS
PROMOTION: ELIGIBLE

## Claim
`serve_buster_ripper.sh` appends a `--skip-tls-verify` flag to `EXTRA_FLAGS` only when
`SKIP_TLS_VERIFY` equals `1`

## What it actually does
The flag is added by exactly one statement, line 39:
`[[ "${SKIP_TLS_VERIFY}" == "1" ]] && EXTRA_FLAGS+=(--skip-tls-verify)`.
The append is the right-hand side of an `&&` list guarded by a `[[ ]]` string-equality
test, so `EXTRA_FLAGS` receives `--skip-tls-verify` if and only if the expansion of
`"${SKIP_TLS_VERIFY}"` is the two-character-free literal string `1`. When the guard
does not match, the statement performs no append.

Two consequences of the literal-string comparison worth recording, since they are what
"equals `1`" pins down: `SKIP_TLS_VERIFY` values such as `01`, `true`, `yes`, `2` or an
empty value do **not** produce the flag, and the test is `==` against the quoted
literal rather than an arithmetic `-eq`, so it is textual identity with `1` and not
truthiness.

## Source
/Users/alexhubitski/Projects/buster-ripper/serve_buster_ripper.sh:39 (whole file read;
54 lines). Checked against the mutable worktree source root supplied as
`Source root:` — no snapshot was provided.

Checked independently of the other claim in this bundle by verifying the append site is
unique rather than by inference: `EXTRA_FLAGS` is created empty at line 36, is appended
to only at lines 37-42, and is read once at line 54
(`${EXTRA_FLAGS[@]+"${EXTRA_FLAGS[@]}"}`). A search for the literal
`--skip-tls-verify` scoped to this file matches line 39 only, so no other statement
contributes the flag and the line-39 guard is the sole condition on its presence.

## Source Version
Mutable worktree file — no VCS revision consulted (Git metadata is out of scope for
this role). Stability is carried by the byte fingerprint below.

## Source Fingerprint
sha256:c58256fbec793c520720c9e86fee8740653939275b508f50b212ab60bd0908af

## Invalidates On
- Any change to the line-39 guard that broadens the condition (e.g. `!= "0"`,
  `-gt 0`, a `1|true|yes` pattern, or an `-n` non-empty test), which would append the
  flag for values other than `1`.
- Adding any further statement that appends `--skip-tls-verify` to `EXTRA_FLAGS`
  unconditionally or under a different variable's guard.
- Replacing the array `EXTRA_FLAGS` with a plain string, or routing the flag to the
  `exec buster-ripper` invocation at line 54 directly instead of through the array.
- Any change to the bytes of `serve_buster_ripper.sh` (the fingerprint no longer
  matches, so the check must be re-run).

## Checked At
2026-10-05
