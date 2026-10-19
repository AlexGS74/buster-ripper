VERDICT: HOLDS
PROMOTION: ELIGIBLE

## Claim
bash's `${VAR-word}` on an unset `VAR` raises no nounset error under `set -u`

## What it actually does
The unset-parameter error is raised at exactly one gate in brace expansion, and the
default-value operators are not among the forms that pass the gate — so `${VAR-word}`
with `VAR` unset simply substitutes `word`. In `subst.c:function parameter_brace_expand`,
the parameter name is first looked up with unbound errors suppressed
(`local_pflags |= PF_IGNUNBOUND|...`), yielding a null result rather than a diagnostic.
The only subsequent error site is guarded by the comment "All the cases where an
expansion can possibly generate an unbound variable error" and the condition

```c
if (want_substring || want_patsub || want_casemod || c == '@' || c == '#' || c == '%' || c == RBRACE)
  { if (var_is_set == 0 && unbound_vars_is_error && ...) err_unboundvar (name); }
```

`c` is the operator character following the name. For `${VAR-word}` that is `'-'`; for
`${VAR:-word}` the parser consumes the `:` (`check_nullness++`, per
`if (c == ':' && VALID_PARAM_EXPAND_CHAR (string[sindex]))`) and leaves `c == '-'`.
Neither `'-'` nor a colon-default sets `want_substring`/`want_patsub`/`want_casemod`, so
the gate is false and `err_unboundvar` is never reached. Control falls into
`case '-': case '=': case '?': case '+':`, takes the `else /* VAR not set or VAR is
NULL. */` branch, and — since only `'='` (for special params) and `'?'` report errors
there, and `'+'` expands to nothing — substitutes the default by calling
`parameter_brace_expand_rhs`. `${VAR}` plain (`c == RBRACE`) *does* error, which is the
distinction the claim is about.

**Finding worth carrying:** bash's own documentation never states this exemption. `set -u`
is documented only as "Treat unset variables ... as an error when performing parameter
expansion", and "nounset" appears once in the manual (the `set -o` list). The default-value
forms are documented as *testing* for unset/null and `${parameter:?word}` is the only form
documented to display an error, so the manual is consistent with the behaviour but does not
establish it — the guarantee lives in the implementation, not the documentation.

## Source
GNU Bash source, `subst.c`, function `parameter_brace_expand` (the
`PF_IGNUNBOUND` lookup, the "All the cases where an expansion can possibly generate an
unbound variable error" gate, and the `case '-': case '=': case '?': case '+':` block), at
the released tag
`https://raw.githubusercontent.com/mirror/bash/refs/tags/bash-5.3/subst.c`;
identical gate text also present at
`https://raw.githubusercontent.com/mirror/bash/master/subst.c` and
`https://raw.githubusercontent.com/mirror/bash/refs/tags/bash-5.1/subst.c`.
Secondary (documentation, for what the manual does and does not say):
`bash(1)` at `https://manpages.debian.org/testing/bash/bash.1.en.html` and
`doc/bashref.texi` at `https://raw.githubusercontent.com/mirror/bash/master/doc/bashref.texi`.
Consulted because the docs are silent on this interaction (source is rank-2 evidence for a
library claim). No probe was run — read-only. `mirror/bash` verified as GNU Bash via
`patchlevel.h` ("This file is part of GNU Bash, the Bourne Again SHell").

## Source Version
GNU Bash 5.3 (`refs/tags/bash-5.3`), plus master (5.4-devel, manual edition 5.3, updated
18 May 2025) and bash-5.1 for cross-version stability of the gate condition.

## Source Fingerprint
sha256:cf96a7f33e7f9281f18c7b02d8840ad2d817f14243dd38377f8090249a7edf85 (bash-5.3 `subst.c`, 382887 bytes)
sha256:501f0941105ce39281f3c60c8e5853d8e77f1558ca2383e1757b8c2cc0a3ba56 (master `subst.c`, 383471 bytes)
sha256:22012e6279690e372ad9112cca5918ac505ab42bb475022640d76b5b4aec4201 (bash-5.1 `subst.c`, 339642 bytes)
sha256:f3d37d57a1061e24d266051de9bd47ffa43dc86584afea11576c535ad2be32d5 (consulted `doc/bashref.texi`, 415804 bytes)

## Invalidates On
- The `unbound_vars_is_error` gate in `parameter_brace_expand` gaining `'-'`/`':'`
  (or a `check_nullness`-based) clause, or `err_unboundvar` being added to the
  `case '-'` default-value branch — re-check against the bash-5.3 digest above and
  against whatever tag is current.
- Any release that changes the `PF_IGNUNBOUND` treatment of the brace-expansion name
  lookup.
- If bash ever documents the exemption differently, the documentation (not this
  source read) becomes the authority — and note that no bash documentation of this
  behaviour exists as of the digests above.

## Checked At
2026-10-04
