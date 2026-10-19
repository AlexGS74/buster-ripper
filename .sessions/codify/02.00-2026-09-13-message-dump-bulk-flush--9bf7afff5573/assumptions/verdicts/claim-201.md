VERDICT: HOLDS
PROMOTION: ELIGIBLE

## Claim
`tests/test_serve_script_argv.py` asserts the complete assembled argv word list for each environment state it exercises

## What it actually does
Each of the four environment states the file runs ends in a whole-list `==` comparison against a fully
enumerated Python list — no membership-only, prefix, slice, or subset assertion carries any of the verdicts.
The left side of every comparison is genuinely the complete word list: the PATH shim records every positional
word with `printf '%s\0' "$@"`, and `_read_argv` decodes all NUL-separated fields (popping only the trailing
empty field produced by the final terminator), so the recorded value is the wrapper's entire assembled argv
rather than a sampled prefix.

The four exercised states and their equality assertions:

| State exercised | Line | Complete-list assertion |
|---|---|---|
| `DUMP_DIR` unset (default) | 111 | `serve_argv() == _TRIO + ["--dump-dir", default] + _THINKING` |
| `DUMP_DIR=""` | 117 | `argv == _TRIO + _THINKING` |
| `DUMP_DIR="/tmp/served-diffs"` | 122 | `argv == _TRIO + ["--dump-dir", "/tmp/served-diffs"] + _THINKING` |
| `DUMP_DIR="" DUMP_FORMAT="change-focused"` | 127 | `argv == _TRIO + ["--dump-format", "change-focused"] + _THINKING` |

`_TRIO` (line 45) enumerates the six always-present words (`--upstream http://localhost:30000 --host 0.0.0.0
--port 30001`) and `_THINKING` (line 46) the trailing `--preserved-thinking`, so the right-hand side spells out
every word the wrapper contributes, in order, in each state. Line 116's `"--dump-dir" not in argv` is an extra
negative assertion inside the `DUMP_DIR=""` case and is redundant to the equality on line 117, not a
substitute for it. `TestDocumentationAgreement` (line 130) asserts against the script's text, not an
environment state, so it is outside the claim's scope.

Two observations that qualify the coverage without contradicting the claim's scope wording. `_CONSUMED`
(lines 29-32) names nine variables the wrapper reads, but the runs override only `DUMP_DIR` and `DUMP_FORMAT`;
in particular no state sets `PRESERVED_THINKING`, so the argv shape with the thinking flag absent is never
exercised, and no state varies `UPSTREAM`/`HOST`/`PORT`/`STRIP_DATE`/`VERBOSE`/`SKIP_TLS_VERIFY`. The claim is
about each state *it exercises*, and every one of those is asserted completely — but "complete argv coverage of
the wrapper" would not follow from this file. Relatedly, because `DUMP_DIR=""` is the only way the flag-absent
shapes arise, all four equalities are simultaneously conditional on the default `PRESERVED_THINKING` state, and
the hardcoded `_TRIO` values pin the expectations to the script's current defaults.

## Source
`/Users/alexhubitski/Projects/buster-ripper/tests/test_serve_script_argv.py` (whole file read; assertion sites
lines 45-46, 111, 117, 122, 127; argv-capture mechanism lines 37-40, 80-102).

Per the P10 exception, the claim's subject IS the test suite, so the test file is the production surface for it
and grounds the verdict rather than merely corroborating one. Bounded to the supplied source root
`/Users/alexhubitski/Projects/buster-ripper/tests`; `serve_buster_ripper.sh` lives above that root and was not
read, and is not needed — the claim is about what this file asserts, not about whether the asserted lists are
correct.

## Source Version
`unavailable` — no VCS metadata consulted (Git metadata boundary). Content identity is carried by the
fingerprint below.

## Source Fingerprint
sha256:461cc81d83c288b0b12488ebcbe3a12ebb68eaffabde283da2bc2c001860be74

## Invalidates On
- Any `TestDumpDirDisableForm` method's whole-list `==` replaced by a partial assertion (membership, `not in`,
  index/slice, or subset check) — that state would then no longer assert the complete argv.
- `_read_argv` or the `_SHIM` recording changed to sample, truncate, reorder, or filter the NUL-separated words,
  making the compared value anything other than the full assembled argv.
- `_TRIO` or `_THINKING` becomes conditional, variadic, or is dropped from an equality's right-hand side, so a
  listed no longer enumerates every word.
- A new environment-state test added that asserts only part of the argv (the claim's "each state" quantifier
  would then fail), or that sets `PRESERVED_THINKING` to a flag-suppressing value without a complete-list
  equality.
- Any edit to these lines that changes the consulted bytes, i.e. anything that changes the fingerprint above.

## Checked At
2026-10-04
