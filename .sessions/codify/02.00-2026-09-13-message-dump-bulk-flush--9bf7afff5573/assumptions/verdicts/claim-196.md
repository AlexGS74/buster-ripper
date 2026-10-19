VERDICT: HOLDS
PROMOTION: ELIGIBLE

## Claim
`main.py` assigns `config.DUMP_DIR` only inside the branch that tests a truthy `--dump-dir` value

## What it actually does
`buster_ripper/main.py` contains exactly one assignment to `config.DUMP_DIR` — line 152,
`config.DUMP_DIR = Path(dump_dir).expanduser().resolve()` — and it sits inside the
`if dump_dir:` block opened at line 151. `dump_dir` is the click-bound value of the
`--dump-dir` option (declared lines 78-79 with `default=None`, received as the `dump_dir`
parameter at line 132), so `if dump_dir:` is precisely a truthiness test on that option's
value. The only other `config.DUMP_DIR` occurrences in the file are not assignments:
line 153 calls `.mkdir(parents=True, exist_ok=True)` on the already-assigned value, and
line 175 reads it for the startup log line. No click callback in `main.py` and no
statement in `_apply_cli_config` (lines 36-58) touches `DUMP_DIR` — those callbacks set
`DUMP_FORMAT`, `DUMP_FLUSH_INTERVAL`, `DUMP_FLUSH_BYTES`, and `SKIP_TLS_VERIFY` only —
and there is no `setattr`/`globals()` write that could assign it indirectly.

## Source
/Users/alexhubitski/Projects/buster-ripper/buster_ripper/main.py — lines 151-153
(the `if dump_dir:` guard and its single assignment), corroborating non-assignment
references at lines 175 and 36-58, and the option/parameter binding at lines 78-79 and 132.
Read from the supplied source root as production source; no test file was consulted, and
none was needed.

## Source Version
unavailable — no `__version__` or revision marker in `buster_ripper/__init__.py`; verdict
is grounded in the working-tree file bytes fingerprinted below.

## Source Fingerprint
sha256:fd36dc12055f67693d9b852fb15a66aa4a3a7864103ac3de0ee67d9520048d47

## Invalidates On
- Any edit to `main.py` that adds an assignment to `config.DUMP_DIR` outside the
  `if dump_dir:` block (e.g. moving it into `_apply_cli_config`, into a new click
  callback, or to module level) — the word "only" in the claim is what such an edit breaks.
- Any change to how `--dump-dir` is bound so that `if dump_dir:` is no longer a plain
  truthiness test on the option value: e.g. `expose_value=False` with a callback doing the
  assignment, a truthy non-`None` default, or a guard rewritten as `if dump_dir is not None`
  (which would admit a falsy-but-present value and change what "truthy" means here).

## Checked At
2026-10-05T01:56:06Z
