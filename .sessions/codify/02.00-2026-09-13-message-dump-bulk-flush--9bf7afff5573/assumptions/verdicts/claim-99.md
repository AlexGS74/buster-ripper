VERDICT: HOLDS
PROMOTION: ELIGIBLE

## Claim
`main.py` offers `--dump-format` alongside `--dump-dir`

## What it actually does
The `main` click command declares `--dump-dir` (lines 78–79, `metavar="PATH"`, default `None`) and, directly alongside it, `--dump-format` (lines 80–85) with `type=click.Choice(list(config.DUMP_FORMATS))`, default `"documented"`, `expose_value=False`, and `callback=_set_dump_format`, which assigns `config.DUMP_FORMAT` at parse time. The choice values exist in `buster_ripper/config.py` line 27: `("documented", "change-focused", "self-contained")`.

## Source
buster_ripper/main.py, lines 78–85 (corroborated by buster_ripper/config.py line 27)

## Source Version
Working tree at /Users/alexhubitski/Projects/buster-ripper (no revision string read; git metadata out of scope)

## Source Fingerprint
sha256:fd36dc12055f67693d9b852fb15a66aa4a3a7864103ac3de0ee67d9520048d47 (buster_ripper/main.py)

## Invalidates On
- `--dump-format` or `--dump-dir` is removed or renamed in buster_ripper/main.py
- The `--dump-format` option stops being a sibling option on the `main` click command (moved to another entrypoint)

## Checked At
2026-09-28
