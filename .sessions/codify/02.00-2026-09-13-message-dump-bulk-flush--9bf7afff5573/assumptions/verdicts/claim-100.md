VERDICT: HOLDS
PROMOTION: ELIGIBLE

## Claim
`main.py` exposes `--dump-flush-interval` and `--dump-flush-bytes` as callback-backed numeric options

## What it actually does
The `main` click command declares `--dump-flush-interval` (lines 86–90) and `--dump-flush-bytes` (lines 91–95), both `type=click.IntRange(min=1)` (numeric), both `expose_value=False` with parse-time callbacks: `callback=_set_dump_flush_interval` and `callback=_set_dump_flush_bytes`, defined at lines 18–27, which assign `config.DUMP_FLUSH_INTERVAL` / `config.DUMP_FLUSH_BYTES` and return the value. Defaults come from those config globals (`10` seconds and `4 * 1024 * 1024` bytes, config.py lines 36–37).

## Source
buster_ripper/main.py, lines 18–27 and 86–95 (defaults corroborated by buster_ripper/config.py lines 36–37)

## Source Version
Working tree at /Users/alexhubitski/Projects/buster-ripper (no revision string read; git metadata out of scope)

## Source Fingerprint
sha256:fd36dc12055f67693d9b852fb15a66aa4a3a7864103ac3de0ee67d9520048d47 (buster_ripper/main.py)

## Invalidates On
- Either option is removed, renamed, or loses its `callback=` in buster_ripper/main.py
- Either option's type stops being numeric (e.g. changed off `click.IntRange`)

## Checked At
2026-09-28
