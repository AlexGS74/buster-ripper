VERDICT: HOLDS
PROMOTION: ELIGIBLE

## Claim
The `click.option` defaults in `buster_ripper/main.py` that refer to `config` attributes are bound when `main.py` is imported

## What it actually does
Exactly three `click.option` defaults reference `config` attributes — `default=config.UPSTREAM` (line 62), `default=config.DUMP_FLUSH_INTERVAL` (line 87), and `default=config.DUMP_FLUSH_BYTES` (line 92) — and all three sit inside module-level decorator expressions on `main`, so Python evaluates them once when the module body executes, i.e. at import time. The resulting values are stored on the click Option objects at decoration; later mutations of `config.UPSTREAM`/`config.DUMP_FLUSH_INTERVAL`/`config.DUMP_FLUSH_BYTES` after import would not change the CLI defaults. (Other config references in the file — `type=click.Choice(list(config.DUMP_FORMATS))`, `type=click.Choice(list(config.PROFILES.keys()))`, and the `--eval-profile` help f-string — are `type`/`help`, not `default`, and are likewise import-time evaluated.)

## Source
buster_ripper/main.py lines 62, 86-95 (decorated `main` at lines 61-131); corroborated by buster_ripper/config.py lines 13, 36, 37 (module-level `UPSTREAM`, `DUMP_FLUSH_INTERVAL`, `DUMP_FLUSH_BYTES`)

## Source Version
Working tree of /Users/alexhubitski/Projects/buster-ripper (no revision read; fingerprint below); click 8.5.0 installed at /opt/homebrew/lib/python3.14/site-packages/click

## Source Fingerprint
sha256:fd36dc12055f67693d9b852fb15a66aa4a3a7864103ac3de0ee67d9520048d47 (buster_ripper/main.py); corroborating buster_ripper/config.py sha256:22348c68d21b91a30e75eeea802d4d390b2bcdabbf974b0ddc743c9354635522

## Invalidates On
Any of the three `default=config.*` expressions is replaced with a literal, a callable, or a lookup deferred to parse time
`config.UPSTREAM` / `DUMP_FLUSH_INTERVAL` / `DUMP_FLUSH_BYTES` is renamed or moved off the `config` module's top level
The options are relocated from module-level decorators into a factory invoked at run time rather than import time

## Checked At
2026-09-28
