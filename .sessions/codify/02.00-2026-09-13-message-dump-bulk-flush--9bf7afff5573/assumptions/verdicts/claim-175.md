VERDICT: HOLDS
PROMOTION: ELIGIBLE

## Claim
`main.py` declares `--skip-tls-verify` as an `is_flag=True` option with `expose_value=False`

## What it actually does
Line 64-67 declares `@click.option("--skip-tls-verify", is_flag=True, expose_value=False, callback=_set_skip_tls_verify, help=...)` on `main` — a boolean flag whose value is not passed to `main()` as a parameter; instead the callback `_set_skip_tls_verify` receives it at parse time. The claim's two assertions (`is_flag=True`, `expose_value=False`) both match verbatim.

## Source
buster_ripper/main.py lines 64-67

## Source Version
Working tree of /Users/alexhubitski/Projects/buster-ripper (no revision read; fingerprint below); click 8.5.0 installed at /opt/homebrew/lib/python3.14/site-packages/click

## Source Fingerprint
sha256:fd36dc12055f67693d9b852fb15a66aa4a3a7864103ac3de0ee67d9520048d47 (buster_ripper/main.py)

## Invalidates On
The `--skip-tls-verify` option declaration changes — `is_flag` removed, `expose_value` flipped to true, or the option renamed/removed

## Checked At
2026-09-28
