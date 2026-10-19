VERDICT: HOLDS
PROMOTION: ELIGIBLE

## Claim
`_set_skip_tls_verify` assigns `config.SKIP_TLS_VERIFY` from the option value

## What it actually does
Lines 30-33 define `_set_skip_tls_verify(ctx, param, value)` whose body is `config.SKIP_TLS_VERIFY = value` (then `return value`). Under click's callback protocol the `value` parameter is the parsed `--skip-tls-verify` option value (a bool, since the option is `is_flag=True`), so the config global is assigned directly from the option value at parse time. The target attribute `SKIP_TLS_VERIFY` exists at `config` module top level (initialized `False`), and no `skip_tls_verify` parameter is passed to `main()` because the option sets `expose_value=False`.

## Source
buster_ripper/main.py lines 30-33 (callback) and 64-65 (wired via `callback=_set_skip_tls_verify` on the `--skip-tls-verify` option); corroborated by buster_ripper/config.py line 17 (`SKIP_TLS_VERIFY: bool = False`)

## Source Version
Working tree of /Users/alexhubitski/Projects/buster-ripper (no revision read; fingerprint below); click 8.5.0 installed at /opt/homebrew/lib/python3.14/site-packages/click

## Source Fingerprint
sha256:fd36dc12055f67693d9b852fb15a66aa4a3a7864103ac3de0ee67d9520048d47 (buster_ripper/main.py); corroborating buster_ripper/config.py sha256:22348c68d21b91a30e75eeea802d4d390b2bcdabbf974b0ddc743c9354635522

## Invalidates On
The callback is deleted, renamed, or its body no longer assigns `config.SKIP_TLS_VERIFY` from its `value` parameter
`SKIP_TLS_VERIFY` is renamed or moved off the `config` module's top level
The `callback=_set_skip_tls_verify` wiring is removed from the `--skip-tls-verify` option

## Checked At
2026-09-28
