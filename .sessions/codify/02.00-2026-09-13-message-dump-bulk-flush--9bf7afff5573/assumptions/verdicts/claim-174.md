VERDICT: HOLDS
PROMOTION: ELIGIBLE

## Claim
`buster_ripper/config.py` declares the module global `SKIP_TLS_VERIFY` with default `False`

## What it actually does
Module-level statement `SKIP_TLS_VERIFY: bool = False` at `buster_ripper/config.py:17`, inside the "Core proxy settings" block. The module docstring states all other modules import it and read `config.<NAME>` so mutations by `main()` are visible everywhere, confirming its role as a mutable module global; the adjacent comment documents the `False` default as "certificates are verified unless the opt-out is explicitly requested" via `--skip-tls-verify` / `SKIP_TLS_VERIFY=1`.

## Source
buster_ripper/config.py, line 17 (local production source in working tree)

## Source Version
Local repo source, working tree at /Users/alexhubitski/Projects/buster-ripper (no package version string; git metadata out of scope for this check)

## Source Fingerprint
sha256:22348c68d21b91a30e75eeea802d4d390b2bcdabbf974b0ddc743c9354635522

## Invalidates On
The `SKIP_TLS_VERIFY` declaration at config.py:17 is removed or renamed
Its default initializer changes from `False` to any other value
The declaration moves out of `buster_ripper/config.py` (e.g., into a settings module or main)
The file bytes change such that the sha256 above no longer matches the consulted content

## Checked At
2026-09-28
