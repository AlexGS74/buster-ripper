VERDICT: HOLDS
PROMOTION: ELIGIBLE

## Claim
`dump.py` imports `json` and `difflib`

## What it actually does
`buster_ripper/dump.py` imports both modules in its import block: `import difflib` on line 4 and `import json` on line 5. Both are also used in the file body — `json.dumps` (lines 34, 50, 82) and `difflib.unified_diff` (line 89) — corroborating that these are live imports, not vestigial lines.

## Source
buster_ripper/dump.py, lines 3–5 (import block)

## Source Version
unavailable — working-tree file; no revision resolved (repo file, not an installed package)

## Source Fingerprint
sha256:243b21434c14e6ea8144329fe1d0ac4e0c499ed9a7e6e09bdcac5e2417a56b1d

## Invalidates On
- `import json` or `import difflib` removed from the import block of buster_ripper/dump.py
- Any edit to buster_ripper/dump.py that changes its bytes (sha256 mismatch against the fingerprint above)

## Checked At
2026-09-19
