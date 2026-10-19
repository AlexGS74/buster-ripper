VERDICT: DOES-NOT-HOLD
PROMOTION: ELIGIBLE

## Claim
No symbol under `buster_ripper/` owns the collapsed-artifact role — no artifact filename constant, no per-session append target

## What it actually does
A per-session append target exists under `buster_ripper/`, owned by `dump_turn` (buster_ripper/dump.py): dump.py:104 computes `index = state.session_dir / "index.txt"`, where `session_dir = config.DUMP_DIR / sid` (dump.py:71) — one directory per session — then dump.py:105 opens that file in append mode (`open("a", encoding="utf-8")`) and dump.py:106 writes `summary + "\n"` on every turn. The result is a per-session file that accumulates one line per turn for the session's lifetime: precisely a per-session append target. The claim's "no per-session append target" half is therefore false, and the claim as a whole does not hold, regardless of what "collapsed" modifies — the claim's own operationalization is contradicted by the code.

The "no artifact filename constant" half is true: artifact filenames are inline literals — "turn_000.json" (dump.py:81), f"turn_{turn:03d}.diff" (dump.py:94), "index.txt" (dump.py:104) — and no named filename constant exists anywhere in the package (config.py defines only `DUMP_DIR`, the directory root; a package-wide sweep for filename-bearing constants matched only DUMP_DIR wiring in main.py:89–90 and routes/messages.py:55). The failure on the append-target half is enough for DOES-NOT-HOLD; the filename-constant half is reported true so it is not silently certified either way.

## Source
buster_ripper/dump.py:104–106 (per-session append target: index.txt opened "a" each turn), with dump.py:71 (session_dir = DUMP_DIR / sid) and dump.py:81/:94 (inline filename literals); buster_ripper/config.py:16 (DUMP_DIR is a directory root, not a filename constant)

## Source Version
repo working tree as consulted on 2026-09-19 (pyproject.toml declares project version 0.1.0)

## Source Fingerprint
sha256:243b21434c14e6ea8144329fe1d0ac4e0c499ed9a7e6e09bdcac5e2417a56b1d buster_ripper/dump.py; supporting: sha256:14ac40180f081d8e7caeb280d9a9ab194cbd06c93e3958ebf9bef362ca7b7aab buster_ripper/config.py

## Invalidates On
- removal or renaming of the index.txt append path/logic at dump.py:104–106, or of the per-session dir computation at dump.py:71
- artifact filename literals at dump.py:81/:94/:104 extracted into a named constant
- a new symbol under buster_ripper/ taking over artifact naming or per-session append ownership

## Checked At
2026-09-19T14:59:49Z
