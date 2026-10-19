VERDICT: HOLDS
PROMOTION: INELIGIBLE — the evidence is absence across a directory enumeration (a broad directory), not exact recoverable bytes; per-file hashes of the current modules would not detect an added eleventh file, so reuse requires re-enumerating tests/

## Claim
`tests/` contains no dump test module today

## What it actually does
tests/ contains exactly eight test modules plus conftest.py and __init__.py; none is named for the dump feature and none references it. The only matches for dump/DUMP/Dump in tests/*.py are `json.dumps` serialization calls in test_session_stats.py, unrelated to the feature. The project's own vocabulary confirms what a "dump test module" would cover: `buster_ripper/dump.py` defines `dump_turn`, `routes/messages.py` imports and calls it, and `main.py` wires a `--dump-dir` option into `config.DUMP_DIR` — no test module exercises any of it, by name or by content.

## Source
tests/ directory enumeration (glob tests/**/*) cross-checked with a case-insensitive grep for dump across tests/*.py; feature vocabulary established from buster_ripper/dump.py (line 61: `def dump_turn`), buster_ripper/routes/messages.py (lines 12, 57), buster_ripper/main.py (lines 25, 88-89). The claim's subject is the test suite layout, so the tests/ files are the production surface here.

## Source Version
unavailable (working tree; git metadata out of scope for this check)

## Source Fingerprint
unavailable — the consulted material is a directory listing plus negative grep results, neither of which is a hashable byte source

## Invalidates On
- any file matching *dump* (any case) appears anywhere under tests/
- any existing tests/*.py module gains a reference to dump_turn, DUMP_DIR, or --dump-dir

## Checked At
2026-09-19
