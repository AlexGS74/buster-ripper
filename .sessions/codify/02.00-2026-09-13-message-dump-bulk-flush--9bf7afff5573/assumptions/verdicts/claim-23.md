VERDICT: HOLDS
PROMOTION: ELIGIBLE

## Claim
`--dump-dir` is the only dump-related CLI option in `main.py` today

## What it actually does
`buster_ripper/main.py` defines 21 click options (lines 13–61) and exactly one of them concerns dumping: `--dump-dir` (line 25, "Enable session prompt diffing under PATH/<session_id>/."), wired to `config.DUMP_DIR` at lines 88–89. The string "dump" appears in main.py only at lines 25, 63 (the `dump_dir` parameter), and 88–89 — all that one option. The dump code path corroborates the exclusivity: `dump.py` consumes only `config.DUMP_DIR` (assert at dump.py:63, path use at :71), and the sole caller gates on `config.DUMP_DIR is not None` (routes/messages.py:55) — no other CLI-wired config value feeds dump behaviour. Even under a strained reading where "dump-related" means any option that touches dump execution, nothing else qualifies: dump.py logs at INFO level (`log.info("dump: ...")`), so `--verbose` (which raises the level to DEBUG) does not gate dump logs, and dump output content depends only on the request body and DUMP_DIR. No definitional choice was needed.

## Source
buster_ripper/main.py, the click command's option decorators (lines 12–61) and the `dump_dir` wiring (lines 88–89); corroborated by buster_ripper/dump.py (lines 61–109) and buster_ripper/routes/messages.py line 55.

## Source Version
Repository working tree (uncommitted checkout; no version string available)

## Source Fingerprint
buster_ripper/main.py: sha256:b4794796d28ad3ab9ce5f2e0e03dddcfa882178299c7b969751697993cb76089
buster_ripper/dump.py: sha256:243b21434c14e6ea8144329fe1d0ac4e0c499ed9a7e6e09bdcac5e2417a56b1d

## Invalidates On
- Any dump-related option added to (or `--dump-dir` removed from) main.py's option decorators
- dump.py or its callers starting to consume another CLI-wired config value
- The dump gate in routes/messages.py changing from `config.DUMP_DIR` to another flag-driven value

## Checked At
2026-09-19T15:10:34Z
