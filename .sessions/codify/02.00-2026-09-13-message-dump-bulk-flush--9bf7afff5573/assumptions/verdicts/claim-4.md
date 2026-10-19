VERDICT: HOLDS
PROMOTION: ELIGIBLE

## Claim
`dump_turn` performs two write-bearing file operations per intercepted turn: the turn-file write and the `index.txt` append

## What it actually does
Each `dump_turn` call performs exactly two content-writing file operations: one `Path.write_text` for the turn file (`turn_000.json` on turn 0, `turn_{turn:03d}.diff` on later turns) and one append-mode write of the summary line to `index.txt`. The only other filesystem call is `session_dir.mkdir(parents=True, exist_ok=True)`, which creates a directory (not file content) and fires only when `state is None` — the first intercepted turn of a session — so it is not a per-turn file write.

## Source
`buster_ripper/dump.py`, `dump_turn` — turn-file write at lines 81–82 (turn 0) and 94–95 (turn ≥1); `index.txt` append at lines 104–106; `mkdir` at line 72

## Source Version
unavailable (git metadata excluded per protocol boundary)

## Source Fingerprint
sha256:243b21434c14e6ea8144329fe1d0ac4e0c499ed9a7e6e09bdcac5e2417a56b1d

## Invalidates On
- any edit to `dump_turn` in `buster_ripper/dump.py` that adds, removes, or merges a file write (e.g., buffering turn files and flushing later, or writing the index inside the turn-file branch)
- `index.txt` writing moved out of `dump_turn` into a caller or a separate flush routine

## Checked At
2026-09-19
