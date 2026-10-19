VERDICT: HOLDS
PROMOTION: ELIGIBLE

## Claim
No symbol in the `buster_ripper` package selects a dump record format — no config global and no CLI flag carries that role

## What it actually does
The dump record format is hardcoded inline in `buster_ripper/dump.py:dump_turn`: turn 0 writes `turn_000.json` via `json.dumps(body, indent=2, ensure_ascii=False)`; every later turn writes `turn_{N:03d}.diff` via `difflib.unified_diff`; `index.txt` accumulates a summary line. No symbol selects between these — the switch is the internal `turn == 0` condition on per-session state. The only dump-related config global is `config.DUMP_DIR: Optional[Path]` (a root path / mode switch: whether and where to dump), and the only dump-related CLI flag is `--dump-dir PATH` (the same role). No `DUMP_FORMAT`-style global, no `--dump-format`-style option, no env-var read (`os.environ`/`getenv` absent from the package), and no format/encoding parameter or constant exists anywhere in the package.

## Source
buster_ripper/dump.py:61-95 (dump_turn's hardcoded format branch); corroborated across the full package surface: buster_ripper/config.py:16 (DUMP_DIR is the sole dump global among all config.py symbols), buster_ripper/main.py:25 (the sole dump CLI option among all 21 click options), buster_ripper/routes/messages.py:55 (guards on `config.DUMP_DIR is not None` — selects whether, not format). Exhaustive package greps for `dump`, `format|FORMAT`, `record|RECORD`, `environ|getenv`, and `DUMP_|dump_format|encoding` found no other carrier: the `format` hits are the logging format string (main.py:102) and the word "transformations"; `encoding` hits are HTTP `transfer-encoding` header names and `encoding=` kwargs of `write_text`/`open`. Note: `.agents/skills/qa-project/SKILL.md` and the `invariants/` registry reference a planned `--dump-format <encoding>` flag and `<format>.log` artifacts, but those are planning/registry documents, not package source — and `invariants/index.md` itself states the "authorized symbols do not exist yet" (planning-time elevation; Forward has not run). Docs cannot establish behaviour; the production source is the ground.

## Source Version
Working tree at /Users/alexhubitski/Projects/buster-ripper (unversioned checkout; no version string consulted)

## Source Fingerprint
sha256:243b21434c14e6ea8144329fe1d0ac4e0c499ed9a7e6e09bdcac5e2417a56b1d  buster_ripper/dump.py
sha256:14ac40180f081d8e7caeb280d9a9ab194cbd06c93e3958ebf9bef362ca7b7aab  buster_ripper/config.py
sha256:b4794796d28ad3ab9ce5f2e0e03dddcfa882178299c7b969751697993cb76089  buster_ripper/main.py

## Invalidates On
- Any `DUMP_FORMAT`-style global or dump-format constant/enum appearing in `buster_ripper/config.py` or any other package module
- Any `--dump-format`-style click option appearing in `buster_ripper/main.py`
- `dump_turn` gaining a format/encoding parameter, or the turn-0/turn-N branches being replaced by a symbol-selected writer

## Checked At
2026-09-19
