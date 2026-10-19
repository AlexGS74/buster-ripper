VERDICT: HOLDS
PROMOTION: ELIGIBLE

## Claim
No `README.md` example uses the `DUMP_DIR=""` disable form

## What it actually does
Every `DUMP_DIR` example in `README.md` assigns a real path — `~/mllm/prompt-diffs` — and the one CLI-form example passes `--dump-dir ~/mllm/prompt-diffs`. No example in the file assigns `DUMP_DIR` to an empty string, to nothing, or to any other disable-style value. The file has four `DUMP_DIR` assignments total (lines 84, 145, 192, 194); all four are enable-style.

## Source
`/Users/alexhubitski/Projects/buster-ripper/README.md` — read in full (289 lines, 10287 bytes), then exhaustively enumerated.

All 7 occurrences of `DUMP_DIR` / `--dump-dir` / `dump_dir` in the file:

| Line | Occurrence | Kind |
|---|---|---|
| 84 | `DUMP_DIR=~/mllm/prompt-diffs bash serve_buster_ripper.sh` | example, enable |
| 139 | `## Dump mode (\`--dump-dir\`)` | heading, not an example |
| 145 | `DUMP_DIR=~/mllm/prompt-diffs bash serve_buster_ripper.sh` | example, enable |
| 151 | `**Output layout** under \`DUMP_DIR/<session_id>/\`:` | prose path template |
| 180 | ``DUMP_DIR/<session_id>/<format>.log`` | prose path template |
| 192 | `DUMP_FORMAT=change-focused DUMP_DIR=~/mllm/prompt-diffs bash serve_buster_ripper.sh` | example, enable |
| 194 | `buster-ripper --dump-format self-contained --dump-dir ~/mllm/prompt-diffs` | example, enable |

Negative probes, all empty (exit 1):
- `DUMP_DIR[[:space:]]*=[[:space:]]*(''|"")` → no match (no empty-string assignment, either quote style, with or without spaces around `=`)
- `--dump-dir[[:space:]]*=?[[:space:]]*(''|"")` → no match (no empty-string CLI argument)
- `DUMP_DIR[[:space:]]*=[[:space:]]*(:|0|off|false|null|none)\b` → no match (no alternative disable idiom)

## Source Version
README.md working-tree bytes, fingerprinted below rather than versioned by revision string.

## Source Fingerprint
sha256:fe80cb9acabfe699fac070a69b43ac425436606d627b243825d4a028d75a1c15

## Invalidates On
- Any edit to `README.md` that adds a `DUMP_DIR=""` (or `--dump-dir ""`) example — changes this fingerprint.
- Adding usage examples in sources outside the authorized root (docs/eval.md, serve_buster_ripper.sh comments, `buster-ripper --help` epilog) — the claim as worded is README-scoped, so those would not falsify it, but they would make a broader "no example anywhere" claim falsifiable; re-check the fingerprint above before reusing this verdict for anything wider than README.md.

## Checked At
2026-10-04 (local)
