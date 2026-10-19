VERDICT: DOES-NOT-HOLD
PROMOTION: ELIGIBLE

## Claim
No numeric `config` global has a CLI flag today (`KV_POLL_INTERVAL` included)

## What it actually does
Eight numeric globals in `buster_ripper/config.py` have CLI flags wired in `buster_ripper/main.py`, each assigned straight onto `config.<NAME>` in `main()`: `MAX_MODEL_LEN` ← `--max-model-len` (config.py:39 → main.py:27,75), `COMPACT_TOKEN_RATIO` ← `--compact-token-ratio` (:40 → :29,76), `COMPACT_LATENCY_MS` ← `--compact-latency-ms` (:41 → :31,77), `COMPACT_KV_RATIO` ← `--compact-kv-ratio` (:43 → :33,78), `EVAL_MAX_TOKENS` ← `--eval-max-tokens` (:34 → :49,81), `EVAL_THINKING_BUDGET` ← `--eval-thinking-budget` (:36 → :54,83), `IMAGE_CACHE_MAX_SIZE` ← `--image-cache-max-size` (:21 → :56,84), `IMAGE_CACHE_TTL` ← `--image-cache-ttl` (:22 → :58,85). The claim's parenthetical is the part that is true: `KV_POLL_INTERVAL` (config.py:44, int) has no CLI flag — its only assignment in the package is its config.py default, and `compaction.py:57` reads it directly. The numeric config globals without a flag are `KV_POLL_INTERVAL`, `COMPACT_NUDGE_RATIO` (config.py:42), and `MIN_COMPLETION_TOKENS` (config.py:103); the rest are flagged, so the universal negative fails. The verdict is reading-independent: it fails both if "numeric config global" means any numeric module-level global in config.py and if it means only the mutable values `main()` assigns (per the module docstring), so no definitional choice was made.

## Source
buster_ripper/main.py, click option decorators (lines 25–61) and their `config.*` assignments in `main()` (lines 75–85); definitions in buster_ripper/config.py lines 21–44. `server.py` lines 106–110 hold a second argparse parser but define only `--host`/`--port`, touching no config global.

## Source Version
Repository working tree (uncommitted checkout; no version string available)

## Source Fingerprint
buster_ripper/main.py: sha256:b4794796d28ad3ab9ce5f2e0e03dddcfa882178299c7b969751697993cb76089
buster_ripper/config.py: sha256:14ac40180f081d8e7caeb280d9a9ab194cbd06c93e3958ebf9bef362ca7b7aab

## Invalidates On
- Any click option added to or removed from main.py that wires a numeric config global
- Any numeric global added, removed, or retyped in config.py's setting sections
- CLI wiring moved out of main.py (e.g. into server.py or a new entrypoint module)

## Checked At
2026-09-19T15:10:34Z
