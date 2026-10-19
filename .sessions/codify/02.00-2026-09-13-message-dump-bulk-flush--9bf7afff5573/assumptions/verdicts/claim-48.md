VERDICT: DOES-NOT-HOLD
PROMOTION: ELIGIBLE

## Claim
The numeric `config` globals wired to CLI flags in `buster_ripper/main.py` are `MAX_MODEL_LEN`, `COMPACT_TOKEN_RATIO`, `COMPACT_LATENCY_MS`, `COMPACT_KV_RATIO`, `EVAL_MAX_TOKENS`, `EVAL_THINKING_BUDGET`, `IMAGE_CACHE_MAX_SIZE`, `IMAGE_CACHE_TTL`; the numeric globals without a CLI flag are `KV_POLL_INTERVAL`, `COMPACT_NUDGE_RATIO` and `MIN_COMPLETION_TOKENS`

## What it actually does
The second census is right, the first is short by two: `main.py` wires **ten** numeric `config` globals to CLI flags, not eight. `DUMP_FLUSH_INTERVAL` and `DUMP_FLUSH_BYTES` are also flag-wired — `--dump-flush-interval` (main.py:86–90) and `--dump-flush-bytes` (main.py:91–95), each with `expose_value=False` and a parse-time callback that assigns the config global directly (`_set_dump_flush_interval` main.py:18–21, `_set_dump_flush_bytes` main.py:24–27). They reach no `main()` parameter and no `_apply_cli_config` key, which is the only route by which the claim's enumeration of flag-wired globals was taken — but the flag exists and the assignment happens, so they belong in the wired set. Both are `int` in config.py:36–37, read at runtime from `dump.py:125` and `dump.py:194`.

Verified independently, both directions:
- All eight named in the wired list **are** wired: `MAX_MODEL_LEN` (main.py:47, option :96), `COMPACT_TOKEN_RATIO` (:48, :98), `COMPACT_LATENCY_MS` (:49, :100), `COMPACT_KV_RATIO` (:50, :102), `EVAL_MAX_TOKENS` (:53, :118), `EVAL_THINKING_BUDGET` (:55, :123), `IMAGE_CACHE_MAX_SIZE` (:56, :125), `IMAGE_CACHE_TTL` (:57, :127).
- All three named in the unflagged list **are** unflagged: `KV_POLL_INTERVAL` (config.py:64), `COMPACT_NUDGE_RATIO` (config.py:62), `MIN_COMPLETION_TOKENS` (config.py:124) — no `click.option` names them anywhere in the source root; `main.py` is the only module containing `@click.option`.
- The complete numeric-global set in config.py is 13: the 10 wired above plus the 3 unflagged. No other numeric global exists, so the partition is exhaustive and the two listed sets are its only halves.

Reading note (for P9 audit): the claim is a closed partition — "the ones wired are A…H; the ones without a flag are I, J and K" — and only carries information if each list is the whole half. Under that reading it is plainly false. Under a hyper-charitable "these are examples" reading the two clauses would both be individually true but the contrast would assert nothing. The falsifier required no redefinition of any term: `DUMP_FLUSH_INTERVAL`/`DUMP_FLUSH_BYTES` are numbers, are module-level globals in `config.py`, and are wired to flags declared in `main.py`.

## Source
`buster_ripper/main.py` — lines 18–27 (parse-time callbacks) and 86–95 (`--dump-flush-interval`, `--dump-flush-bytes`) establish the counterexample; full option/`_apply_cli_config` scan lines 36–149. Enumeration of numeric globals corroborated by `buster_ripper/config.py` lines 36–37, 41–42, 54–64, 124. Production source only; no test, fixture, or snapshot consulted.

## Source Version
working tree at `/Users/alexhubitski/Projects/buster-ripper/buster_ripper` (no revision string consulted — Git metadata is out of scope for this role); fingerprints below pin the exact bytes.

## Source Fingerprint
sha256:fd36dc12055f67693d9b852fb15a66aa4a3a7864103ac3de0ee67d9520048d47 (main.py)
sha256:22348c68d21b91a30e75eeea802d4d390b2bcdabbf974b0ddc743c9354635522 (config.py)

## Invalidates On
- Either `main.py` or `config.py` byte-hash changes above.
- A `--dump-flush-interval` / `--dump-flush-bytes` option or its callback is removed, renamed, or stopped from assigning the config global (would move those two globals back into the unflagged half).
- Any new numeric module-level global is added to `config.py`, or an existing numeric global is deleted or retyped (changes either half of the census).
- A numeric `config` global becomes wired via a new `@click.option` or a new `_apply_cli_config` assignment.
- The flag wiring for the dump-flush settings moves out of `main.py` into another module that declares CLI options.

## Checked At
2026-10-05T21:04:40Z
