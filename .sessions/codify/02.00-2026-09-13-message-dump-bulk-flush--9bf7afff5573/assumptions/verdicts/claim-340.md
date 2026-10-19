VERDICT: HOLDS
PROMOTION: ELIGIBLE

## Claim
Exactly these ten numeric config globals get a CLI flag in `buster_ripper/main.py`: `MAX_MODEL_LEN`, `COMPACT_TOKEN_RATIO`, `COMPACT_LATENCY_MS`, `COMPACT_KV_RATIO`, `EVAL_MAX_TOKENS`, `EVAL_THINKING_BUDGET`, `IMAGE_CACHE_MAX_SIZE`, `IMAGE_CACHE_TTL`, `DUMP_FLUSH_INTERVAL`, `DUMP_FLUSH_BYTES`

## What it actually does
`main.py`'s single `@click.command()` declares 25 options, and exactly ten of them correspond to
numeric-valued `config` globals: `--max-model-len` (L96), `--compact-token-ratio` (L98),
`--compact-latency-ms` (L100), `--compact-kv-ratio` (L102), `--eval-max-tokens` (L118),
`--eval-thinking-budget` (L123), `--image-cache-max-size` (L125), `--image-cache-ttl` (L127) reach
their globals through `_apply_cli_config` (L47–57); `--dump-flush-interval` (L86) and
`--dump-flush-bytes` (L91) are `expose_value=False` and reach theirs through the parse-time
callbacks `_set_dump_flush_interval`/`_set_dump_flush_bytes` (L18–27). No eleventh numeric global is
flagged: `--port` is an `int` option but has no `config` counterpart, the remaining options are
`str`/`Path`/`Choice`/`is_flag` typed, and the three other numeric globals in `config.py`
(`KV_POLL_INTERVAL`, `COMPACT_NUDGE_RATIO`, `MIN_COMPLETION_TOKENS`) get no option.

Term boundary (named, not chosen to make the claim false): "numeric" is read as int/float-valued,
which is how the source itself types them (`: int`, `: float`) and how the unflagged three are typed.
Every `bool` global in `config.py` is annotated `: bool` and its option uses `is_flag=True`, so
counting bools as numeric would be a reading the source contradicts; under the plain reading the
enumeration above is exhaustive in both directions.

Inventory basis: `config.py` was read in full (124 lines) and every module-level uppercase binding
was enumerated — 30 bindings, of which 13 are int/float valued
(L36, L37, L41, L42, L54, L56, L59, L60, L61, L62, L63, L64, L124); ten of those thirteen are
flagged per the option list, so the claim's set is exact.

## Source
/Users/alexhubitski/Projects/buster-ripper/buster_ripper/main.py — L12–59 (`_set_dump_flush_*`,
`_apply_cli_config`), L61–130 (complete `@click.option` inventory), L131–149 (`main()` signature and
`_apply_cli_config` call). Corroborating inventory of numeric globals:
/Users/alexhubitski/Projects/buster-ripper/buster_ripper/config.py (whole file). Production source
only; no test, fixture, or mock consulted, and nothing was executed (P1).

## Source Version
Local source root `/Users/alexhubitski/Projects/buster-ripper/buster_ripper` as present at check
time; version/revision string unavailable (git metadata is out of scope for this role).

## Source Fingerprint
sha256:fd36dc12055f67693d9b852fb15a66aa4a3a7864103ac3de0ee67d9520048d47 (main.py)
sha256:22348c68d21b91a30e75eeea802d4d390b2bcdabbf974b0ddc743c9354635522 (config.py)

## Invalidates On
- main.py bytes change such that a `@click.option` is added for `KV_POLL_INTERVAL`,
  `COMPACT_NUDGE_RATIO`, or `MIN_COMPLETION_TOKENS` (breaks "exactly ten")
- main.py bytes change such that any of the ten listed options is removed, renamed, or no longer
  assigned onto its `config` global
- config.py bytes change such that a new int/float module-level global appears, or such that one of
  the ten listed globals is deleted or retyped (changes the numeric population being quantified over)

## Checked At
2026-10-05T21:36:27Z
