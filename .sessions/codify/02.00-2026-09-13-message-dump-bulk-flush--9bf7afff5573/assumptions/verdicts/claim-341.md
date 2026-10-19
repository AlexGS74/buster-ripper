VERDICT: HOLDS
PROMOTION: ELIGIBLE

## Claim
The numeric `config` globals with no CLI flag are only `KV_POLL_INTERVAL`, `COMPACT_NUDGE_RATIO` and `MIN_COMPLETION_TOKENS`

## What it actually does
`config.py` defines 13 int/float-valued module globals — `DUMP_FLUSH_INTERVAL` (L36),
`DUMP_FLUSH_BYTES` (L37), `IMAGE_CACHE_MAX_SIZE` (L41), `IMAGE_CACHE_TTL` (L42),
`EVAL_MAX_TOKENS` (L54), `EVAL_THINKING_BUDGET` (L56), `MAX_MODEL_LEN` (L59),
`COMPACT_TOKEN_RATIO` (L60), `COMPACT_LATENCY_MS` (L61), `COMPACT_NUDGE_RATIO` (L62),
`COMPACT_KV_RATIO` (L63), `KV_POLL_INTERVAL` (L64), `MIN_COMPLETION_TOKENS` (L124).
Checking each against the package's only CLI surface independently of claim 340: `main.py` is the
sole command declaration in the source root (grep for `@click.option|@click.command|@click.group|
@click.argument|argparse|add_argument|sys.argv` across `buster_ripper/` returns hits only in
`main.py`), and its 25 options cover ten of the thirteen numeric globals. The remaining three are
unflagged: `KV_POLL_INTERVAL` and `MIN_COMPLETION_TOKENS` do not appear anywhere in `main.py`, and
`COMPACT_NUDGE_RATIO` appears only as a read-only value in the startup log call
(`config.COMPACT_NUDGE_RATIO`, L183) with no option declaring it — the three other unflagged
non-numeric globals (`SKIP_TLS_VERIFY` etc.) are all flagged or non-numeric, so the complement set
is exactly the three named.

Term boundary (named, not chosen to make the claim false): "numeric" = int/float-valued, matching
the source's own `: int`/`: float` annotations; `bool` globals are separately annotated `: bool` and
declared with `is_flag=True`. All bool globals happen to be flagged, so this reading does not affect
the complement set.

## Source
/Users/alexhubitski/Projects/buster-ripper/buster_ripper/config.py — whole file read; numeric
bindings at L36, L37, L41, L42, L54, L56, L59–64, L124.
/Users/alexhubitski/Projects/buster-ripper/buster_ripper/main.py — complete `@click.option` inventory
L61–130, log-only use of `COMPACT_NUDGE_RATIO` at L179–184. Production source only; no test or
fixture consulted, nothing executed (P1).

## Source Version
Local source root `/Users/alexhubitski/Projects/buster-ripper/buster_ripper` as present at check
time; version/revision string unavailable (git metadata is out of scope for this role).

## Source Fingerprint
sha256:22348c68d21b91a30e75eeea802d4d390b2bcdabbf974b0ddc743c9354635522 (config.py)
sha256:fd36dc12055f67693d9b852fb15a66aa4a3a7864103ac3de0ee67d9520048d47 (main.py)

## Invalidates On
- main.py gains a `@click.option` that sets `KV_POLL_INTERVAL`, `COMPACT_NUDGE_RATIO`, or
  `MIN_COMPLETION_TOKENS` (the "only" becomes stale)
- main.py loses or stops assigning one of the ten numeric options, promoting a fourth name into the
  unflagged set
- config.py adds an int/float module global with no option, or deletes/retypes one of the three
  named globals
- a second CLI entrypoint (argparse or another click command) is introduced in the package, making
  `main.py` an incomplete flag inventory

## Checked At
2026-10-05T21:36:27Z
