VERDICT: HOLDS
PROMOTION: ELIGIBLE

## Claim
`buster_ripper`'s `on_cleanup` performs no dump flush during shutdown

## What it actually does
`on_cleanup` (the only one in the package, `buster_ripper/server.py:60-72`) does exactly two things on shutdown: it cancels and awaits the `kv_monitor_task` (swallowing `CancelledError`) and closes the shared `aiohttp.ClientSession`. It makes no call to anything dump-related. The negative holds doubly: there is no flush call anywhere in `on_cleanup`, and no flush function exists in the package to call — `dump.py`'s only writer, `dump_turn`, writes each turn's file synchronously per request (`write_text` at dump.py:82/95, index append at dump.py:104-106) from `routes/messages.py:57`; there is no pending buffer, no `atexit`/signal hook, and no dump interaction inside the cancelled coroutine's path (`compaction.poll_kv_cache` has no `finally` and touches no dump state).

## Source
buster_ripper/server.py, lines 60-72 (function body of `on_cleanup`); corroborated by buster_ripper/dump.py (no flush function), buster_ripper/compaction.py:28-57 (no dump interaction on cancellation), buster_ripper/routes/messages.py:57 (only dump write site, per-request), buster_ripper/main.py:88-89 (only sets `config.DUMP_DIR`)

## Source Version
unavailable (local working tree, 2026-09-19; no revision string — sha256 fingerprints below are the stability anchor)

## Source Fingerprint
sha256:e008d3c971db787ebd1eb890544c6d73f0479bb97d0060ee64d4f77cf150b824 (buster_ripper/server.py — primary)
sha256:243b21434c14e6ea8144329fe1d0ac4e0c499ed9a7e6e09bdcac5e2417a56b1d (buster_ripper/dump.py)
sha256:ac9d42143c98694a0b6672dd753afcef537fc5e03ec747b4bb3c1208e23535e7 (buster_ripper/compaction.py)
sha256:9c234c4f183aa03a80ab82356a6fdfaad0220f3bef69edf3bab5789ef808277d (buster_ripper/routes/messages.py)
sha256:b4794796d28ad3ab9ce5f2e0e03dddcfa882178299c7b969751697993cb76089 (buster_ripper/main.py)

## Invalidates On
- Any edit to `on_cleanup` in buster_ripper/server.py that adds a call to dump/flush logic (server.py fingerprint change)
- Any addition of a flush/pending-buffer function to buster_ripper/dump.py that `on_cleanup` or its callees invoke (dump.py fingerprint change)
- Any `finally`/cancellation-path dump flush added to `compaction.poll_kv_cache` (compaction.py fingerprint change)
- A second `on_cleanup` (or alternate shutdown hook registering dump flushing) appearing in the package

## Checked At
2026-09-19
