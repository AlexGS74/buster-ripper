VERDICT: UNCLEAR
PROMOTION: INELIGIBLE — verdict is UNCLEAR (promotion requires HOLDS or DOES-NOT-HOLD); additionally the by-name half rests on a directory-wide search, which is not one exact fingerprintable locator.

## Claim
No periodic-timer, scheduler or tick symbol exists under `buster_ripper/`

## What it actually does
The claim's verdict flips on the reading of "symbol", and both readings are readings of the claim as written, so per P9 no decisive verdict is possible without picking a definition (acted on P9; checked against P6 — the contradiction is plain under one reading and absent under the other, so the word "symbol" is the finding, not the behaviour):

- **By-name reading (verified TRUE):** a case-insensitive search for `timer|sched|tick` across every file under `buster_ripper/` returns zero matches. No identifier anywhere in the package bears timer, scheduler, or tick vocabulary, in any casing.

- **By-function reading (verified FALSE):** the package contains the complete symbol set of a periodic timer. `config.KV_POLL_INTERVAL: int = 10` (config.py:44) is the period constant; `compaction.poll_kv_cache()` runs `while True:` (compaction.py:34) ending in `await asyncio.sleep(config.KV_POLL_INTERVAL)` (compaction.py:57) — the canonical asyncio periodic-loop idiom; `server.spawn_kv_cache_monitor()` schedules it at startup via `asyncio.create_task(...)` (server.py:33) and stores the handle as `app["kv_monitor_task"]` (server.py:34), called from `on_startup` (server.py:57); `on_cleanup` cancels it (server.py:63–69). The code's own docstrings say "Start periodic KV cache monitoring. Runs for app lifetime." (server.py:15–16) and "Spawn background task for periodic KV cache monitoring." (server.py:25–26).

- A third reading — "no periodic timer exists under `buster_ripper/`" (mechanism, not symbol) — is flatly false under the same evidence: the KV-cache poll loop is a periodic timer running every 10 seconds for the app's lifetime.

Which reading settles it: re-extract the claim to one wording. "No identifier named with timer/scheduler/tick vocabulary exists under `buster_ripper/`" → HOLDS (grep-verifiable, as above). "No periodic background task or timer mechanism exists under `buster_ripper/`" → DOES-NOT-HOLD (`poll_kv_cache` is exactly that).

## Source
buster_ripper/compaction.py lines 28–57 (poll_kv_cache loop) — establishes the by-function half. Corroborating locators: buster_ripper/server.py lines 15–34, 57, 63–69 (spawn/cancel for app lifetime); buster_ripper/config.py line 44 (KV_POLL_INTERVAL). The by-name half is established by `rg -i 'timer|sched|tick'` over the whole `buster_ripper/` tree returning no matches — a directory-wide search, not a single locator.

## Source Version
pyproject.toml `version = "0.1.0"` (local working tree; no installed package metadata consulted)

## Source Fingerprint
sha256:ac9d42143c98694a0b6672dd753afcef537fc5e03ec747b4bb3c1208e23535e7 (buster_ripper/compaction.py, consulted in full)
sha256:e008d3c971db787ebd1eb890544c6d73f0479bb97d0060ee64d4f77cf150b824 (buster_ripper/server.py, consulted in full)
sha256:14ac40180f081d8e7caeb280d9a9ab194cbd06c93e3958ebf9bef362ca7b7aab (buster_ripper/config.py, consulted at line 44 only)
The by-name negative half is a directory-wide search result and is not fingerprintable.

## Invalidates On
- Any file under `buster_ripper/` gains an identifier containing timer, sched, or tick (any casing) — flips the by-name half to false
- `compaction.poll_kv_cache` loses its `while True` + `await asyncio.sleep(config.KV_POLL_INTERVAL)` structure (loop removed, made one-shot, or restructured) — flips the by-function half to true
- `KV_POLL_INTERVAL` removed from config.py or renamed away from interval vocabulary
- `spawn_kv_cache_monitor` no longer called from `server.on_startup`, or the task handle no longer stored in app state and cancelled in `on_cleanup`

## Checked At
2026-09-19
