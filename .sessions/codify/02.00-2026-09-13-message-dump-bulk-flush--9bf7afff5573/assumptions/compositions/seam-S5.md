VERDICT: SAFE
PROMOTION: INELIGIBLE — the supplied member verdict files carry no permanent claim handles (only rows 74 and 40 have registry handles named in the seam ledger: `CL:buster_ripper.main/cli-default-binding-time` @ `fd36dc12…`, `CL:click/option-callback-timing` @ `53a05fc6…`; the other eleven members' handles are absent from the supplied materials), so the required handle-and-fingerprint composite pin cannot be constructed.

## Seam
S5 — trigger parameter values (`DUMP_FLUSH_INTERVAL`, `DUMP_FLUSH_BYTES`) crossing CLI options → config globals → the two trigger sites (`asyncio.sleep` in `poll_flush`; inclusive byte comparison in `_append_record`).

## Composition
Producer → transformations → consumer, on current working-tree bytes:

1. **Import (`buster_ripper/main.py` body executes):** `from . import config` (main.py:8) binds the shared config module object; the module-level decorators on `main` evaluate `default=config.DUMP_FLUSH_INTERVAL` (10) and `default=config.DUMP_FLUSH_BYTES` (4·1024·1024) once, snapshotting them onto the click Option objects (74; main.py:87,92; config.py:36-37).
2. **Parse (click, before the command body):** `--dump-flush-interval` / `--dump-flush-bytes` are `type=click.IntRange(min=1)`, `expose_value=False`, with parse-time callbacks (100; main.py:86-95, callbacks at 18-27). `Parameter.process_value` invokes each callback during `parse_args`, strictly before `Command.invoke` runs `main()` (40; click core.py:2438-2463, 1226-1227, 1268-1269, 1405-1406). The callbacks assign `config.DUMP_FLUSH_INTERVAL = value` / `config.DUMP_FLUSH_BYTES = value` and return the value; with `expose_value=False` the config global is the sole channel — the values never enter `main()`'s signature. The callback runs with the argv value or, when the flag is absent, with the import-bound default, so the globals are (re)assigned in every invocation. `IntRange(min=1)` rejects ≤0 at parse time.
3. **Runtime census (no hidden writer):** a package-wide grep shows the only production writers of these two globals are the config.py literals and the two main.py callbacks; the only readers are the two trigger sites (dump.py:125, dump.py:194). After parse, no writer exists, so the runtime reads below observe the parse-time value for the process lifetime. (`_apply_cli_config`, main.py:36-58, does not touch the flush globals.)
4. **Consumer A — timer:** `on_startup` → `spawn_dump_flush` stores `asyncio.create_task(dump.poll_flush())` at `app["dump_flush_task"]` (96; server.py:37-45,81). `poll_flush` is the hand-built periodic loop asyncio does not provide (43): `while True: flush_all()` inside `try/except Exception` (log-and-continue), then `await asyncio.sleep(config.DUMP_FLUSH_INTERVAL)` — the interval global is re-read on every iteration (76, 101; dump.py:114-125). The first iteration flushes before the first sleep. `flush_all` is synchronous (no await), so the sleep is the loop's only suspension point; cleanup cancels the stored task (96; server.py:87,99), so cancellation lands at that await and the interval value plays no role in it. Per-session flush failures are logged with the batch retained (dump.py:274-281); a retained ≥-threshold batch re-attempts on the next append.
5. **Consumer B — threshold (collapsed formats only):** `dump_turn` dispatches `documented` → `_write_documented` (per-turn files; never touches `pending`/`pending_bytes`) and both collapsed values → `_append_record` (dump.py:91-96). `_append_record` computes `rendered_bytes = len(current_text.encode("utf-8"))` before `state.pending.append(record)` (116; dump.py:191-192), accumulates it into `state.pending_bytes` — rendered-text bytes, not record bytes (117; dump.py:193) — and calls `flush_all()` when `state.pending_bytes >= config.DUMP_FLUSH_BYTES`, an inclusive comparison against the runtime-read global (85, 110; dump.py:194-195). `len()` over the encoded `bytes` counts encoded bytes (112), which for non-ASCII text exceeds the code-point count (111), matching the threshold's documented unit (config.py:34,37). The documented session is double-guarded from the timer path by the empty-pending check and `_flush_state`'s `fmt == "documented"` early return (dump.py:276-277,286-287).
6. **Ownership:** the only flush trigger tasks are the two above; `poll_kv_cache` is the KV metrics poll (server.py:15-34, delegating to `compaction.poll_kv_cache` every the distinct `KV_POLL_INTERVAL` global, config.py:64) and never references the dump/flush path (dimension guard against the numerically coincident default 10).

Cross-trigger interaction: threshold (request path) and timer (background task) converge on the same synchronous `flush_all`; invocations are serialized by the event loop, each drain is one append-write-then-clear block (dump.py:284-294), and a drained session leaves nothing for a later timer pass.

## Dimension Coverage
Re-traced this run (changed members participate): **default binding vs runtime reads** (100/74/40 fresh against `fd36dc12…`; 76/85 runtime reads); **collapsed-path-only scoping**; **timer catch-and-continue**; **cancel at sleep await** (96's server.py basis moved to `a9e49823…`; 76/101's dump.py pins are stale pre-T009 bytes `c8f093c8…` — cited `poll_flush` unit at 114-125 re-verified textually identical in current `07b0b076…` bytes); **`poll_kv_cache` is metrics poll, not flush owner** (re-verified in current server.py); **cross-trigger convergence**.

Carried forward from the prior composition `compositions/seam-S5.md` (2026-09-27, SAFE, INELIGIBLE — its `Invalidates On` fired at T006's `main.py` change), re-confirmed by direct read against unchanged pins (`dump.py` `07b0b076…` byte-identical; doc fingerprints unchanged): **inclusive threshold comparison on the UTF-8 rendered-text byte accumulator** (85/110/116/117, docs 111/112) and **pre-mutation encoded-byte computation** (116).

Delta finding: the prior composition's counterexample-note ("the CLI's integer options do not impose positivity — zero interval / zero threshold admitted") is superseded on current bytes: both options are `click.IntRange(min=1)` (fresh claim-100; main.py:86,91), so ≤0 values are rejected at parse before any callback or trigger site runs. The admitted extreme is now 1 (a 1 s interval; a 1-byte threshold flushing on every collapsed append) — degenerate but well-defined; no misroute.

## Counterexample
None found. Every enumerated dimension was traced through the verified claims on current bytes: the import-bound default's only runtime role is feeding the same parse-time assignment (no divergent namespace — one shared config module object; no third writer anywhere in the package); the two triggers read distinct globals at their own sites; the format union's documented member never reaches the byte trigger and is inert-but-harmless under the timer; both collapsed members take an identical trigger path; the inclusive comparison fires on the pre-mutation-computed encoded byte count; ordinary exceptions cannot kill the timer loop and per-session failures retain the batch for retry; cancellation lands at the loop's only await. The CLI's `IntRange(min=1)` bounds both parameters at parse time, so the prior composition's zero/negative-value concern no longer applies.

## Sources
- `buster_ripper/main.py:8,18-27,36-58,61-95,131-149` (callbacks, decorators, `_apply_cli_config`, `main`)
- `buster_ripper/config.py:13,27-37,64` (flush globals, format set, `KV_POLL_INTERVAL` peer)
- `buster_ripper/dump.py:24,83-133,189-195,274-294` (import, `dump_turn` dispatch, `poll_flush`, `_resolve_format`, `_append_record`, drain)
- `buster_ripper/server.py:15-45,55-107` (kv monitor vs `spawn_dump_flush`, `on_startup`/`on_cleanup`/`_cancel_background_task`)
- `.venv/lib/python3.13/site-packages/click/core.py:1226-1227,1268-1269,1405-1406,2438-2463` (per claim-40)
- Installed CPython 3.14.3 asyncio sources and consulted Python 3.14 Unicode/bytes documentation — locators and fingerprints per claims 43, 111, 112

## Source Versions
Working tree at /Users/alexhubitski/Projects/buster-ripper (git metadata out of scope; no revision strings read): `main.py` `fd36dc12…`, `config.py` `22348c68…`, `dump.py` `07b0b076…`, `server.py` `a9e49823…`. Click 8.3.1 (repo `.venv`, pinned by `uv.lock`), `core.py` `53a05fc6…`. asyncio: installed CPython 3.14.3 (homebrew). Unicode HOWTO: Python 3.14 documentation fetched 2026-09-27. stdtypes: CPython documentation branch 3.14, revision unavailable.

## Source Fingerprint
sha256:6948f13287f9b21aed50bc6454d5c9fbfcb7568183f5781cfc5308d08c199b2b — over the documented preimage: one `member|<row>|<verdict fingerprints…>` line per member in ledger order (100, 74, 40, 43, 76, 85, 116, 117, 111, 112, 96, 101, 110; claim-row ids substituted for the absent permanent handles), then `source|<path>|<sha256>` lines for the five consulted production sources above.

## Invalidates On
- Any change to a listed member's verdict or fingerprint, or re-extraction of any member row (100, 74, 40, 43, 76, 85, 116, 117, 111, 112, 96, 101, 110); publication of permanent handles for the eleven handle-less members (required to clear the promotion ineligibility).
- `main.py` flush-option declarations, callbacks, defaults, or `IntRange` bounds; `_apply_cli_config` gaining flush-global writes; any new writer of `DUMP_FLUSH_INTERVAL`/`DUMP_FLUSH_BYTES` anywhere in `buster_ripper/`.
- `dump.py` `poll_flush` loop/sleep/exception handling, `_append_record` sizing/accumulation/threshold comparison, `dump_turn` format dispatch, or the drain's write-and-clear block.
- `server.py` `spawn_dump_flush`, `on_cleanup`/`_cancel_background_task` ordering, or the kv-monitor wiring (`compaction.poll_kv_cache` acquiring any flush role).
- Click `core.py` (`53a05fc6…`) parameter-processing order; installed asyncio sleep/scheduling semantics (CPython ≠ 3.14.3); consulted Unicode/bytes documentation bytes.

## Checked At
2026-09-28
