# Iteration 032-01 — Forward — T003: Wire the flush (volume trigger, timer, shutdown)

## Orchestrator Prompt (verbatim)

```
Execute the forward-interactive skill for this session. Arguments: $ARGUMENTS
```

Dispatch payload: `Session ID: 02.00-2026-09-13-message-dump-bulk-flush--9bf7afff5573` / `Task: T003`.

## Increment Brief

- **Task/goal**: T003 — give `flush_all` its three callers: the append-time volume check (G-005), the periodic timer coroutine (G-004), and the shutdown seam in `on_cleanup` (G-002), each scoped to the collapsed path.
- **Anchors touched**: flush timer → `dump.py:poll_flush` (new) → **NEW**; flush trigger parameters → `config.py:DUMP_FLUSH_INTERVAL/DUMP_FLUSH_BYTES` (new) → **NEW**; dump entry point → `dump.py:dump_turn` (+`_append_record`) → **extend**; shutdown flush seam → `server.py:on_cleanup` → **extend**; bulk flush → `dump.py:flush_all` → **reuse unchanged**.
- **Will NOT create**: no new buffer type, no scheduler/registry, no second spawner protocol — `spawn_kv_cache_monitor`'s lifecycle is reused as the pattern; `poll_kv_cache` is NOT reused for the flush (wrong concern, RB-002).
- **Reuse evidence**: both `NEW` anchors cite **RB-002**; replayed this run — the bounded name search (`timer|sched|interval|periodic|tick|deadline|call_later|create_task|asyncio\.sleep` over `buster_ripper/` + `tests/`) still finds only `poll_kv_cache` + `KV_POLL_INTERVAL` + `spawn_kv_cache_monitor` (all wrong concern), `count_tokens.py:19` prose, and the legacy monolith (excluded). No viable owner appeared since RB-002 was authored. Native-capability check unchanged: no asyncio periodic facility; aiohttp `cleanup_ctx` deliberately not adopted (RB-002's recorded decision).
- **Inventions: none** — every artifact is an Anchor-Map row (`poll_flush` realizes *flush timer*; the two globals realize *flush trigger parameters*; the two CLI options are the row's `main.py` wiring; the spawner/`app["dump_flush_task"]` key and teardown order are T003's ADDED items and the `shutdown flush seam` row's extend stance).

## Failure-Path Evidence

| Operation / condition | Expected outcome and authority | Handling / observation evidence | Disposition |
|---|---|---|---|
| Volume check fires and `flush_all` hits a write failure | flush never raises; batch retained, logged — flush contract (`Mode: NEW` bulk-flush row: "must not raise") | `_flush_session` try/except (T002); `test_flush_never_raises_on_write_failure` (T002, still green) | safe |
| Volume check boundary: pending_bytes == threshold | flush fires — G-005 failure witness ("off by one") | `test_boundary_equality_flushes` (>=, not >) | safe |
| Volume check reads wrong field / evaluated pre-append | reads `pending_bytes` the append just wrote, cumulative — G-005 witness | `test_check_reads_accumulated_pending_bytes` | safe |
| Documented path: check must never fire | nothing buffered → no flush, no `.log` — F-15/018 scope qualifier | `test_volume_check_never_fires_on_documented_path` (three documented files still live) | safe |
| Flush raises inside the timer loop | logged, loop continues on later ticks — G-004 failure witness (dead trigger) | `poll_flush` catches `Exception` per iteration; `test_raising_flush_is_logged_and_loop_continues` (first tick raises, later tick flushes, caplog records) | safe |
| Timer cancelled (shutdown) | stops without traceback; cancellation lands at the sleep await — claim-18 done-when | sleep is outside the `except Exception`; `test_cancelled_timer_stops_cleanly` (`task.cancelled()`; `except Exception` cannot swallow `CancelledError`) | safe |
| `DUMP_DIR is None` when the timer ticks / shutdown flush runs | no-op, no raise — bulk-flush row ("no-ops when `DUMP_DIR is None`") | `flush_all` early return; `test_flush_noop_when_dump_dir_none` (T002); `test_handles_missing_dump_flush_task` exercises `_final_dump_flush` with `DUMP_DIR=None` | safe |
| Timer runs on documented path | no-op, nothing pending — F-15 | `test_timer_noop_on_documented_path` | safe |
| `app` missing the flush task key at cleanup | guard skips; flush+close still run — peer of missing kv key | `_cancel_background_task` key guard; `test_handles_missing_dump_flush_task` | safe |
| Shutdown flush raises | cannot raise out of `on_cleanup` (claim 36: a cleanup exception propagates out of `run_app` → traceback exit, violating G-002's clean exit) | `_final_dump_flush` wraps + logs; `test_shutdown_flush_failure_never_raises` | safe |
| Teardown order violated (flush before cancels / close before flush) | cancel both tasks → flush → close session — G-002 coverage row + `E:T003/server/shutdown-flush-seam` gloss | `test_teardown_order_cancel_then_flush_then_close` observes both `cancelled()` states at flush time and the flush count at close time | safe |
| Background task ends with a non-cancel exception before cleanup | unreachable for `poll_flush` (loop catches `Exception`; only `CancelledError` escapes) — peer shape of the kv monitor's existing `await task` | construction argument + resilience test; identical to the pre-existing kv path, not changed by this increment | safe |
| Non-integer CLI trigger value | click rejects at parse time (`type=int`, `CL:click/choice-validation` sibling) | `test_non_integer_value_is_rejected` | safe |
| Mid-run `config.DUMP_FORMAT` mutation between append and flush | flush files under the buffer-time format — `PI:T002/dump_format/format_is_per_artifact` ("chosen once per process") | `state.fmt` captured at append; `test_midrun_format_change_does_not_misfile_records` (backlog absorb) | safe |
| Negative/zero `--dump-flush-interval` | accepted (no validation — `KV_POLL_INTERVAL` peer shape; zero → hot-but-fair loop via `sleep(0)`) | no validation authored (plan specifies none); **concern recorded below**, not a defect | not-applicable (peer-consistent) |

## Capacity gate

Planned production touch set (all additive except the server.py rewrite of `on_cleanup` internals and the backlog-authorized deletion):

| Surface | Kind | Pre-edit | Post-edit |
|---|---|---|---|
| `config.py` | additive (2 globals + comment block) | 111 lines, scan clean | 120 lines, scan clean |
| `main.py` | additive (2 callbacks + 2 options; `main()` body/signature **byte-identical**) | 142 lines; `main` Long Function + Too Many Parameters (pre-existing hard violations, absorbed closure 017) | 164 lines; same 2 findings, same functions — parameter list unchanged (`expose_value=False`), no metric worsened |
| `dump.py` | additive (volume check, `poll_flush`, `fmt` capture) | 284 lines; only pre-existing `_render_body` L:26 N:2 C:9 (backlog, self-deferred) | 304 lines (review band ≤300 exceeded by 4 — see Concerns); **0 function issues**; `dump_turn` L:12 N:1 C:2 P:1, `_append_record` L:4 N:1 C:2 P:3, `poll_flush` L:5 N:1 C:2 P:0 |
| `server.py` | additive + `on_cleanup` internal rewrite + dead-code delete | 112 lines, scan clean | 123 lines, scan clean (`on_cleanup` L:4, `_cancel_background_task` L:7 N:2 C:2 P:2, `_final_dump_flush` L:4 N:1 C:1 P:0, `spawn_dump_flush` L:1 P:1) |

No extraction was triggered by this increment: no touched function crossed a threshold, no touched file entered 501+, and the pre-existing `main.py:main` violation is preserved exactly (ratchet forbids increasing; diff shows only decorator/callback additions outside the function). Full suite 210 green pre/post (single final run at 92.87%).

## Backlog pickup

- **Absorbed** `QB:entry-points/duplicate-main` — server.py was in the touch-set for the teardown rewrite; deleted the dead dev launcher (`server.py:99-112`, `def main` + `__main__` argparse): zero references in README/docs/tests/serve-script; console entry is `main.py:main` (pyproject:11). Evidence: `rg "server\.main|python -m buster_ripper.server"` → no hits outside the deleted block itself.
- **Absorbed** `QB:dump-flush/format-reread` — `_flush_state` re-read `config.DUMP_FORMAT` at flush time; remedy applied (capture on session state): `SessionState.fmt` set at append time in `dump_turn`, `_flush_state` files under `state.fmt`. Regression: `test_midrun_format_change_does_not_misfile_records`. Unreachable via the CLI (parse-time writer), but the volume/shutdown triggers this task wires now route through the flush mid-session, making the capture load-bearing.
- **Not absorbed (bounded out)**: `QB:render-body/gray-compound` (item's own text defers it to a later quality pass; `_render_body` untouched by this diff), `QB:capacity-baseline/function-metrics` (planning-side plan text — not Forward's to rewrite; this report's Capacity-gate table records per-function baselines for every touched function, the pattern the remedy asks for), `QB:test-dump/file-size` (test_dump.py not in this increment's touch-set; new trigger tests went to a new file instead).

## Changes by file

**`buster_ripper/config.py`** — `DUMP_FLUSH_INTERVAL: int = 10`, `DUMP_FLUSH_BYTES: int = 4 * 1024 * 1024` under a "Dump flush settings" banner (FK-A/FK-B Forward discretion: 10 s `KV_POLL_INTERVAL` peer; 4 MiB single-digit, rendered-text basis).

**`buster_ripper/main.py`** — `_set_dump_flush_interval` / `_set_dump_flush_bytes` callbacks (the `_set_dump_format` shape) + `--dump-flush-interval` / `--dump-flush-bytes` options (`type=int`, `expose_value=False`, `show_default=True`, defaults from the config globals). `main()` gains no parameter — the 017 capacity closure, and claim-22's done-when (mechanism, defaults, values unchanged: 10 s / 4 MiB).

**`buster_ripper/dump.py`** —
- `import asyncio`;
- `_append_record` gains the volume check after the append: `if state.pending_bytes >= config.DUMP_FLUSH_BYTES: flush_all()` — G-005's "reads the same field the append wrote", structurally collapsed-path-only (the documented path never calls `_append_record`);
- `poll_flush()` — the periodic coroutine (`while True: flush_all(); await asyncio.sleep(config.DUMP_FLUSH_INTERVAL)` with per-iteration `except Exception: log.exception`), following the `poll_kv_cache` split (domain loop in the domain module; claim-18's done-when: cancellation lands at the sleep await, no timer/sched/tick vocabulary required);
- `SessionState.fmt` + `state.fmt = fmt` in `dump_turn` + `_flush_state` uses `state.fmt` (backlog absorb, above);
- module docstring notes the three triggers.

**`buster_ripper/server.py`** —
- `spawn_dump_flush(app)` → `app["dump_flush_task"] = asyncio.create_task(dump.poll_flush())`, spawned in `on_startup` beside the kv monitor (the `spawn_kv_cache_monitor` lifecycle);
- `on_cleanup` now: `_cancel_background_task(kv)` → `_cancel_background_task(flush)` → `_final_dump_flush()` → `await app["session"].close()` — the plan's exact order (cancel timer → final flush → close session);
- `_cancel_background_task` extracts the pre-existing cancel-and-await-swallow (behavior-identical for the kv path);
- `_final_dump_flush` wraps `dump.flush_all()` so nothing can raise out of cleanup;
- deleted the duplicate dev `main`/`__main__` block (backlog absorb).

**Tests** —
- **NEW** `tests/test_dump_triggers.py` (volume trigger ×4 + boundary/equality/accumulation/documented-no-op, timer tick, timer resilience with a raising first tick, clean cancel, documented-path no-op, mid-run-format-misfile regression): 9 tests.
- `tests/test_server_hooks.py` — `TestSpawnDumpFlush` (2), `TestOnCleanupDumpFlush` (teardown order with at-flush/at-close observers, missing-key, raising shutdown flush): 5 tests.
- `tests/test_main_cli.py` — `TestDumpFlushOptions` (parse-time assignment ×2, defaults restore, non-integer rejection, help listing): 5 tests.

## L→N→C→P compliance

All new/changed functions (executable statements only):

| Function | L | N | C | P | Verdict |
|---|---|---|---|---|---|
| `dump.poll_flush` | 5 | 1 | 2 | 0 | ideal |
| `dump._append_record` | 4 | 1 | 2 | 3 | ideal |
| `dump.dump_turn` (extended) | 12 | 1 | 2 | 1 | ideal |
| `dump._flush_state` (rewritten head) | 8 | 1 | 2 | 1 | ideal |
| `main._set_dump_flush_interval` / `_set_dump_flush_bytes` | 2 | 0 | 1 | 3 | ideal |
| `server.spawn_dump_flush` | 1 | 0 | 1 | 1 | ideal |
| `server.on_cleanup` (rewritten) | 4 | 0 | 1 | 1 | ideal |
| `server._cancel_background_task` | 7 | 2 | 2 | 2 | ideal |
| `server._final_dump_flush` | 4 | 1 | 1 | 0 | ideal |

`codify scan path --git --detailed` per touched file: dump.py **0 function issues** (file-level note: 304 lines, review band), server.py clean, config.py clean, main.py — the two pre-existing `main` findings, unchanged by this diff (verified: no diff line touches the `def main` signature or body).

## Tests and coverage

- Narrow runs during implementation (test_dump_triggers/test_dump_flush/test_dump/test_routes_integration): dump.py **100%** line coverage.
- **Final full suite (canonical `uv run pytest`): 210 passed, total 92.87%** (from 189 at 92.72%); gate 80% met; `dump.py` 100%, `config.py` 95% (misses are the pre-existing `ModelProfile.describe`/profile lines, untouched).
- 19 new tests this increment (9 + 5 + 5). No mocking via `Mock`/`patch` imports — `mocker` fixture throughout.

## Dependency-claim checks

- **claim-18 (timer basis)**: encoded as ruled — new surface, pattern peer not reused, `asyncio.sleep` mechanism (`CL:python/asyncio-periodic-timer` HOLDS). RB-002 replay recorded above. Listed **single** (T003 only) in `assumptions/claims.md`.
- **claim-22 (CLI precedent)**: implemented per the corrected premise — the flags' basis is `US-010`'s harness contract; both corrected surfaces (plan-summary §Anchor Map flush-trigger-parameters row; this report) carry the eight-wired-precedents fact; grep of the falsified form ("no numeric `config` global has a CLI flag") returns zero hits on both surfaces. Claim is **shared** (T002/T003) — no behavioral divergence introduced: T003's options are the exact `_set_dump_format` mechanism T002 landed.
- **claim-35/36 (aiohttp signal shutdown / cleanup propagation)**: ruled HOLD in planning; this increment does not depend on their unverified edges — `_final_dump_flush` and the cancel wraps make nothing able to raise out of `on_cleanup` regardless. No correction to record.
- **claim-40/41 (click callback timing / choice validation)**: followed the proven T002 mechanism; parse-time assignment pinned by `test_*_assigns_global_at_parse_time`.
- No claim the plan ruled on was found false; no shared claim was resolved locally.

## Protected invariants honored

- flush contains no `await` (`flush_all` unchanged, sync); cannot raise out of `on_cleanup` (wrapped); idempotent per pending batch (clears on flush; re-flush no-ops); `turn`/`prev_text` advance at append time only; `on_cleanup` order cancel-timer → flush → close session (tested).
- No invariant record was created, edited, or deleted (triad rule). The registry's deferred T003 entities (`flush timer` → `dump.py` periodic-loop symbol; `flush trigger parameters` → the two config globals) now have landable bindings for **Yield** to register: `poll_flush` and `DUMP_FLUSH_INTERVAL`/`DUMP_FLUSH_BYTES`.
- `PI:T002/dump_format/format_is_per_artifact` strengthened in effect (buffer-time format capture).

## Concerns (for Yield, non-blocking)

1. **`dump.py` at 304 lines** — 4 into the 301-500 review band. The plan's placement decision (timer loop body must live in `dump.py` because `server.py` is coverage-omitted) forces the growth; docstrings were trimmed 3 lines to minimize it. A record/writer module split is quality-pass scope, not T003's.
2. **No CLI validation for `--dump-flush-interval ≤ 0`** — matches the `KV_POLL_INTERVAL` peer (unvalidated); zero/negative yields a hot but yielding loop. Plan specifies no validation; flagging for the record.
3. **`main` L metric discrepancy**: the plan's 016 measurement recorded ≈31 executable statements; `codify scan` reports L:52 both before and after this increment (methodology difference). No worsening either way — the function body is untouched.

## Durable signal

- **Overrides exercised**: none (no override register entry applies to T003).
- **Decision changes/reversals**: none. Implementation followed the plan text exactly (values 10 s / 4 MiB, callback wiring, teardown order, loop shape).
- **Bug-fix rationale that outlives the code**: the backlog-absorbed format capture — a mid-run `DUMP_FORMAT` mutation could previously misfile records under the wrong artifact identity; with mid-session triggers (T003) the flush now runs while a session is live, which made the capture load-bearing rather than theoretical.
- **Invariant-relevant deltas**: none written by Forward; two deferred registry entities became bindable for Yield (above).
- **Repair-loop note**: this checkpoint is durable review evidence; no compaction/squash is requested or implied.

## Demo evidence

`evidence/t003-flush-triggers.txt` — real `--help` (both flags with defaults) + three live scenes through the real module (volume trigger at append time; one `poll_flush` tick with no shutdown; the shutdown seam's `flush_all`). INDEX row appended, Disposition left empty.

DEMO-EVIDENCE: .sessions/codify/02.00-2026-09-13-message-dump-bulk-flush--9bf7afff5573/evidence/INDEX.md — no pause

## Questions asked

None — no implementation choice met the escalation floor; `FK-A`/`FK-B` were settled discretion, all ambiguity was absorbed by the plan text and the two absorbed assumption closures (DC-claim-18-r1, DC-claim-22-r1), both implemented as specified with their Yield replays in mind.

## Status for Yield

T003 implemented in full: all six ADDED items landed, REMOVED: nothing (the server.py deletion is the quality-backlog pickup, not a plan REMOVED). Ready for validation — suggested replay surfaces: VP-T003-interval-flush and VP-T003-timer-resilience have local black-box analogues in `tests/test_dump_triggers.py`; the process-level VP-T003-shutdown-flush belongs to QAP-MS002 (not authored here, per the QA boundary). MS-002's constituent task set is now complete pending Yield.
