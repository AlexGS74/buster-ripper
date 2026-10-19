# Iteration 050-01 — Forward (T004)

## Orchestrator Prompt (verbatim)

> You are a subagent spawned by another session.
> Session ID: 02.00-2026-09-13-message-dump-bulk-flush--9bf7afff5573
> Task ID: T004
>
> Execute the forward-interactive skill for this session. Arguments: $ARGUMENTS

*(No task-specific arguments were expanded; the session files supplied the binding — T004's finalized task entry in [plan-summary.md](../plan-summary.md) §Task List, the `US-016` re-order in §Current Phase, and the two T004 Planned Task Context blocks (claim-93; claim-3 + claim-20) in [handoff-context.md](../handoff-context.md).)*

## Increment Brief

- **Task**: T004 — tests and coverage: the dump cluster's complete hermetic witness set, including the instrumented write-count measurement.
- **Goal**: every T004 ADDED clause holds in `tests/` — write-count per shipped collapsed format with payload bytes, the paired documented-vs-collapsed measurement, file counts, artifact-level readability/turn-0 witnesses, flush isolation, route-level dump-failure isolation, and the coverage floor.
- **Anchors touched**: **none in production** — test-only increment; the Anchor Map's rows are consumed as test targets (`dump entry point`, `bulk flush`, `flush timer`, `dump format selector`, `collapsed dump artifact`, `dump caller` all `reuse`).
- **Will NOT create**: any production symbol, format, flush policy, reader/viewer, or CLI surface; **no tool** (`FK-007` option 1).
- **Reuse evidence**: no `Mode: NEW` anchor created → no `RB-*` replay owed. `Inventions: none` in production. Test-side: two new test modules (`tests/test_dump_records.py`, `tests/test_dump_write_count.py`) — supporting local tests under T004's ADDED clause, with the records split being the quality backlog's own prescribed remedy (QB:test-dump/file-size), not new referents.
- **Pre-flight**: Interview Finalize approval present (historical 2026-09-17, `US-015`); **US-016 (2026-09-28) explicitly authorizes T004 now** ("reorder and finish coding first") — the 035 execution-order gate is waived in order only, MS-002 re-verdict still owed. `qa-plan.md` exists; T004's MS-003 gate is the existing suite + coverage floor (no QA author pass); its VP handle set recorded below.
- **Clustering**: none — T006 is next but is a separate, independent increment (TLS cluster); clustering a full task into T004's checkpoint would blur the demo-able unit and the milestone gates.

## Plan Adherence Record

T004's ADDED clause list, each item mapped to its implementing test (full ownership — items landed by T001–T003's accepted test files are documented as thoroughly as the new ones):

| T004 clause | Where it holds |
|---|---|
| buffer append and byte accumulation | `test_dump.py` `TestCollapsedBuffering` (records per turn, FK-B rendered-text UTF-8 bytes, per-shape, session-local, append-time advance) — landed T002/T009 |
| **exactly one write-bearing op per flush (K ≥ 2) and its payload bytes** | **NEW** `test_dump_write_count.py` `TestOneWritePerFlush::test_flush_of_k_pending_turns_is_one_write` (K=3, per collapsed format, payload byte length pinned, artifact content pinned) |
| collapsed-artifact shape (exactly one file, no documented files) | `test_dump_flush.py` `TestFlushAll::test_flush_writes_one_artifact_named_by_format` + `test_dump.py` `TestCollapsedBuffering::test_documented_files_never_written_on_collapsed_path` — landed T002 |
| filename states the configured value | same `test_flush_writes_one_artifact_named_by_format` (`[f"{fmt}.log"]` exact listing) |
| arrival order | `test_dump_flush.py` `test_artifact_carries_records_in_arrival_order` (artifact == join of pending; turn 000 < 001 < 002) |
| change readability **per collapsed format** | **NEW** `test_dump_records.py` `TestArtifactReadability` — change-focused: turn 2's diff record itself shows `+REWRITTEN-MARKER` / `-stable two`; self-contained: reader compares turn 2's full record against turn 1's (F-11's two readings, read from the flushed artifact, not the buffer) |
| turn-0 availability **per collapsed format** | **NEW** `test_dump_records.py` `TestArtifactTurnZero::test_body_and_rendering_recoverable_from_artifact` — `json.loads` of the extracted `--- turn 000 body ---` block round-trips the original body; the rendering block is present in full (F-6's both halves) |
| session isolation | `test_dump.py` `test_sessions_buffer_independently` (buffer) + **NEW** `test_dump_flush.py` `TestSessionIsolationAtFlush::test_two_sessions_flush_independently` (two dirs, two artifacts, each carrying exactly its own records, both buffers cleared — VP-T001-session-isolation's two-dirs-and-contents form) |
| dump/flush failure isolation | flush side: `test_dump_flush.py` `test_flush_never_raises_on_write_failure` (landed T002); **NEW route side**: `test_routes_integration.py` `TestDumpFailureIsolated::test_raising_dump_still_yields_upstream_response` — a raising `dump_turn` on `/v1/messages` still yields the upstream 200 |
| the volume trigger | `test_dump_triggers.py` `TestVolumeTrigger` (threshold, cumulative read, inclusive boundary, multibyte UTF-8 both sides, documented-path no-op) — landed T003/T009 |
| the timer loop and its resilience | `test_dump_triggers.py` `TestPollFlush` (tick flush, raise-logged-continue, clean cancellation, documented no-op) — landed T003 |
| `on_cleanup` | `test_server_hooks.py` `TestOnCleanupDumpFlush` (cancel → flush → close order, failure never raises, both task positions, runner-count path, outer-cancellation propagation) — landed T003/T010/T010.001 |
| **write-count test run per shipped collapsed format** | **NEW** — every count-bearing test in `test_dump_write_count.py` is parametrized over `["change-focused", "self-contained"]`, giving `PI:T002/dump_format/format_does_not_change_write_count` its only gate |
| unknown-format and `DUMP_DIR is None` guards | `test_dump.py` `TestFormatDispatch` (hard error, nothing on disk; real `ValueError`, not an assert) — landed T002 |
| real-signal shutdown scenario | correctly **not authored here** — `TC-110`'s, under `tests/qa/` at MS-002 |
| **documented-format preservation test** (018) | `test_dump.py` `TestDocumentedFormatPreserved` (names+content byte-exact for `None`/`"documented"`, index line formats regex-pinned) — landed T002 |
| **file-count test** (018, VP-T002-file-count-reduced) | **NEW** `test_dump_write_count.py` `TestFileCount::test_one_artifact_vs_n_plus_one_files` — N=4: documented dir holds `N + 1` = 5 files, collapsed dir holds exactly `[<fmt>.log]` |
| **VP-T002-default-encoding** (finalized 019 clause) | `test_dump.py` `TestFormatDispatch::test_default_format_is_documented` (no format configured → the three files, no `.log`) plus the new count baseline `TestDocumentedCount` (2N). The R2 candidate clause was **not** authored |
| coverage of every new `dump.py` statement, `U ≤ 3.2 + 0.2·K` | `dump.py` 176 stmts / **0 miss = 100%**; total **93.87%** ≥ 80% (`pyproject.toml` gate). `U = 0` satisfies the bound for every `K` |
| REMOVED: nothing | no test deleted; the record-shape classes **moved** verbatim to `test_dump_records.py` (backlog-prescribed split, content identical) |

**The paired measurement (G-008(b))** — `TestPairedMeasurement::test_collapsed_f_beats_documented_2n_at_same_n`: N=4 turns, documented run writes `2N = 8`, collapsed run (each format) writes `F = 1`, reduction `1 − F/(2N) = 1 − 1/8` — the in-suite measurement the MS-002 Yield Phase's comparison record consumes. `TestFlushCountIsF` drives the injected-`F` replay (F=4=N, the floor case) and proves witness B stays excluded: each of the F appends carries only its own batch (`[e.byte_len] == [batch_bytes(b)]`), and the bytes written across the session equal the on-disk artifact size — growth tracks `F`, never `N²`.

## Assumption Findings (Planned Task Context) — implemented, NOT resolved

- **claim-3** (`DC-claim-3-r1`, `absorbed`, scoped:S1, shared T001–T004): the write-count instrument binds the settled unit — implemented as `_WriteHook` (test-local): one hook on `Path.open` for append/write modes filtering `DUMP_DIR`-relative paths, recording each returned file's `write` as one event with UTF-8 payload bytes. `write_text` funnels through `Path.open` in this CPython (see dependency check), so one hook covers both write shapes with no double counting. `TestWriteUnit::test_mkdir_and_outside_paths_are_not_counted` proves the exclusions observably: `Path.mkdir` is spied (called ≥ 1 for the session dir) yet never appears in the count, and every event is a file under the dump root; the SQLite path (`sqlite3.connect`) never routes through `Path.open` and is structurally outside. Done-when met; **Yield owes the independent replay** (collapsed with injected F; documented at same N → 2N).
- **claim-20** (`DC-claim-20-r1`, `absorbed`, scoped:S1, shared T002/T005): the documented-file reads in `TestDocumentedFormatPreserved` are retained byte-for-byte (the moved split preserved them — `test_dump.py` keeps the class; the three artifact types are read and content-pinned). No production reader or viewer introduced (test-only diff; `grep -rn "index.txt|turn_000" buster_ripper/` → writers only). **Not marked resolved** — S1 recomposition is Forward Collect/Yield's.
- **claim-93** (`DC-claim-93-r1`, `absorbed`, scoped:S1/S2, shared T001–T003/T009/T010): the corrected conditional premise is what the new tests encode — `TestOneWritePerFlush` and `TestPairedMeasurement` use **separate fixed-configuration runs** (one sid's turns all under one format; the format never changes between a run's turns), and the documented no-buffer clause has a dedicated coherent witness: **NEW** `test_dump.py` `TestDocumentedRunWitness::test_immediate_files_empty_pending_no_log` (each turn's file exists immediately after the turn, pending stays empty throughout, no `.log` ever). No runtime switching, migration, or new flush policy implemented. **Not marked resolved** — the S1/S2 recomposition with corrected atomic premises remains Forward Collect's.

No claim status was edited; no verdict file touched.

## Dependency-Claim Check (recorded per skill contract)

- **`Path.write_text` funnels through `Path.open`** — verified by reading installed CPython 3.13.8: `pathlib/__init__.py` `Path.write_text` delegates to `PathBase.write_text`; `pathlib/_abc.py:644-652` shows `with self.open(mode='w', ...) as f: return f.write(data)`. The plan never ruled on the hook mechanism (implied-by-task claim), so this was implemented per the verified fact and recorded here. **Check that changed something: none** — the first design (hooking both `write_text` and `open`) would have double-counted; the verified single-hook design is what shipped.
- Claims' reach read in `assumptions/claims.md`: claim-3 (T001–T004), claim-20 (T002/T005), claim-93 (T001–T003/T009/T010) are all **shared** — but every one is already routed `absorbed` with a recorded deterministic closure (`DC-claim-3-r1`, `DC-claim-20-r1`, `DC-claim-93-r1`) whose required behavior is what this increment implements verbatim; no local resolution decision was made and none is owed here (the co-listed tasks are all accepted; recomposition belongs to Forward Collect at the composition boundary).

## Verification

**Full suite (checkpoint gate)** — `uv run pytest`: **268 passed** (was 250; +18 new tests), 22 pre-existing warnings, coverage **93.87%** ≥ 80%; `dump.py` 176 stmts / 0 miss (**100%**, unchanged). Narrow scope re-ran the five touched files (112 passed) before the full run.

**New tests (18)**: `test_dump_write_count.py` (10: unit-binding 1, one-write-per-flush 2, F-count 2, documented-2N 1, paired measurement 2, file-count 2), `test_dump_records.py` (+5 over the moved 8: readability 2 reading-specific + turn-0 availability 2 + the moved state-defaults 1 re-homed), `test_dump_flush.py` (+2 isolation), `test_dump.py` (+1 documented-run witness; net −72 lines via the split), `test_routes_integration.py` (+1 route-level isolation).

**LNCP scan gate (`codify scan path <file> --git --detailed`, one file per invocation)**:
- `test_dump.py`: file-level review-band note only (341 lines, down from 413), **0 function issues**.
- `test_dump_records.py`: clean ("No refactoring issues found").
- `test_dump_write_count.py`: 7 functions flagged **Medium Parameters only** — each has L/N/C at ideal and exactly one gray metric (P:3–4: the parametrization axes `(dump_env, mocker, fmt)` per the accepted backlog precedent, plus the instrument's intrinsic `recording_open(path, mode, *args, **kwargs)` and `_RecordingFile.__init__`). Single-gray per function → acceptable per the gate.
- `test_dump_flush.py`: `test_two_sessions_flush_independently` Medium **Length only** (L:17 ≤ 30, N:1, C:2 ideal) — single gray, acceptable.
- `test_routes_integration.py`: the new route test was initially **two-gray (L:24 + P)** — refactored before checkpoint (extracted `_MESSAGES_BODY` constant, `mocker.patch.object(config, "DUMP_DIR", tmp_path)` auto-restore replacing try/finally) → now L:11 N:0 C:1 with Parameters-only gray. File-level: 461 lines, review band (pre-existing QB item; see Backlog pickup).

**Evidence** — [evidence/t004-write-count-pair.txt](../evidence/t004-write-count-pair.txt): verbose run of the 10 write-count tests, all PASSED, enumerated per collapsed format (the coverage FAIL line inside that capture is the module-scoped run's artifact — the full-suite gate passed at 93.87%). INDEX.md row appended, Disposition left empty.

**MS-003 readiness (report-only, not executed)**: its gate is the existing suite green + coverage floor — both now hold; the milestone verdict belongs to its Yield Phase.

## Failure-Path Evidence

| Operation / condition | Expected outcome and authority | Handling / observation evidence | Disposition |
|---|---|---|---|
| `dump_turn` with `DUMP_DIR is None` | hard `ValueError`, no writes (F-5 guard) | `TestFormatDispatch::test_dump_dir_none_raises_value_error` | safe |
| unknown `config.DUMP_FORMAT` | hard error, nothing on disk (`PI:format_set_is_closed`) | `test_unknown_format_raises_and_writes_nothing` | safe |
| flush write failure (`OSError`) | never raises out of `flush_all`; batch retained; logged | `test_flush_never_raises_on_write_failure` | safe |
| symlinked/non-regular artifact or turn-file target | refused before write; target untouched; batch retained (R2) | `TestDurableWriterGuards` (3 tests) | safe |
| unencodable rendering (lone surrogate) | `UnicodeEncodeError` before any partial append; next turn healthy | `test_unencodable_rendering_does_not_poison_next_turn` | safe |
| timer-loop flush raises | logged per-iteration; loop keeps flushing (G-004 witness) | `test_raising_flush_is_logged_and_loop_continues` | safe |
| timer cancellation | stops cancelled, no traceback | `test_cancelled_timer_stops_cleanly` | safe |
| route-level `dump_turn` raises | request still succeeds with upstream body (`EI:.../dump_failure_isolated`) | **NEW** `test_raising_dump_still_yields_upstream_response` (200 asserted) | safe |
| flush with nothing pending / `DUMP_DIR is None` | no-op, no artifact | `test_flush_noop_*` (2 tests) | safe |
| documented state with pending batch | flush skips (no `.log`); batch retained | `test_flush_skips_batch_when_format_is_documented` | safe |
| mid-run format mutation (out-of-contract embedder path) | records filed under buffer-time format | `test_midrun_format_change_does_not_misfile_records` (QB regression) | safe |
| recorder sees a write outside `DUMP_DIR` | not counted (settled unit is `DUMP_DIR`-scoped) | `test_mkdir_and_outside_paths_are_not_counted` (all events root-relative files) | safe |
| Excluded: SIGKILL/hard-shutdown durability | waived by `US-001` (non-goal) | — | not-applicable |
| Excluded: SQLite stats-DB writes | outside the measured surface (G-001 scope; `sqlite3.connect` never routes through `Path.open`) | structural + doc'd in module docstring | not-applicable |

## Capacity gate

`Capacity gate: not applicable - no additive production surface` — the production touch set is empty (test-only increment; `buster_ripper/` unchanged, `git diff --stat` confined to `tests/` and session files).

## Backlog pickup

`Backlog pickup: QB:test-dump/file-size ABSORBED (the item's own remedy executed)` — the record-shape classes moved to `tests/test_dump_records.py` as the item prescribed for exactly this moment ("split … when T004 grows the file further"); `tests/test_dump.py` went 413 → 341 lines with the T004 additions landed, and the new `test_dump_records.py` (253) and `test_dump_write_count.py` (275) are both Green (<300). The item is marked `[x]` with evidence below. **Deferred with reason**: `QB:test-routes/file-size` — the file is touched (+36 lines → 461, review band <500, ratchet not crossed), but the remedy ("consider splitting fixtures or behavior groups") is an open-ended reorganization of an accepted 433-line suite; absorbing it here would blur T004's reviewable unit. Noted next to the item; left for the quality pass.

## Concerns

- **None blocking.** One observation for Yield: `tests/test_dump.py` remains in the FL-LNCP review band at 341 lines even after the prescribed split (the buffering/preservation classes are T004 clause owners and could not move without splitting clause witnesses across more files than the backlog remedy prescribed). Recorded rather than re-split — the backlog item's criterion was the prescribed remedy, which is executed; further splits are a quality-pass decision.
- The write-count instrument is deliberately **module-local** rather than a `conftest.py` fixture: `conftest.py` documents "no shared fixtures needed — tests use mocker directly", and only `test_dump_write_count.py` consumes the hook. If a later consumer appears (e.g. a QA gate wanting the same unit), promoting it to a shared fixture is the natural move — flagged here so Yield sees the choice.

## Durable Signal

- **Overrides exercised**: none — no invariant, override register entry, or protected surface touched (test-only diff; the triad's invariant side untouched).
- **Decision changes/reversals**: none in plan decisions. One implementation-shape choice recorded above (single `Path.open` hook vs. hooking both write shapes — verified against stdlib, first design would have double-counted).
- **Bug-fix rationales that still matter**: the initial route-isolation test carried two gray LNCP metrics and was refactored before checkpoint (constant extraction + auto-restore patch) — the shape is the template for future route tests using `_MESSAGES_BODY`.
- **Invariant entries changed by this increment**: none (write ban respected; no `.invariants/` or session invariant files edited).
- **Repair-loop separation**: this is a first-pass increment, not a repair iteration; the checkpoint below is the review evidence for the whole T004 unit.

DEMO-EVIDENCE: .sessions/codify/02.00-2026-09-13-message-dump-bulk-flush--9bf7afff5573/evidence/INDEX.md — no pause
