# 031 — forward, cycle 02 (T002: format branch, collapsed artifact, single-write flush)

## Orchestrator Prompt (verbatim)

> Session ID: 02.00-2026-09-13-message-dump-bulk-flush--9bf7afff5573
> Task: T002

## Increment Brief

- **Task/goal**: T002 — make the dump format-selectable: `dump_turn` branches on `config.DUMP_FORMAT`
  (documented default / change-focused / self-contained), collapsed formats buffer per-turn records and
  `flush_all()` appends each session's batch to `DUMP_DIR/<sid>/<format>.log` in ONE write; the plan-mandated
  capacity extraction runs BEFORE any additive behavior.
- **Anchors touched**: dump entry point → `dump.py:dump_turn` → extend (owns the format branch per the row);
  per-session dump state → `SessionState` → extend (append conditioned, F-21); bulk flush → `flush_all` → **NEW**
  (`RB-003`); collapsed dump artifact → filename constant + path builder + append target → **NEW** (`RB-003`);
  dump format selector → `config.DUMP_FORMAT`/`DUMP_FORMATS` + `--dump-format` → **NEW** (`RB-004`);
  dump summary line → relocated construction → extend (encoding-independent, both forms).
- **Will NOT create**: no flush timer / trigger parameters (`T003`), no volume check (`T003`), no `on_cleanup`
  wiring (`T003` — `flush_all` intentionally has **no caller** this increment), no serve-script env var (`T005`),
  no README section (`T005`), no read-side tool (FK-007 option 1), no new buffer type.
- **Reuse evidence (replayed pre-write)**: `RB-003` re-run — name/role searches over `buster_ripper/`+`tests/`
  still find no pre-existing flush/artifact/append owner (the only new symbols are this session's own landed ones);
  package-wide write sites remain dump.py-only (3 documented + 1 new append). `RB-004` re-run — no format-selector
  symbol beyond the landed `DUMP_FORMAT`/`--dump-format`. Native check: `Path.open("a")` is the reused append
  technique (dump.py's own); `click.Choice` native, already used (`--eval-profile` shape). No viable owner found → NEW stands.
- **Inventions: none** — every new symbol is a row-listed `Mode: NEW` anchor or a module-private extraction helper
  ("Private helpers are not rows", plan §Anchor Map).

## Pre-flight verification

- **Plan approval gate**: PASSED — round 18 final approval recorded (handoff `## Session Metadata`); T001 already
  accepted through Yield; this dispatch is the T002 forward.
- **QA contract consistency**: `qa-plan.md` exists. T002's VP handles: `VP-T002-documented-format-preserved`,
  `VP-T002-default-encoding` (both T002→MS-001 via QAP-MS001) and `VP-T001-writes-deferred` (corrected to T002,
  MS-001). Authored as **local supporting tests only**; no QA gate test written, no `tests/qa/` created
  (forward-qa owns QAP-MS001 after all MS-001 tasks pass local Yield).
- **Task-specific planned context**: the `claim-19` block applies (see Assumption finding below).
- **Invariant registry (narrowed: `invariants/buster_ripper/dump/invariants.md`)**: the three dump `OI:` blocks are
  `Decision: preserve` — implemented toward them (documented writer byte-preserved; chain advances at append time;
  turn-0 full body in both encodings). The six `PI:T002/dump_format/*` are `Status: planned` **for this session** —
  implemented toward them, none contradicted. `E:T002/dump/dump-entry-point`'s gloss matches the landed branch.
  **No protected invariant conflict — no stop.** Registry untouched (triad rule; the deferred-entity table's
  T002 rows are Yield's to register).
- **Working tree**: clean at `93d853b` (Checkpoint 63) before the first edit.

## Implementation

`dump.py` 115 → 284 lines (Green, < 300 as the plan requires). **Extraction FIRST** (the Yield guard's condition),
then the format machinery — one commit, but authored in that order and verified mid-flight.

1. **Capacity extraction** (the six plan-named concepts → module-private helpers): `_lookup_or_create_state`
   (state lookup/creation), `_write_documented`/`_write_full_turn`/`_write_diff_turn`/`_append_index_line`
   (the documented writer — today's three write sites + index, relocated), `_change_focused_record` /
   `_self_contained_record` / `_turn_zero_block` (the two encodings), `_full_summary` / `_diff_summary` /
   `_msg_tool_counts` (the summary line — encoding-independent, both forms), `_artifact_path` + `_ARTIFACT_SUFFIX`
   (artifact path builder). `dump_turn` itself now owns the format branch (the Anchor row's exact wording) at
   **L:11 N:1 C:2 P:1** — the hard-L violation (L:44) is cured.
2. **Format selector**: `config.DUMP_FORMATS = ("documented", "change-focused", "self-contained")`,
   `config.DUMP_FORMAT = "documented"` (closed set spanning the layout axis — the settled `F-24` reading; the R2
   candidate set NOT authored). `--dump-format` follows the `--eval-profile` shape with
   `type=click.Choice(list(config.DUMP_FORMATS))`, `default="documented"`, `show_default=True`, and assigns
   `config.DUMP_FORMAT` **through the option's own callback** with `expose_value=False` — `main()` gains no
   parameter and no body statement (P stays 21, L unchanged — the recorded main.py ratchet holds).
3. **Dispatch**: `_resolve_format()` raises `ValueError` on any value outside `DUMP_FORMATS`
   (`PI:format_set_is_closed`); it runs **before** state creation, so an unknown value has zero fs effect
   (request isolation stays with the caller's swallow-all). The `documented`/`change-focused`/`self-contained`
   branch selects exactly one writer per turn (`PI:one_write_path_per_turn`).
4. **Records** (byte-for-byte content pieces preserved from `dump.py:77-102` as-was): change-focused — turn 0
   carries `json.dumps(body, indent=2, ensure_ascii=False)` **and** the `_render_body` rendering behind
   `--- turn 000 body ---`/`--- turn 000 rendered ---` headers; turn ≥ 1 carries the `difflib.unified_diff` text.
   self-contained — every turn carries its complete rendering (turn 0 additionally the JSON body, `F-6`);
   its turn-≥1 summary computes `+added/-removed` from a **discarded** diff (counts in both formats, text stored
   only in change-focused — the plan's exact clause). Summary lines are identical constructions to the documented
   writer's (`FULL` / `+a/-r lines` forms).
5. **F-21 conditioning**: the append block (`pending.append(record)`, `pending_bytes += len(current_text)`) runs
   **only on the collapsed branch**; the documented path writes at append time exactly as today and buffers
   nothing (an unconditional append + flush would yield `2N + F`). `turn`/`prev_text` advance unconditionally at
   append time (`OI:sequential_diff_chain`). `pending_bytes` stays the FK-B measure (`len(current_text)`) — it now
   diverges from record bytes by design (tested as a property).
6. **Artifact + flush**: `_artifact_path` builds `DUMP_DIR/<sid>/<format>.log` (append-only, one write per
   non-empty flush, records joined in arrival order); `flush_all()` iterates sessions, no-ops on
   `DUMP_DIR is None`/empty-pending, never raises (per-session failures logged, batch retained), contains no
   `await`, never touches `turn`/`prev_text`. **No caller exists** — the triggers are T003's (022 correction).
   Defensive clause: `_flush_state` returns without writing when the format is `documented` (nothing may be
   pending there; a mid-run config mutation cannot mint a `documented.log`).
7. **F-5 guard**: the bare `assert config.DUMP_DIR is not None` replaced by `raise ValueError`.
8. **Module docstring** rewritten: collapsed artifact + both encodings described.
9. **`norecursedirs`** added to `[tool.pytest.ini_options]`: the 9 pytest-default patterns retained (verified
   against installed pytest 9.0.2's `addini` default — exact match incl. `_darcs`) plus `qa`, so the ini's
   replace-semantics cannot re-enable recursion into `tests/qa/` and the default run stays hermetic.
10. **Backlog absorption** (same touch-set): `main.py`'s `--upstream` now reads `default=config.UPSTREAM`
    (QB:upstream-default — the duplicate literal is linked; `--help` output unchanged).

## Failure-Path Evidence

| Operation / condition | Expected outcome and authority | Handling / observation evidence | Disposition |
|---|---|---|---|
| `DUMP_DIR is None` at `dump_turn` | Real `ValueError` guard, never a bare assert (F-5, T002 done-when); caller guards + swallows (claim 13) | `dump_turn` raises `ValueError` before any effect; `test_dump_dir_none_raises_value_error`; caller contract untouched | safe (tested) |
| Unknown `config.DUMP_FORMAT` | Hard failure, never silent fallback (`PI:format_set_is_closed`); request never fails (swallow-all) | `_resolve_format` raises before state creation — no fs effect; `test_unknown_format_raises_and_writes_nothing` (dir listing empty); CLI rejects via `click.Choice` (`test_invalid_choice_is_rejected`) | safe (tested both seams) |
| Write through a symlinked/absent-identity target (flush artifact, turn files, index) | R2 durable-writer discipline: reject symlinked/non-regular targets before every write into the pre-existing dump tree; flush must not raise, documented writer propagates (swallowed upstream) | `_writable_target` checks `is_symlink()` first (dangling included) then exists/is_file; fail-first tests: `test_flush_refuses_symlinked_artifact`, `test_flush_refuses_dangling_symlink_artifact` (target file never created, batch retained, no raise), `test_documented_writer_refuses_symlinked_turn_file` (raises, target untouched) | safe (tested) |
| Flush write fails mid-append (disk full, permissions) | `flush_all` never raises; batch retained for the next flush; logged (plan: "must not raise") | `test_flush_never_raises_on_write_failure` (mocker OSError on `Path.open`; pending intact, no propagation) | safe (tested) |
| Documented path partial failure (turn file ok, index raises) | Same shape as today: exception propagates to the caller's swallow-all; no buffering side effect (documented path buffers nothing) | Guard/write order preserved (turn file → index); F-21 keeps `pending` empty on this path | safe (preserved shape) |
| Flush with `DUMP_DIR is None` or nothing pending | No-op, no artifact, no raise (plan clause) | `test_flush_noop_when_dump_dir_none`, `test_flush_noop_when_nothing_pending`, `test_flush_skips_batch_when_format_is_documented`, `test_documented_run_then_flush_creates_no_artifact` | safe (tested) |
| Format changed across process restarts (same DUMP_DIR) | Artifacts keyed `(sid, format)` never merge; documented-layout leftovers neither migrated nor removed (015/019 artifact identity) | `test_two_formats_never_merge_artifacts` (both `.log` files coexist, each own records); documented files in a pre-used dir are never touched by the collapsed path | safe (tested) |
| Cross-test pollution via module globals | Tests isolated (flaky-test P0 rule) | `dump_env` fixture sets `DUMP_DIR`/`DUMP_FORMAT`, clears `_sessions` before AND after, restores both; CLI tests use `restore_format` fixture + `ctx.close()` | safe |
| Concurrent same-session requests | Pre-existing design: sync function on one event loop; no `await` added anywhere in the dump path | `flush_all` documented + verified `async`-free; mutation points unchanged | not-applicable (pre-existing, unchanged) |
| Crash between append and flush | Buffered turns lost with the process — accepted plan consequence (buffered dump, shutdown flush arrives T003); no crash-consistency claim (no fsync required by plan) | Append-only artifact never rewritten, so a crash mid-write cannot corrupt earlier flushed batches (append is one `write` call) | planned (T003 shutdown flush; noted) |

## Dependency claims checked (installed source, one read per claim)

- **`expose_value=False` + `callback` keeps the value out of `main()`**: verified — click 8.3.1 `core.py:2589-2593`
  skips `ctx.params[name]` when `expose_value` is false; callback invoked in `process_value` (`:2438-2448`) at parse
  time. Encoded in `_set_dump_format` + `expose_value=False`; pinned by `test_flag_assigns_global_at_parse_time`
  (asserts both the assignment and `"dump_format" not in ctx.params`).
- **`make_context` exercises parsing without running the body**: verified — `Command.make_context`
  (`core.py:1182`) parses and returns the ctx; used as the CLI test seam so no server starts.
- **`norecursedirs` ini replaces the default list**: verified — `_pytest/main.py` `addini` default is exactly the
  9 patterns the plan names (incl. `_darcs`); `pytest_ignore_collect` matches dirs via `fnmatch_ex`; `_in_venv`
  independently covers `.venv`. Encoded as the pyproject value + `test_pytest_config.py` pins every default + `qa`.
- No claim the plan ruled on was found false — **no corrections, no silent overrides**. Claims resting on T002 in
  `assumptions/claims.md` (1, 2, 5-8, 11-13, 17, 19-23) re-read; all still HOLD against the landed tree (deps
  unchanged: aiohttp + click only; write forms byte-preserved; state keying/advance unchanged; `--dump-dir` still
  the only *other* dump option; the three absorbed claims 19/21/22 follow their DC closures — the symbols they
  said did not exist now exist **as this plan's authorized NEW anchors**).

## Capacity gate

**TRIGGERED (pre-recorded)** — the plan §Goals and Constraints + Yield's T001 guard required the extraction before
additive behavior. Executed in that authoring order:

- **Baseline (pre-edit)**: `dump_turn` L:44 N:1 C:7 (hard-L, the QB item); `_render_body` L:26 N:2 C:9 (pre-existing
  gray, untouched — its QB item explicitly defers to a later quality pass); `main.main` L:52 N:1 C:4 P:21
  (pre-existing hard P+L, plan-recorded as "reported, not cured"; T006 owns the L extraction); file sizes
  dump.py 115 / config.py 104 / main.py 130.
- **Extraction-first evidence**: helpers landed and the suite run green before the format branch/records/flush
  were written into `dump_turn`'s body (single increment, ordered authoring; the final scan confirms the outcome).
- **Post-change**: `codify scan path buster_ripper/dump.py --git --detailed` → **"No refactoring issues found"**
  (`dump_turn` L:11 N:1 C:2 P:1; all helpers ideal except `_diff_counts` C:2); `dump.py` **284 lines** (Green,
  exceeds the stale ~150-170 projection exactly as 018 predicted it would); `config.py` clean; `main.py`
  **unchanged on every recorded metric** (no new parameter — P not increased; no new body statement — L not
  increased); `flush_all` L:4 N:1 C:2, `_flush_session` L:4 N:1 C:2.

## L→N→C→P compliance

All new `dump.py` functions at ideal targets (≤15 L, ≤1 N, ≤5 C, ≤3 P) — scan clean. Test files: `test_dump_flush.py`,
`test_main_cli.py`, `test_pytest_config.py` clean; `test_dump.py` carries 4 **single-gray** findings — one L:18
(the byte-for-byte documented-preservation witness over three files) and three P:4 parametrized tests
(`(self, dump_env, fmt, n_turns)` — the `fmt` axis is the plan-required "run per shipped collapsed format").
At most one gray metric per function — within the checkpoint criteria; flagged for Yield's awareness.

## Tests written and coverage

`tests/test_dump.py` restructured (395 lines) + `tests/test_dump_flush.py` (237) + `tests/test_main_cli.py` (61) +
`tests/test_pytest_config.py` (39) — **56 dump/flush/CLI/config tests, 46 of them new** (T001's 16 reconditioned,
none deleted — see Durable signal):

- **Dispatch/guards**: documented default (three files, no `.log`); unknown format raises with zero fs effect;
  `DUMP_DIR` None raises `ValueError`; CLI choice-closed + default + parse-time assignment + `--help` listing.
- **Documented preservation** (local `VP-T002-documented-format-preserved`/`-default-encoding` witness):
  turn files byte-equal to independently recomputed `json.dumps`/`difflib` expectations; `index.txt` line formats
  regex-pinned (FULL and +a/-r forms with live counts); no `.log` on the default path — parametrized
  [no format, `documented` selected].
- **Collapsed buffering** (`VP-T001-writes-deferred` witness, corrected-to-T002 row): per-format buffering counts,
  arrival order, FK-B byte measure (== sum of rendering lengths, **≠** record bytes — divergence-by-design pinned),
  empty session dir before flush, no documented files ever, documented-path buffers nothing, session isolation,
  append-time chain advance, accumulation across every `_render_body` shape (restored after a coverage dip).
- **Record shapes** per format: turn-0 body+rendering (both), turn-≥1 diff text (change-focused) vs full rendering
  with computed-but-discarded diff counts (self-contained).
- **Flush**: exactly one `<format>.log` named by the configured value; contents == pending batch in arrival order;
  buffer cleared; `turn`/`prev_text` untouched; all no-op shapes; never-raises on OSError; documented-format batch
  skipped. **R2/R5 fail-first**: symlink/dangling-symlink refusal (artifact + turn file), two-format identity
  non-merge.

**Results**: full suite `uv run pytest` — **189 passed, 0 failed, 0 skipped**; total coverage **92.72%** (gate 80);
`buster_ripper/dump.py` **100%** (164 stmts, 0 miss); `config.py` 95% (the two missed lines are the pre-existing
`ModelProfile.describe`-adjacent block, untouched). Property-based note: `hypothesis` absent from the dev group
(unchanged from T001); the parametrized fmt × N × shape axes carry the property coverage (counts, order, byte-sum,
identity) inside the approved dependency set.

## Backlog pickup

`Backlog pickup: 2 absorbed, 3 not matching` — **absorbed**: QB:dump-turn/length-extraction (the plan-mandated
extraction itself; scan evidence above), QB:upstream-default/duplicate-declaration (main.py already in touch-set;
`default=config.UPSTREAM` links the literals, `--help` unchanged). **Not matching**: QB:entry-points/duplicate-main
(server.py not in touch-set — T002 touches no server code), QB:capacity-baseline/function-metrics (plan text,
not Forward's to rewrite; this report's Capacity gate carries the per-function baseline it asks for),
QB:render-body/gray-compound (`_render_body` untouched by design — the item itself schedules a later pass).

## Assumption finding (claim-19 — T002's Planned Task Context)

- **Finding**: claim-19, disposition `absorbed`, closure `DC-claim-19-r1` — the collapsed-artifact role stays
  `Mode: NEW` on `RB-003`; the falsifying `index.txt` append is the documented layout's own, preserved untouched.
- **Implemented per Required behavior**: the increment's collapsed surface is exactly the filename constant +
  path builder + append target for the two collapsed formats; `Path.open("a")` reused as-is; the documented
  `index.txt` append retained byte-unchanged inside the documented branch.
- **Done-when satisfied**: fresh session dir + collapsed format → exactly one `<format>.log`, records in arrival
  order, no `turn_000.json`/`turn_NNN.diff`/`index.txt` from the collapsed path (tested); documented path still
  produces its three files byte-unchanged (tested; the instrumented `2N` count is T004's gate).
- **Assumption impact**: `scoped:S1`. **Status: pending → implemented, reported for Yield** (not marked resolved —
  Yield moves it after replaying the fresh-dir collapsed run and the no-format documented run).

## Durable signal

- **Overrides exercised**: none displaced. The three dump `OI:` blocks' `preserve` decisions are honored
  (documented writer byte-preserved; append-time chain; turn-0 full body in every format).
- **Decision changes**: none. Forward-detail choices inside the plan's bounds, recorded for review: (a) record
  block format — summary line + `--- turn NNN body/rendered/diff ---` headers (locateability the VPs require;
  summary lines byte-identical to the documented forms); (b) flush join `"\n\n".join(batch) + "\n"`;
  (c) unknown-format check runs before state creation (no fs side effect); (d) `_flush_state`'s documented-format
  no-op guard.
- **Bug-fix rationale**: none (no defect found).
- **Test-suite reconditioning (planned behavior change, not weakening)**: T001's tests assumed the unconditional
  append; F-21 makes buffering collapsed-only, so `TestPendingBuffer`/isolation/advance tests now run under a
  collapsed format, the bytes-track-records property inverted to the divergence-by-design property (FK-B), and
  `TestDocumentedWritesStayLive` became the documented-preservation witness asserting `pending` stays **empty** on
  the documented path. Coverage of `_render_body`'s shape branches was restored after a rewrite dip (dump.py 100%).
- **Invariant entries changed**: none (triad rule — Forward writes no invariant records; the registry's deferred
  T002 rows are Yield's to register).
- **Backlog**: 2 items closed with evidence (see Backlog pickup).

## Concerns

- **R2 guards on the documented write sites** (turn files + index, not just the new artifact append): the
  adversarial-surface skill mandates auditing *every* writer into the pre-existing tree, and sibling writers keep
  the hole otherwise. For regular files the behavior is byte-identical (the preservation tests prove it); only
  symlinked/non-regular targets now raise (`ValueError`, swallowed upstream) instead of being written through.
  Flagging explicitly since the plan's "documented path unchanged" clause didn't enumerate this case.
- **main.py's pre-existing hard violations** (P=21, L:52) untouched by design — the recorded closure assigns the
  L-extraction to T006; this increment adds zero parameters and zero body statements there.
- `test_dump.py` (395 lines) sits in the FL "review" band; split moved the flush/guard/identity classes to
  `test_dump_flush.py` to stay under the 500 must-extract line. Further splitting deferred — file is cohesive
  (format dispatch + records).
- No `TodoWrite` tool exists in this runtime — progress tracked in this report.
- **Visual demo evidence**: T002 is the first evidence-bearing increment — captured and indexed
  (`t002-collapsed-artifact.txt`: real `--help` with `--dump-format`, real one-artifact session-dir listing,
  real artifact contents for both collapsed formats).

## Questions Asked

None — no materially different admissible outcomes remained. The plan pins the format set, defaults, callback
wiring, artifact identity, record contents, and the no-caller flush; remaining freedom (block headers, join
separator, guard placement) has one reasonable reading each, documented in Durable signal.

## Status for Yield

T002 implemented exactly as specified: format selector (config + CLI, callback-wired), format branch in `dump_turn`,
both collapsed record builders, artifact path builder + `flush_all` (no caller — T003's), F-5 guard, module
docstring, `norecursedirs` exclusion, capacity extraction BEFORE additive behavior (the T001 guard discharged),
R2/R5 fail-first guards. Full suite **189 passed at 92.72%**, `dump.py` 284 lines / 100% / scan-clean; backlog
pickup recorded; claim-19 implemented and reported. Next planned increment: **T003** (volume trigger + timer +
`on_cleanup` wiring — `flush_all`'s only callers).

DEMO-EVIDENCE: .sessions/codify/02.00-2026-09-13-message-dump-bulk-flush--9bf7afff5573/evidence/INDEX.md — no pause
