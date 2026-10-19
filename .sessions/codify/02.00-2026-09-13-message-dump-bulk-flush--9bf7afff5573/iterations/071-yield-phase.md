# Iteration 071 — Yield Phase (MS-001 / QAP-MS001)

## Orchestrator Prompt (verbatim)

```text
You are a subagent spawned by another session.
Session ID: 02.00-2026-09-13-message-dump-bulk-flush--9bf7afff5573
Milestone: MS-001
QA Pass: QAP-MS001
Target base: main
Target object: 72e3a1c6861bf0d8de6317adf52289e5f6cfa4e1
```

Orchestrator instruction line:

```
Execute the yield-phase-interactive skill for this session. Arguments: $ARGUMENTS
```

## Invocation Contract Verification

- Session handle `02.00-2026-09-13-message-dump-bulk-flush--9bf7afff5573` verified against
  `session-identity.json` (`session_id: 02.00-2026-09-13-message-dump-bulk-flush`,
  `session_uid: 9bf7afff-5573-402a-b102-827423a1dded`) — **MATCH** (`--9bf7afff5573` = the uid's
  first 8 + next 4 hex groups). Files read directly; nothing globbed or searched for.
- Full milestone gate: `Milestone: MS-001` + `QA Pass: QAP-MS001` + `Target base: main` + full
  `Target object` — all present. Contract satisfied.
- **BASE-SYNC gate: verified here** — see §Base-Sync Re-Verification.
- Constituent tasks: `MS-001 = T001, T002, T005` (Milestone Matrix). All three accepted by local
  Yield: T001 [031-01-yield](031-01-yield.md), T002 [031-02-yield](031-02-yield.md), T005
  [049-01-yield](049-01-yield.md); all archived in `plan-completed.md` rows 7, 8, 10.
- `QAP-MS001` authored at [070-forward-qa](070-forward-qa.md); the invocation's `QAP-*` matches
  `MS-001`'s row in `qa-plan.md` §QA Author Passes (line 63).
- **Upstream reachability — independently confirmed at this boundary** (see §Check 6): required to
  discharge `US-009`'s owned factual claim, whose arbiter is this milestone
  (`decisions.md:792` — `Verification: yield-phase:MS-001 PENDING`).
- Review scope: cumulative diff `72e3a1c6861bf0d8de6317adf52289e5f6cfa4e1...HEAD`, with the
  milestone's owned surface (`dump.py` buffering + format branch, `config.py` selector globals,
  `main.py --dump-format`, `README.md` §Dump mode) re-derived because `MS-001`'s gate is being
  certified after `MS-002`/`MS-004` closed.

## Base-Sync Re-Verification

Latest base-sync report is `BASE-SYNC: unchanged` ([054-01-forward](054-01-forward.md)), whose target
base object id is `72e3a1c6861bf0d8de6317adf52289e5f6cfa4e1` — **the same object this invocation
names**. No newer base-sync report exists (grep over `iterations/*.md`: the only `BASE-SYNC` carriers
are 033, 035, 054, 056, 058, 069 + `chronicle.md`/`plan-summary.md`, and 054 is the newest report).
Re-verified at HEAD `c7420ce05c921dafcc4720186b186608f7bf8d86`:

```
$ git rev-parse main origin/main
72e3a1c6861bf0d8de6317adf52289e5f6cfa4e1     # = the target object exactly
72e3a1c6861bf0d8de6317adf52289e5f6cfa4e1
$ git merge-base --is-ancestor 72e3a1c6… HEAD && echo true
true
$ git rev-list --count 72e3a1c6…HEAD ; git rev-list --count HEAD..72e3a1c6…
190
0
```

- Exact object is an ancestor of `HEAD`; **190 ahead / 0 behind**; the only worktree change during this
  pass is this report file (`git status --porcelain` → one `??` entry under `.sessions/`), no
  production source moved.
- Every correctness-required integration repair passed local Yield: T001/T002/T005
  ([031-01](031-01-yield.md)/[031-02](031-02-yield.md)/[049-01](049-01-yield.md)) plus the whole
  accepted-local chain since (T003, T003.001, T004, T006–T014).
- No final-phase preventive merge work is deferred: `MS-004` is the final phase and closed at
  [056-yield-phase](056-yield-phase.md), and [054-01-forward](054-01-forward.md) reviewed all 15 open
  `quality-backlog.md` items for merge exposure and emitted no prework packet.
- **`BASE-SYNC: unchanged` therefore holds at this boundary → the integration verdict is permitted to run.**

## Check 1: Caller Verification — PASS

Scope: every new public symbol the MS-001 slice adds (`dump.py` buffer/flush/format surface,
`config.py` selector globals, `main.py` CLI option) plus the wiring each task's done-when names.

**New public entry points → production callers**

| Symbol | Added by | Production callers | Verdict |
|---|---|---|---|
| `dump.flush_all()` | T002 | `dump.py:206` (`poll_flush` loop body, T003), `dump.py:133` (`_append_record` volume check, T003), `server.py:112` (`_final_dump_flush` ← `on_cleanup`, T003) | 3 callers — wired |
| `dump.poll_flush()` | T003 (outside MS-001) | `server.py:45` `spawn_dump_flush` ← `on_startup` `server.py:75` | wired |
| `dump.dump_turn()` | extended T001/T002 | `routes/messages.py:59`, guarded by `config.DUMP_DIR is not None`, inside the swallow-all `try` at `:57-60` | wired, contract intact |
| `config.DUMP_FORMATS` / `config.DUMP_FORMAT` | T002 | read at `dump.py:141-143` and `main.py:80`; written only at `main.py:14` | wired, one write site |
| `main._set_dump_format` (the `--dump-format` callback) | T002 | bound at `main.py:82` (`callback=`), option declared `main.py:80-85` | wired |
| `serve_buster_ripper.sh` `DUMP_FORMAT` env → `--dump-format` | T005 | `serve_buster_ripper.sh:29` default + `:41` guard/append, forwarded at `:54` | wired |

`grep -rn "flush_all\|poll_flush\|DUMP_FORMAT" --include='*.py' buster_ripper/` output is the table's
evidence; no zero-caller public symbol on the milestone's surface.

**Module-private helpers (dead-code sweep, the same check for `_`-prefixed entry points)** — every one
of the 25 private symbols in `dump.py` has ≥ 1 production call site (per-symbol count, excluding the
`def` line): `_resolve_format` 1, `_lookup_or_create_state` 1, `_write_documented` 1,
`_write_full_turn` 1, `_write_diff_turn` 1, `_append_index_line` 1, `_append_record` 2,
`_change_focused_record` 1, `_self_contained_record` 1, `_turn_zero_block` 2, `_full_summary` 2,
`_diff_summary` 3, `_msg_tool_counts` 2, `_unified_diff` 3, `_diff_counts` 3, `_artifact_path` 1,
`_flush_session` 1, `_flush_state` 1, `_require_writable` 3, `_writable_target` 2, `_render_tools` 1,
`_system_text` 1, `_content_text` 1, `_render_body` 2, `_now` 5. No orphan left by the capacity
extraction.

**Removed callbacks/hooks**: MS-001 removes none. `on_startup`/`on_cleanup` are `extend`, not replace
(`plan-summary.md` §Anchor Map, "shutdown flush seam"), and the pre-existing `spawn_kv_cache_monitor`
/`kv_monitor_task` wiring is still live at `server.py:72`/`server.py:90`.

**P2 sweep (called only from tests)**: none. `tests/qa/test_qap_ms001.py` reaches the same surface
through the real binary, so no production symbol here is test-only.

## Check 2: Migration Parity — PASS

The replaced path is the documented dump writer: `buster_ripper.py:_dump_turn` at the synchronized
target object (the only dump writer that existed at `main @ 72e3a1c6…` — `git ls-tree` shows the base
carries no `buster_ripper/` package at all) → `buster_ripper/dump.py`'s documented branch, reached
through the format dispatch T002 added.

Side-effect-by-side-effect parity (old body read from `git show 72e3a1c6…:buster_ripper.py`, lines
504-565; new from `dump.py:94-215`):

| Old side effect (`_dump_turn` at base) | Counterpart in the new documented writer | Parity |
|---|---|---|
| `assert DUMP_DIR is not None` | `dump.py:96-97` raises `ValueError("dump_turn requires config.DUMP_DIR to be set")` | **intentional change, documented** — T002's clause "`dump_turn`'s bare `assert` … is replaced by a real guard raising `ValueError` (`F-5`)" |
| `sid = _session_id(body)` | `dump.py:152` `session_id(body)` (`session.py:93-98`, same `sha256(system)[:12]`) | same |
| `session_dir.mkdir(parents=True, exist_ok=True)` on first turn | `_lookup_or_create_state` `dump.py:157-159` | same |
| `log.info("dump: new session %s → %s", sid, session_dir)` | `dump.py:161` — identical message and args | same |
| `current_text = _render_body(body)` | `dump.py:100`; `_render_body`/`_render_tools`/`_system_text`/`_content_text` reproduce the base's section builder verbatim (tools names-line + full defs, `[system]`, `[message i / role]`, `"─"*80` separators, leading `"\n\n"`) | same |
| turn 0 → `turn_000.json` = `json.dumps(body, indent=2, ensure_ascii=False)`, `encoding="utf-8"` | `_write_full_turn` `dump.py:171-175` — identical call and encoding | same |
| turn 0 summary `f"turn 000  {now}  FULL  {n_msgs} msgs  {n_tools} tools  {size_kb:.1f} KB"`, `size_kb = len(current_text)/1024` | `_full_summary` `dump.py:225-229` — field-for-field identical, `len(text)/1024:.1f` | same |
| turn ≥ 1 → `difflib.unified_diff(prev, curr, fromfile=f"turn_{turn-1:03d}", tofile=f"turn_{turn:03d}", lineterm="")`, written as `turn_{turn:03d}.diff` with `"".join(diff_lines)` | `_unified_diff` + `_write_diff_turn` `dump.py:178-186`, `dump.py:246-253` — same args, same join | same |
| added/removed counts excluding `+++`/`---` headers | `_diff_counts` `dump.py:262-266` — same two comprehensions | same |
| turn ≥ 1 summary `f"turn {turn:03d}  {now}  +{added}/-{removed} lines  {n_msgs} msgs  {n_tools} tools"` | `_diff_summary` `dump.py:232-236` | same |
| `log.info("dump: %s", summary)` on both turns | `dump.py:176`, `dump.py:186` | same |
| `index.txt` append `summary + "\n"` via `index.open("a", encoding="utf-8")` | `_append_index_line` `dump.py:189-194` — same mode, encoding, and payload | same |
| `state.prev_text = current_text; state.turn += 1` at the end | `dump.py:107-108` — after the branch, so both writers advance identically; `flush_all` "never touches `turn`/`prev_text`" (`dump.py:118` docstring; `_flush_state` writes only the artifact) | same |

**Old side effect with no counterpart: none.**

**New behavior on the retained writer, checked for justification (the reverse direction):**
`_require_writable` now refuses a symlinked/non-regular target before each of the three documented
writes, where the base would have written through. This is an addition, not an omission, and it is
already documented and adjudicated in this milestone's own slice: [031-02-forward](031-02-forward.md)
§Adversarial Surface rows name it under the R2 durable-writer discipline with three fail-first tests,
and [031-02-yield](031-02-yield.md) classified `OI:…/output_layout` + `documented_format_is_retained`
as **Aligned** — "byte-preserved for regular targets; the R2 refusal applies only to symlinked /
non-regular targets, which no invariant enumerates and the adversarial-surface discipline mandates".
`MS-001`'s own gate re-proves it from outside the process (`test_hostile_r2_documented_writer_does_not_escape_the_dump_root`).

**T005 parity (documentation migration):** `git show ad5c914 -- README.md` is `+53 / −0` — the dump
section is extended, nothing in the documented block is removed or rewritten. The published reader
contract the branch inherits from the base (`turn 000  10:31:02  FULL  3 msgs  83 tools  412.3 KB`,
`turn 001  10:32:18  +47/-3 lines  5 msgs  83 tools`, `turn 002  …   ← large file read`) is present
verbatim, and `TC-108` asserts the implementation against those same published lines. No tool section
was added (`FK-007` option 1 ships none) and none exists in `README.md`'s heading inventory.

## Check 3: Data Flow Completeness — PASS

Traced by reading code, end to end, on the current snapshot.

**Path A — the documented writer (`G-008`(a)'s path, MS-001's default):**

```
client → POST /v1/messages
  → server.py:124 router.add_post → routes/messages.handle_messages
  → body parsed/normalized (messages.py:20-55)
  → messages.py:57  if config.DUMP_DIR is not None        ← guard: producer enabled
  → messages.py:59  dump_turn(body)                        ← inside try/except, :60 log.exception("dump failed")
  → dump.py:99      fmt = _resolve_format()                ← config.DUMP_FORMAT (main.py:14, parse time)
  → dump.py:104-105 documented branch → _write_documented
  → _write_full_turn / _write_diff_turn → _append_index_line
  → DUMP_DIR/<sid>/{turn_000.json|turn_NNN.diff, index.txt}
  → back in handle_messages: forward_headers → app["session"].post(upstream) → response
```
Every handoff is a direct call in the same coroutine; nothing is dropped between `dump_turn` and the
write, and the request path does not depend on the dump path's success. **COMPLETE.**

**Path B — the collapsed writer (buffer → drain):**

```
dump.py:106-108 change-focused / self-contained branch
  → _change_focused_record / _self_contained_record → record (str)
  → _append_record: state.pending.append(record); state.pending_bytes += len(current_text.encode("utf-8"))
  → (in-session volume drain) dump.py:133 flush_all()          ← producer of the write
  → (periodic drain) server.py:45 app["dump_flush_task"] = create_task(dump.poll_flush())
                      dump.py:203-207 poll_flush → flush_all(); await sleep(config.DUMP_FLUSH_INTERVAL)
  → (shutdown drain) server.py:88 _final_dump_flush() → dump.flush_all()  [server.py:112]
  → flush_all: dump.py:119 DUMP_DIR None → return; else per-state _flush_session → _flush_state
  → _artifact_path(state, state.fmt) = DUMP_DIR/<sid>/<fmt>.log → ONE open("a") + write
```

Wiring verified by name-match, not by inference: the spawner writes `app["dump_flush_task"]`
(`server.py:45`) and `on_cleanup` cancels the identical key `server.py:87`; `spawn_dump_flush(app)` is
called from `on_startup` at `server.py:81`, and `on_startup`/`on_cleanup` are registered at
`server.py:120-121`. The task is therefore **actually started**, not merely defined.

Teardown ordering matches the plan's clause exactly — cancel the KV monitor, cancel the flush task,
**then** flush, **then** close the shared session (`server.py:86-93`) — so the final flush still has a
live process and no in-flight timer tick racing it.

**No-drop / no-corruption checks at each handoff:**
- Producer → buffer: the record is the fully-built string; nothing is re-derived at flush time, so a
  flush cannot lose a turn whose record was appended (witnessed: `TC-100` finds all three turns'
  markers in the one post-run artifact).
- Buffer → artifact: `"\n\n".join(state.pending) + "\n"` keeps arrival order and per-turn identity
  (each record opens with its own summary line).
- Failure paths retain rather than discard: `_flush_session` logs and leaves `pending` intact
  (`dump.py:270-277`), `_flush_state` clears `pending`/`pending_bytes` **only after** the write
  succeeds, and `flush_all` cannot raise into `on_cleanup` (`server.py:110-113`). A failed flush is
  therefore a retryable stall, not lost data — and MS-001's gate proves the stalled case writes
  nothing to the unwritable directory (`test_tc001_collapsed_flush_failure_…`).
- Cross-step timing gap: the buffer is process-local (module `_sessions`), and the only writer is
  `flush_all`; `dump_turn` runs on the request coroutine while `poll_flush` runs on the timer task.
  Both are in one event loop and neither `await`s between the append and the threshold check
  (`flush_all`/`_flush_state` contain no `await` — the plan's own clause), so a tick cannot interleave
  half-built state. `G-003`'s "no new await on the request path" holds.

**Data paths MS-001 does NOT break but that share the surface:** `/v1/chat/completions`,
`/v1/messages/count_tokens` and the passthrough route never call `dump_turn` (only `messages.py:59`
does) — confirmed by the `dump_turn` caller grep. `config.DUMP_DIR`'s other consumers are
`main.py:152-153` (producer) and `main.py:175` (startup log line), unchanged.

## Check 4: Consumer Contract Tracing — PASS

**Dispatch-bearing fields:**

| Producer | Field | Values it can create | Consumers | Handled? |
|---|---|---|---|---|
| `main._set_dump_format` / `config.DUMP_FORMAT` | `config.DUMP_FORMAT` | only `click.Choice(list(config.DUMP_FORMATS))` members — `documented`, `change-focused`, `self-contained`; anything else is refused before serving | `dump._resolve_format` (membership test, `ValueError` otherwise), `dump_turn`'s three-way branch, `_flush_state`'s `== "documented"` early return | **yes** — the closed set and the branch are the same three values; an out-of-set value cannot reach a consumer because `_resolve_format` raises first and the caller swallows |
| `dump_turn` | `state.fmt` | the value `_resolve_format` just returned | `_flush_state` (path builder / early return), `_artifact_path` | **yes** — assigned immediately before the branch that fills `pending`, so a batch and its filename share one value |
| `_full_summary` / `_diff_summary` | summary lines in `index.txt` and in every collapsed record | `turn NNN  HH:MM:SS  FULL … KB` and `turn NNN  HH:MM:SS  +A/-B lines … msgs … tools` | the **human/shell reader** documented at `README.md` §Reading the index / §Reading a collapsed artifact, whose published pattern is `^turn [0-9]{3}  [0-9]{2}:[0-9]{2}` and whose columns are `FULL|+A/-R lines`, `<n> msgs`, `<n> tools`, `[KB]` | **yes** — and the reader contract is gated, not assumed: `test_tc108_sensitivity_documented_line_pattern_distinguishes_drift` proves the matcher accepts all three README-published lines and rejects five drift forms, so the content assertions in `TC-107`/`TC-108` can actually fail |
| `counted_proxy_entry.main` | the JSON write report | keys `dump_root`, `ops[{path,mode,bytes}]`, `total_ops`, `total_bytes` | `TC-100`/`TC-108` via `count_report`/`ops_by_name` | **yes** — schema emitted matches schema consumed |

**Merged/coalesced data (the case this check exists for):** `flush_all` coalesces a session's whole
pending batch into **one** write. The dispatch-bearing field inside that merge is the record's own
`turn NNN` identity, which is *per record* and survives the join (`"\n\n".join`), so ordering and
per-turn identity are preserved — verified by `TC-100` (all three markers, one file, one write op).
The merge never spans sessions: `flush_all` iterates `_sessions.values()` and `_flush_state` writes
one path per state, so a per-`sid` identity cannot be blended — verified by `R5`'s near-miss pair.

**The one heterogeneous-merge shape the plan forbids rather than gates:** a session whose `pending`
holds records built under format A while `state.fmt` reads B. That state is reachable only by mutating
`config.DUMP_FORMAT` at runtime, which `PI:T002/dump_format/format_is_per_artifact` declares does not
exist ("chosen once per process, set by `main()`, never mutated") — and the source agrees: the only
production write to that global is the click callback (`main.py:14`), which runs at parse time, before
`create_app()`/`web.run_app` (`main.py:188-189`). Recorded at claim-93's closure
(`DC-claim-93-r1`, `plan-summary.md` §Task List `T004`) that no runtime format-migration behavior is
authorized. For the same reason `_flush_state`'s early return on a `documented` state with a non-empty
batch **retains** the batch rather than discarding it (`pending` is cleared only after a successful
write), so even the unauthorized interleaving cannot silently lose data. **Not a finding**; the
surviving consequence — that such a batch would never be written while the process stays in
`documented` — is outside the approved contract and is not a shipped configuration.

## Check 5: Plan Fulfillment — PASS

Done-when clauses taken from `plan-summary.md` §Task List as they stood at T002/T005's acceptance
(Checkpoints 64/101/102 — `git show` of the file at those commits, since the current §Task List
archives them to `plan-completed.md`). Verified against the source at HEAD, not against a report.

**`T001` — the per-session pending-record buffer.** *Purely additive; `REMOVED: nothing`.*
- ADDED `pending: list[str]` + `pending_bytes: int` on `SessionState` → `dump.py:37-41` ✔
- ADDED `turn`/`prev_text` advance at append time → `dump.py:107-108`, after the exclusive branch ✔
- REMOVAL CHECK (the ratchet the plan set: "the three write sites and the `index.txt` append stay
  live") → all four write sites present: `dump.py:174` (`turn_000.json`), `:186` (`turn_NNN.diff`),
  `:196` (`index.txt` append), `:302` (the collapsed artifact). Nothing was removed early ✔
- The one clause T001 itself says `T002` must later condition ("in this increment the append is
  unconditional") **is** conditioned at HEAD: `_append_record` is reachable only from the two collapsed
  arms (`dump.py:104-108`), so the documented path never buffers — `F-21`/`F-13` honored, and
  `PI:…/one_write_path_per_turn` holds structurally: `dump_turn` is a three-arm exclusive dispatch, so
  no turn can be written by both writers and no configuration can reach `2N + F` (witness C).

**`T002` — format branch, collapsed artifact, single-write flush.**
| Clause | Where | Verified |
|---|---|---|
| `config.DUMP_FORMATS` closed 3-set + `DUMP_FORMAT = "documented"` | `config.py:27-28` | ✔ exact values; the R2 candidate 2-set was not authored |
| `--dump-format` = `click.Choice(list(config.DUMP_FORMATS))`, `default="documented"`, `show_default=True`, assigns **through the option's own callback** | `main.py:80-85` + `main.py:12-15` | ✔ `main()`'s signature (`main.py:131-137`) carries **no** `dump_format` parameter — the capacity closure held |
| Capacity extraction **spanning both writers**, before additive behavior | `_lookup_or_create_state`, `_resolve_format`, `_write_documented` (+`_write_full_turn`/`_write_diff_turn`/`_append_index_line`), `_change_focused_record`, `_self_contained_record`, `_artifact_path` | ✔ all six named roles exist as single-concern module-private helpers; `dump_turn` is now L:11 (was L:44, a hard-L violation) and contains no inline record construction |
| Unknown format raises `ValueError`; caller's swallow-all keeps the request alive | `dump.py:141-143` → `routes/messages.py:57-60` | ✔ and gated from outside at `test_hostile_r1_unknown_dump_format_is_refused_and_writes_nothing` |
| `flush_all()`: one write per session, clears buffer, no-ops on `DUMP_DIR is None` / nothing pending, must not raise, **no `await`**, never touches `turn`/`prev_text` | `dump.py:111-123`, `:270-306` | ✔ the only `await` in the module is `poll_flush`'s sleep (`:207`, T003's); `_flush_state` clears after the write and never assigns `turn`/`prev_text` |
| Artifact = `DUMP_DIR/<sid>/<format>.log`, `<format>` exactly the configured collapsed value; `documented` builds no artifact path | `_ARTIFACT_SUFFIX` `:30`, `_artifact_path` `:280-282`, early return `:296-297` | ✔ |
| Both collapsed records per spec (turn 0 body **and** rendering; later diffs / renderings); summary line built in both, format-independent; documented summary lines unchanged | `:210-236` | ✔ |
| Bare `assert` → real guard | `dump.py:96-97` | ✔ (`F-5`) |
| **(022) `pyproject.toml` `norecursedirs` delivering the hermetic/network-free constraint, retaining pytest's defaults** | `pyproject.toml:20-24` | ✔ `qa` present **and** all ten default patterns retained; gated by `tests/test_pytest_config.py`; re-proven at this boundary — the bare run collected 407 tests in 9.49 s with no network and no subprocess |
| **REMOVED (dropped by 018/019) — must NOT be present:** the three documented write sites deleted | — | ✔ **not removed**: `G-008`(a)'s writer is retained as the documented branch. The plan's own capacity paragraph projected "still Green (<300)" for post-`T002` `dump.py`; T002 landed at **284** ✔. Current `dump.py` is 316 lines, growth the plan names as plan-forced (`server.py` is coverage-omitted, so the timer loop body must live in `dump.py`) and which is owned, with the current figure, by `QB:dump/file-size-yellow` → `roadmap.md` §RM-002. Not an unowned overrun. |

**`T005` — extend the README's dump section.** *"`REMOVED: no tool section`"*
- Collapsed artifact name/shape, every shipped format and how it is selected, per-format `grep`/`less`
  reading workflow, the `DUMP_FORMAT=… bash serve_buster_ripper.sh` run path, `F-8`'s arrival-order
  caveat, and the pre-existing-directory note → all present at `README.md:169-221`; the documented
  block stays where it was ✔
- `serve_buster_ripper.sh` gains `DUMP_FORMAT` mirroring `DUMP_DIR`'s shape (empty default ⇒ the
  default path is untouched, `-n` guard) → `serve_buster_ripper.sh:29`, `:41` ✔. The clause's literal
  `EXTRA_FLAGS="${EXTRA_FLAGS} …"` string form was later replaced by the array form at T012
  (word-integrity fix, `evidence/t012-value-word-integrity.txt`); the clause's *requirements* — empty
  default, presence guard, exactly one flag when set — are what T012's witnesses pin, so this is not a
  deviation from the plan's intent.
- No tool section exists anywhere in `README.md`'s heading inventory ✔ (`FK-007` option 1).

**Milestone `Complete when` (MS-001 row) — each conjunct against an artifact at this boundary:**
| Conjunct | Witness |
|---|---|
| documented-format session → today's three files, today's content/line format, at `2N` writes | `TC-108` — `write_ops=6 == 2N`, per-path decomposition `index.txt:3 / turn_000.json:1 / turn_001.diff:1 / turn_002.diff:1`, contents asserted against the README's published lines |
| README workflow unchanged | `git show ad5c914 -- README.md` = `+53/−0`; §Check 2 parity |
| no format configured → documented layout | `TC-107` — exactly the four documented files, zero `.log` anywhere under the root, env scrubbed by an autouse fixture |
| collapsed format selected → N turns produce **no artifact on disk**, every write deferred | `TC-100` — three pre-flush snapshots `[(0.01, []), (0.02, []), (0.03, [])]`, then one artifact, `write_ops=1` |
| a raising dump/flush still cannot fail a request | `TC-001` both paths — status equal to a healthy control session, `PermissionError` logged, `rc == 0` |
| ⚠ 022: the milestone declares **no** flush | `grep -n "dump-flush" tests/qa/test_qap_ms001.py` → one hit, the docstring stating the rule; no `extra_args` carries either flag |

**Relay-vs-transfer test (the failure mode this check exists for):** `MS-001`'s architectural goal is
an *addition plus a retention*, not an ownership transfer — `US-012` forbids displacing the documented
writer before the `MS-002` comparison persuades the user. So the correct removal test here is the
inverse of the usual one: is there any `2N + F` double-write, any silent deletion of the retained
writer, any relay that merely forwards while the old ownership persists? The four-write-site census
and the exclusive three-arm dispatch answer no; and the switch the plan defers (`roadmap.md`'s default
flip) is genuinely still deferred — `config.DUMP_FORMAT`'s default is `"documented"` and the CLI's
`show_default` agrees.

## Check 6: QA Milestone Closure — PASS

**Handles and coverage (steps 1-3).** `MS-001` resolves from the Milestone Matrix to constituent tasks
`T001, T002, T005`, QA author pass `QAP-MS001` — the invocation's `QAP-*` matches. All three
constituents passed local Yield before this gate ran ([031-01](031-01-yield.md),
[031-02](031-02-yield.md), [049-01](049-01-yield.md)). Scenarios whose `Required at` is exactly
`MS-001`: **`TC-100`, `TC-001`, `MS-001`'s `TC-107`, `TC-108`** — and only those (`TC-002` is
`MS-003`'s, `TC-101…TC-112` are `MS-002`'s, `TC-003/113/114` are `MS-004`'s). Each scenario names a
`VP-*` whose `Observable after` / `Validate at` in `plan-summary.md` §Validation Point Matrix agree
with `qa-plan.md`, and all four of the milestone's VPs are covered — no orphan VP, no orphan scenario:

| Scenario | VP | Observable after | Validate at | Matches plan |
|---|---|---|---|---|
| `TC-100` | `VP-T001-writes-deferred` | T002 | MS-001 | ✔ |
| `TC-001` | `VP-T001-dump-failure-isolated` | T001 | MS-001 | ✔ |
| `TC-107` | `VP-T002-default-encoding` | T002 | MS-001 | ✔ |
| `TC-108` | `VP-T002-documented-format-preserved` | T002 | MS-001 | ✔ |

**Authorship (step 4).** `QAP-MS001` supplied the persistent gate module
`tests/qa/test_qap_ms001.py` (10 tests, committed at `c7420ce`), its Stage-1 §Contract Draft table, a
§Boundary Inspection Exceptions log (no implementation source or implementation report was read in
Stage 1; one over-strict assertion was removed *against* the implementation, not with it), and the
promotion record. Adequate — no rejection to `forward-qa` on authorship grounds.

**Contract fidelity (step 5), clause by clause against `qa-plan.md` §Scenario Contracts.**

| Contract element | How the gate implements it | Fidelity |
|---|---|---|
| `TC-100` trigger: real proxy, `--dump-dir` fresh tmp, `--dump-format change-focused`, **no flush parameter**, N turns through `POST /v1/messages` | `Proxy(dump, logs, extra_args=["--dump-format", COLLAPSED], counted_report=…)`; `N_TURNS = 3`; real subprocess + real HTTP | ✔ exactly, and the 022 directive is honored (see below) |
| `TC-100` expected owner: `dump_turn` buffers, `flush_all` is the only writer on this path | post-run: `listing(sid_dir) == ["change-focused.log"]`, no non-`.log` file anywhere under the root, and the instrumented report's op set `== {change-focused.log}` — ownership is asserted from the write stream, not from a code reading | ✔ |
| `TC-100` visible result: the session dir holds no artifact containing those turns | three pre-flush snapshots, each `[]`, each taken **after** that turn's acknowledgement | ✔ |
| `TC-100` forbidden: a per-turn file or `index.txt` on the collapsed path; one artifact per pending turn | asserted absent; `len(log_artifacts(dump)) == 1` | ✔ |
| `TC-100` lifecycle: healthy across N turns, the deferral **ends** at a flush | `rc == 0`; all three markers found in the one post-run artifact | ✔ |
| `TC-001` trigger: a request whose dump path raises, injected in the write/append path | process-level fault: the session dir is pre-created `0o500`, so every create inside raises `PermissionError` — no monkeypatching of production code, on both the documented write path and the collapsed flush path | ✔ real trigger |
| `TC-001` owner + visible: the swallow-all at `handle_messages`; upstream response returned, exception logged | `status_broken == status_control` against a healthy sibling session in the same process, `assert_upstream_response` (2xx-4xx, no `Cannot connect`), `poll_until("PermissionError" in stderr)`, and the control session's own documented layout afterwards | ✔ — see the oracle note below |
| `TC-001` forbidden: a 5xx from dump behavior; a silent failure | a 5xx is excluded for **both** sessions by the control comparison + `< 500`; silence fails the stderr poll | ✔ |
| `TC-001` lifecycle: keeps serving; collapsed variant's raise must not escape `on_cleanup` | `proxy.proc.poll() is None` mid-run; `rc == 0` with the fault **still injected across the shutdown flush** (the chmod is deliberately restored only after `stop()`) | ✔ the sharpest form of the test |
| `TC-107` trigger: no `--dump-format`, **no config override** | `Proxy(dump, logs)` + an autouse fixture deleting every `*DUMP*` environment variable, so "no override" is enforced rather than assumed | ✔ |
| `TC-107` visible / forbidden | exactly the four documented files, contents asserted, `log_artifacts(dump) == []` (witness W1 of `G-008`), plus a sensitivity pre-assertion that the session dir does not exist before the first request | ✔ |
| `TC-108` trigger: documented format at the same N, instrumented, no flush parameter | `--dump-format documented` + `counted_report=` | ✔ |
| `TC-108` visible: three files, today's content and line format, at `2N` write-bearing operations | `total_ops == 2N`, decomposed per path (`index.txt` N, each turn file 1) so a rewrite-per-turn or a lost index write cannot hide behind the total; contents asserted against `README.md`'s own published pattern and column set | ✔ |
| `TC-108` forbidden: drifted `index.txt` format, missing/renamed file, `.log` on the documented path | all three asserted, and `test_tc108_sensitivity_…` proves the matcher rejects five drift forms and accepts the README's three published example lines — the content assertions can fail | ✔ |
| lifecycle (all): counts are per session; quiescence | every counted op asserted to sit inside `DUMP_DIR/<sid>/`; readiness by port poll and quiescence by `poll_until`/process exit — no fixed sleeps | ✔ `qa-project` §Synchronization |
| telemetry: "none required" | every contract row's Telemetry cell is `none`, matching the project's declaration (no telemetry capability; `aiohttp` + `click` only) | ✔ |

**Oracle note (recorded, not a finding).** `TC-001`'s "the upstream response body is returned
unchanged" is witnessed as *the same status as a healthy control session plus no proxy transport
failure*, not as byte-equality of the model payload. That is the harness's own declared oracle
(`qa-project` §Determinism: "Gates must assert on the proxy's dump artifact and process behavior,
never on model output"), and the live upstream answers the gates' invented model with a `404`, so a
byte-equality oracle would be asserting the upstream's output rather than the proxy's non-interference.
The trigger, the owner, and the forbidden result are all asserted as written.

**022 directive — honored literally, and its expired premise ruled.** `qa-plan.md` §Phase Requirements
forbids any `MS-001` scenario from passing `--dump-flush-interval`/`--dump-flush-bytes`, on the stated
premise that "those options do not exist until `T003`". At this boundary they do exist (`T003` is
accepted), so the premise has expired while the directive has not. Ruling: **the directive stands, and
no plan change is owed.** `grep -n "dump-flush" tests/qa/test_qap_ms001.py` returns one hit — the
module docstring stating the rule — and no `extra_args` in the file carries either flag. `TC-100`
establishes "before any flush" by *observation* instead: the artifact is append-only and never removed,
so a snapshot holding no artifact **is** a pre-flush reading, and the test fails loudly and names the
reason if an environment is ever slow enough that a flush beats the first acknowledgement. It is
therefore not a sleep-and-hope race, which is the only thing the directive's expired premise was ever
protecting against. Relaxing it (injecting a disabled timer, which is what `TC-101`/`TC-106` do at
`MS-002`) would widen the milestone's contract rather than the test's; the current form satisfies
`VP-T001-writes-deferred`'s own wording — "the assertion is an **absence**, so it needs no trigger".

**Execution (steps 6-7).** Run independently at this boundary, not replayed from 070. Full transcript
in [`../evidence/ms001-gate-and-upstream-adjudication.txt`](../evidence/ms001-gate-and-upstream-adjudication.txt)
§2. `10 passed in 2.40s`, with the printed measurements identical to the authoring run
(`write_ops=1`, pre-flush snapshots `[(0.01, []), (0.02, []), (0.03, [])]`; `write_ops=6`, `files=4`).

| Handle | Verdict |
|---|---|
| `VP-T001-writes-deferred` / `TC-100` | **pass** |
| `VP-T001-dump-failure-isolated` / `TC-001` (documented write path) | **pass** |
| `VP-T001-dump-failure-isolated` / `TC-001` (collapsed flush path) | **pass** |
| `VP-T002-default-encoding` / `TC-107` | **pass** |
| `VP-T002-documented-format-preserved` / `TC-108` | **pass** |
| `VP-T002-documented-format-preserved` / `TC-108` sensitivity | **pass** (proves the above can fail) |
| hostile tier R1 selector / R1 body / R2 containment, identity tier R5 | **pass** ×4 |
| `not_applicable`: `TC-101…TC-112`, `TC-003`, `TC-113`, `TC-114` | not `MS-001`'s; `TC-002` is `MS-003`'s |

**Upstream reachability, established first-party (this is also `US-009` claim 2's arbitration):**
`require_upstream()` — the same fail-closed preflight every gate here runs through — passed in 0.05 s;
`GET /v1/models` answers `200` with a vLLM-shaped model list (`"owned_by":"vllm"`), and a real proxied
`POST /v1/messages` returned the upstream's own `404 NotFoundError` (`"The model \`qa-gate-model\` does
not exist."`) — not the proxy-local `502 … Cannot connect` that an unreachable host produces — while
creating that session's `turn_000.json` and `index.txt` on disk. This is a pass on the real trigger,
not a connection-error baseline credit.

**Intermediate QA sweep (step 8).** The `tests/qa/` tree contains exactly the three `QAP-*` gate
modules, their two instrumented entries (`counted_proxy_entry.py`, `tls_probe_entry.py`), the shared
`harness.py`, and `MS-004`'s peer material. No `MS-001` intermediate lingers. `070-forward-qa`'s
promotion claim verified independently: `read_text`, `rglob_files`, `log_artifacts`,
`assert_upstream_response` are each **single-defined** in `harness.py` with zero surviving duplicates in
`test_qap_ms002.py`, and `startup_refused` is single-defined and used by this pass. `MS-002`'s
remaining module-local helpers (`big_text`, `summary_lines`, `record_section`, `TIMER_OFF_INTERVAL`,
`TURN_SIZE`, `THRESHOLD_AT_TURN_2`, `UNREACHABLE_BYTES`) are genuinely specific to the collapsed-path
thresholds this milestone has no use for — retained, per the author pass's record. Nothing was deleted
because nothing met the delete criterion. `MS-001`'s unit-level modules (`test_dump.py`,
`test_dump_flush.py`, `test_dump_records.py`, `test_dump_write_count.py`, `test_main_cli.py`,
`test_pytest_config.py`) are permanent suite members carrying the `MS-003` coverage floor, not
intermediates; the four route-level modules the branch added earlier (`test_chat_routes.py`,
`test_messages_routes.py`, `test_passthrough_routes.py`, `test_session_pooling.py`) are already gone
from `git ls-files`.

## Check 7: Integration Suite Execution — PASS

**Scope derived from the diff before running anything.** The synchronized cumulative diff
(`72e3a1c6…HEAD`) changes code in exactly one package, `buster_ripper/` (all 14 modules), plus its two
test trees (`tests/`, `tests/qa/`), `serve_buster_ripper.sh`, `pyproject.toml` and `README.md`. The
package *is* the application, so its downstream dependents are its own consumers: the hermetic
integration layer (notably `tests/test_routes_integration.py`, which is `TC-002`/`VP-T003-passthrough-regression`)
and every gate in `tests/qa/` — all of which read `buster_ripper.config`'s globals and drive the real
binary. **Excluded, with the reason:** `buster_ripper_legacy.py`, the tracked pre-refactor monolith
referenced by nothing under `buster_ripper/` and excluded from scope by the plan itself
(`plan-summary.md` §Task List, "Not tasks"); `docs/` and `.agents/` are non-code. Nothing downstream of
a changed module was dropped, and the scope was not narrowed after any failure — nothing failed.

**Step 0 — the full unit suite once on this boundary's snapshot** (per-task Yield runs only affected
tests, so this is the run that could catch a regression in an unrelated module):
`uv run pytest` → **407 passed, 95.75%** in 9.49 s, zero failures, so no bisect was owed.
`dump.py` is at **100%** (180 stmts, 0 miss). Hermetic-ness of that run was proved, not assumed: the
default run collects **zero** `tests/qa` items, and `tests/test_pytest_config.py` gates that
`norecursedirs` carries `qa` alongside all ten of pytest's defaults.

**Steps 1-2 — the integration/E2E layer over that scope:**

| Run | Command | Result |
|---|---|---|
| `MS-001` gate (Check 6's own execution) | `.venv/bin/python -m pytest tests/qa/test_qap_ms001.py -o addopts= -q` | **10 passed** in 2.40 s |
| `MS-002` gate on these bytes | `.venv/bin/python -m pytest tests/qa/test_qap_ms002.py -o addopts= -q` | **15 passed** |
| `MS-004` gate on these bytes | `.venv/bin/python -m pytest tests/qa/test_qap_ms004.py -o addopts= -q` | **6 passed** |
| (the two `tests/qa` rows above were executed as one command) | — | **21 passed in 60.33 s** |

Gate tree total at this boundary: **31 process-level gates, all green**, on one snapshot, plus the 407
hermetic tests. `MS-002`'s and `MS-004`'s gates had to be re-run here rather than inherited: 070
changed `tests/qa/harness.py`, the file all three passes share, and "changed snapshots or
configurations require fresh applicable execution". Both are green, which also means the harness
promotion that `MS-001`'s pass performed broke neither neighbor — `MS-002`'s paired
`G-008`(b) measurement re-printed byte-for-byte (`F=2/2/2N=12`, files `1/1/7`,
bytes `482762/282647/442217`).

**Shared-run rule applied.** This single, independently executed invocation satisfies Checks 6 and 7:
its runs cover both complete required scopes on the same snapshot and the same harness configuration,
and the mapping is above — Check 6's required scope is `QAP-MS001`'s 10 gates with the four `VP-*`
handles, Check 7's is the hermetic suite plus the whole `tests/qa/` tree. No test was run twice to
produce two reports, and Forward QA's run was not counted as this pass's execution.

## Routing: owned factual claims

Every atomic entry under `Factual claims` in `decisions.md` carrying
`Verification: yield-phase:MS-001` was enumerated — **exactly one**, and no legacy handoff or interview
record holds an MS-001 claim without a canonical counterpart (every such reference in
`handoff-context.md`, `handoff-archive.md` and `assumptions/claims.md` points at this same row as its
named arbiter, so no `MIGRATION-REQUIRED` return is owed):

**`US-009` claim 2** — *"`http://hyperipper.local:30001` is reachable from the host that runs the QA
gates and serves the vLLM-compatible API the proxy expects."* (`decisions.md:791-798`)

- Plan-owned arbiter exercised: the milestone's own QA harness, at the phase scope, first-party.
- Verdict: **HOLDS**.
- Evidence locator now written into the record: `iterations/071-yield-phase.md` §Check 6
  ("Upstream reachability, established first-party") + this section, raw transcript
  `evidence/ms001-gate-and-upstream-adjudication.txt` §3.
- What was observed: the harness's own fail-closed preflight `require_upstream()` passed in 0.05 s;
  `GET /v1/models` → `200` with a vLLM-shaped model list (`"owned_by":"vllm"`, `max_model_len`); and a
  real proxied `POST /v1/messages` through a freshly started binary returned the **upstream's** own
  typed `404 NotFoundError` (`"The model \`qa-gate-model\` does not exist."`) while creating that
  session's `turn_000.json` + `index.txt`. The counter-signature — a proxy-local
  `502 … Cannot connect` — is what the same host's absence produced at
  [035-yield-phase](035-yield-phase.md), and it is the discrimination that makes this a HOLDS rather
  than a baseline credit.
- `Consequence: none`, per the HOLDS rule. The claim's `DOES-NOT-HOLD` branch ("every `MS-001` gate
  must re-scope to a stub or the milestone cannot be gated at all") does **not** fire, so no re-scope
  is owed and `US-009`'s normative choice stands unamended. No separable normative choice surfaced, so
  nothing here reaches the User Authority Boundary and no question is askable-or-owed by this gate.
- Scope guard kept explicit: this settles reachability-and-surface only. `US-009` claim 1 (host runtime
  state: nothing listening locally on 30000/30001) is `Verification: assumptions` — stale by nature and
  not this milestone's to arbitrate.

`FK-008` claims 1–2, the only other PENDING pair in the ledger, were arbitrated **HOLDS** at
[056-yield-phase](056-yield-phase.md) for `MS-004`. **No owned claim remains `PENDING` or `UNCLEAR` in
this session's `decisions.md`.**

## Findings (detail)

**None.** Zero P1, zero P2, zero P3 across Checks 1-7.

For the record, the three things a reader might expect to be findings and that were inspected and
cleared rather than raised:

1. **`dump.py` is 316 lines against the plan's "(<300)" Green projection.** Not unowned growth: the
   plan names the cause in the same paragraph (`server.py` is coverage-omitted, so the timer loop body
   must live in `dump.py`), the projection carries its own warning that it "must not be quoted as
   current", `T002` itself landed at 284, and the overrun is tracked with the current figure at
   `QB:dump/file-size-yellow` → `roadmap.md` §RM-002 (module split / misplacement check, `DEFERRED
   (spans modules)`). Outside this milestone's scope by the scope rule.
2. **The documented writer now refuses symlinked/non-regular targets, where the base wrote through.**
   An addition on a hostile-input corner, authored inside this milestone's own slice with fail-first
   tests, and already classified **Aligned** against `OI:…/output_layout` and
   `PI:…/documented_format_is_retained` at [031-02-yield](031-02-yield.md). `MS-001`'s gate re-proves
   the containment from outside the process.
3. **`TC-107`'s "at `2N` writes" clause is not instrumented in `TC-107` itself.** The VP's authoritative
   `Evidence needed` for `VP-T002-default-encoding` is "`DUMP_DIR/<sid>/` listing + file contents, with
   no `--dump-format` passed" — both asserted, and the `2N` conjunct is instrumented on the *same*
   writer by `TC-108` at the *same* `N` (`write_ops=6 == 2N`, per-path decomposition). The distinction
   `TC-107` exists to gate — that the *default* is not the collapsed layout (`G-008` witness W1) — is
   asserted directly, and no other test fails for that reason.

## Verdict (detail)

### **SEMANTIC-APPROVE** — `MS-001` / `QAP-MS001`, zero findings

- Checks 1-7 all **PASS**. No rejection, so the `Forward->Yield Retries` counter for the milestone's
  root tasks is **not** incremented and the retry cap is not approached.
- No Codex findings were supplied with this dispatch; at a full boundary the orchestrator runs the
  review in parallel and merges. Nothing here needs re-threading for it: no production source was
  touched by this pass.
- **Next, per the pipeline:** assumptions recheck and then **Yield Red**. This gate does **not** append
  a `Delivered` chronicle entry, publish child completion, or mark the phase accepted —
  Yield Red owns those after its final CLEAN verdict, and no commit here is classified as squashable.
- Session state after this pass: all four milestones now have a certified integration verdict
  (`MS-001` here, `MS-002` at [069](069-yield-phase.md), `MS-003` by its existing-suite gate,
  `MS-004` at [056](056-yield-phase.md)), and the session's last owned `PENDING` claim is closed.

---

**CODIFY Session ID**: `02.00-2026-09-13-message-dump-bulk-flush--9bf7afff5573`
**Session Directory**: `.sessions/codify/02.00-2026-09-13-message-dump-bulk-flush--9bf7afff5573/`
**Snapshot certified**: HEAD `c7420ce05c921dafcc4720186b186608f7bf8d86`, target object
`72e3a1c6861bf0d8de6317adf52289e5f6cfa4e1` (ancestor, 190 ahead / 0 behind). No production or test
source was touched by this pass (`git diff --exit-code HEAD -- buster_ripper tests
serve_buster_ripper.sh pyproject.toml README.md` → clean); the only writes are this report,
`evidence/ms001-gate-and-upstream-adjudication.txt`, and the four session ledgers.

## Integration Verification Summary

- **Check 1 (Caller Verification): PASS** — `flush_all` has 3 production callers, `poll_flush` 1
  (spawned in `on_startup`), `dump_turn` 1 (the interception site, inside the swallow-all), the
  selector global 1 write / 2 read sites, the `--dump-format` callback bound at `main.py:82`, and the
  serve-script path wired at `serve_buster_ripper.sh:29,41`. All 25 private `dump.py` helpers have ≥ 1
  production call site — no orphan from the capacity extraction. No removed callback lacks a replacement.
- **Check 2 (Migration Parity): PASS** — 13 side effects enumerated from
  `buster_ripper.py:_dump_turn` at the base object; every one has a counterpart in the retained
  documented writer (file names, encodings, `json.dumps` args, diff `fromfile`/`tofile`/`lineterm`,
  both summary-line formats, index append, `turn`/`prev_text` advance, log wording). **Zero omissions.**
  The one new refusal on that writer (symlink/non-regular targets) is an addition, documented at
  031-02 and gated here. T005's README change is `+53/−0`.
- **Check 3 (Data Flow Completeness): PASS** — both paths traced producer→consumer by reading code;
  `app["dump_flush_task"]` written and cancelled under the identical key; teardown order
  cancel → flush → close session confirmed; no `await` between append and threshold check; failed
  flushes retain rather than discard.
- **Check 4 (Consumer Contract Tracing): PASS** — the closed format set, `state.fmt`, the summary-line
  contract consumed by the README-documented shell reader (gated by a sensitivity test that proves it
  can fail), and the instrumented write report's schema all match their consumers. The coalesced
  one-write-per-flush merge preserves per-record turn identity and never merges across sessions.
- **Check 5 (Plan Fulfillment): PASS** — every ADDED clause of T001/T002/T005 present, and the
  **removal-side** tests passed in both directions: the write sites T002 was forbidden to delete are
  alive (so `US-012` was honored), `dump_turn`'s inline record construction is relocated rather than
  duplicated (dispatch is 11 lines, `dump.py` went 109 → 284 at T002), the dropped `REMOVED` clauses
  were never authored, no tool section exists, and no configuration can reach `2N + F`. `MS-001`'s
  six milestone conjuncts each map to an artifact produced at this boundary.
- **Check 6 (QA Milestone Closure): PASS** — `QAP-MS001` re-executed independently: **10/10**, four
  `VP-*` handles `pass`, contract fidelity verified clause-by-clause, the `022` directive ruled
  honored-as-written with its premise expired (no plan change owed), upstream reachability proven
  first-party, and the intermediate sweep clean (070's four promoted helpers single-defined; nothing to
  delete).
- **Check 7 (Integration Suite): PASS** — scope derived from the diff before running: one changed
  package plus both test trees; excluded only `buster_ripper_legacy.py` (plan-excluded dead legacy) and
  non-code docs. Ran the full hermetic suite once (**407 @ 95.75%**, zero failures → no bisect owed)
  and the entire `tests/qa/` tree (**31/31**: `MS-001` 10, `MS-002` 15, `MS-004` 6) on this snapshot —
  the latter two re-run rather than inherited because 070 changed the shared `harness.py`. Shared-run
  rule applied to Checks 6 and 7; no scope reduction after any failure.

## Findings

**Zero.** No P1, no P2, no P3. No Codex findings were supplied with this dispatch.

## Verdict

**SEMANTIC-APPROVE** — `MS-001` / `QAP-MS001`.

Route: **assumptions recheck → Yield Red**. This pass deliberately did **not** append a `Delivered`
chronicle entry, publish child completion, mark the phase accepted, or move `HEAD`; no commit here is
classified as squashable. The `Forward->Yield Retries` counter is untouched (no rejection issued).
The session's last owned `PENDING` claim (`US-009` c2) is now **HOLDS** with `Consequence: none`, so
`decisions.md` carries no PENDING row.
