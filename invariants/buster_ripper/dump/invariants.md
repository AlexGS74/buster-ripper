# Invariants — `buster_ripper.dump`

Registry file for the dump cluster. Created 2026-09-16 by session
`02.00-2026-09-13-message-dump-bulk-flush--9bf7afff5573` (interview round 18, final-approval gate).
This repository had no invariant registry before this session; the entries below are the session's approved
set, elevated by handle-keyed intersection from `plan-summary.md`. Nothing was copied over an existing file and
no pre-existing entry was deleted.

## Entities

Entity: E:T001/dump/dump-caller
Binding: buster_ripper/routes/messages.py:handle_messages
Gloss: The one `/v1/messages` interception site that calls the dump entry point; it owns the swallow-all contract that keeps a dump failure from failing a request.
Confusable: commonly misread as owning dump logic — it needs no edit in this session, but the record-building path it calls grows, so the swallow-all contract now covers more code — session Anchor Map
Source: session=02.00-2026-09-13-message-dump-bulk-flush--9bf7afff5573 task=T001

---

Entity: E:T001/dump/session-state
Binding: buster_ripper/dump.py:SessionState
Gloss: The per-session dump state keyed by session id; it gains the pending record list and the pending byte count, and it advances the turn index and previous rendered text at append time.
Confusable: commonly misread as needing a separate buffer type — the pending list is a field on this existing class, not a new class — session Anchor Map
Source: session=02.00-2026-09-13-message-dump-bulk-flush--9bf7afff5573 task=T001

---

Entity: E:T002/dump/dump-dir-config
Binding: buster_ripper/config.py:DUMP_DIR
Gloss: The dump mode switch — `None` means no dump is written; it is the guard the dump entry point must honour and the flush must tolerate.
Confusable: commonly misread as the `--upstream` default, which is declared twice in this repository (`config.py:13` vs `main.py:13`, the latter overwritten unconditionally at `main.py:70`) — quality-backlog finding 3
Source: session=02.00-2026-09-13-message-dump-bulk-flush--9bf7afff5573 task=T002

---

Entity: E:T002/dump/dump-entry-point
Binding: buster_ripper/dump.py:dump_turn
Gloss: The dump write entry point, called once per intercepted turn; it owns the format branch that selects which writer runs for a turn.
Confusable: commonly misread as owning the collapsed artifact's name and append target — those belong to the collapsed-artifact referent, not to this one — session Anchor Map (decide 015)
Source: session=02.00-2026-09-13-message-dump-bulk-flush--9bf7afff5573 task=T002

---

Entity: E:T002/dump/bulk-flush
Binding: buster_ripper/dump.py:flush_all
Gloss: The collapsed-path drain: for each session with pending records, appends the whole batch to that session's artifact in one write-bearing operation, then clears the buffer; no-ops when DUMP_DIR is None or nothing is pending; never raises and never touches turn/prev_text. No caller until T003's triggers land.
Confusable: commonly misread as owning the artifact's name — it appends to the path the collapsed-artifact referent builds; it is also not the documented path's writer (nothing is ever pending there)
Source: session=02.00-2026-09-13-message-dump-bulk-flush--9bf7afff5573 task=T002 (registered at T002 Yield; symbol landed in Checkpoint 64)

---

Entity: E:T002/dump/dump-format-selector
Binding: buster_ripper/config.py:DUMP_FORMAT
Gloss: The injectability seam for the dump's output shape: `DUMP_FORMAT` holds the current value from the closed `DUMP_FORMATS` three-value set (`documented`, `change-focused`, `self-contained`) with `documented` as the default, assigned once at parse time through the `--dump-format` option's own callback.
Confusable: commonly misread as an encoding-only selector — a value names a complete output shape including the layout axis (decide 019); the R2 candidate two-value set must not be authored
Source: session=02.00-2026-09-13-message-dump-bulk-flush--9bf7afff5573 task=T002 (registered at T002 Yield; symbol landed in Checkpoint 64)

---

Entity: E:T002/dump/dump-summary-line
Binding: buster_ripper/dump.py
Gloss: The per-record summary-line construction, format-independent: `_full_summary` supplies turn 0's FULL form and `_diff_summary` supplies turn ≥ 1's +added/-removed form — identical constructions to the documented writer's summary lines.
Confusable: commonly misread as format-specific — both collapsed formats and the documented writer emit the same summary forms; only the record payload differs
Source: session=02.00-2026-09-13-message-dump-bulk-flush--9bf7afff5573 task=T002 (registered at T002 Yield; symbols gained their own names in T002's capacity extraction)

---

Entity: E:T002/dump/collapsed-dump-artifact
Binding: buster_ripper/dump.py:_artifact_path
Gloss: The collapsed artifact's identity: one append-only file per (session id, format) at `DUMP_DIR/<sid>/<format>.log`, named by exactly the configured collapsed value; the documented value produces the three documented files and no `.log`.
Confusable: commonly misread as owning the format branch (that is the dump entry point's) or the summary lines (encoding-independent); the append target itself is the flush referent's write site
Source: session=02.00-2026-09-13-message-dump-bulk-flush--9bf7afff5573 task=T002 (registered at T002 Yield; symbol landed in Checkpoint 64)

---

Entity: E:T003/dump/flush-timer
Binding: buster_ripper/dump.py:poll_flush
Gloss: The periodic flush drain for the collapsed formats: an endless flush-then-sleep coroutine (`flush_all()`, then `await asyncio.sleep(config.DUMP_FLUSH_INTERVAL)`) with per-iteration exceptions caught and logged so one failed tick kills no later tick; cancellation lands at the sleep await. Spawned at startup (`spawn_dump_flush` → `app["dump_flush_task"]`), cancel-and-awaited in `on_cleanup` ahead of the shutdown flush. No-ops on the documented path (nothing is ever pending there).
Confusable: commonly misread as a scheduler/timer facility (asyncio has none — `CL:python/asyncio-periodic-timer`); also misread as kin of `compaction.poll_kv_cache` — that is the periodic *metrics* poll for KV cache, the pattern peer, not a flush owner (RB-002)
Source: session=02.00-2026-09-13-message-dump-bulk-flush--9bf7afff5573 task=T003 (registered at T003 Yield; symbol landed in Checkpoint 66)

---

Entity: E:T003/dump/flush-trigger-parameters
Binding: buster_ripper/config.py
Gloss: The injectability seam for the flush triggers: `DUMP_FLUSH_INTERVAL` and `--dump-flush-interval` set interval seconds between periodic flushes (default 10, `KV_POLL_INTERVAL` peer); `DUMP_FLUSH_BYTES` and `--dump-flush-bytes` set pending rendered-text bytes before a volume flush (default 4 MiB, `FK-B`'s single-digit MiB). Each is assigned once at parse time through its option's own callback so `main()` gains no parameter; both parameterize only the collapsed path's machinery.
Confusable: commonly misread as bounds or validators — neither flag validates positivity (peer-consistent, plan-silent; quality-backlog QB:dump-flush/trigger-value-validation); also misread as constraining the documented path, where nothing buffers and both triggers are vacuous
Source: session=02.00-2026-09-13-message-dump-bulk-flush--9bf7afff5573 task=T003 (registered at T003 Yield; symbols landed in Checkpoint 66)

---

## Overrides (existing invariants in the session's contradiction surface)

Each block records the provenance of an existing dump property this session refines. The session's own
direction token was `refine` (corrected from `override` by organize 018: nothing is displaced, because the
documented property survives whenever the documented format is selected). `Decision: preserve` is the Speeckle
enum value for exactly that — the old invariant remains in force, scoped — and the refinement itself is in
`Reason:` and in the replacement statement. Per the elevation rule for a handle the registry never held, these are
**provenance blocks, not new active entries**: no `EI:` statement block is invented below.

Override: OI:buster_ripper.dump/dump_turn/output_layout
Old: EI:buster_ripper.dump/dump_turn/output_layout
New: PI:T002/dump_format/format_is_per_artifact
Decision: preserve
Reason: Session direction `refine` (organize 018, F-14). The documented property survives and is only **scoped per selected format**: with the documented format selected — or nothing configured — the session dir still holds `turn_000.json`, `turn_{N:03d}.diff` and `index.txt` unchanged at `2N` write-bearing operations; when a collapsed format is selected the dir instead holds one `<format>.log` appended once per non-empty flush. The default is `documented`, so "the documented layout is retained" does not mean "a collapsed format may be the default". The removal half of the original override was refused by the user (US-012) as premature; the refusal is discharged by the rewritten statement.
Approved: yes
Session: 02.00-2026-09-13-message-dump-bulk-flush--9bf7afff5573
Source: session=02.00-2026-09-13-message-dump-bulk-flush--9bf7afff5573, plan-summary.md §Approved Override Register entry 1; user ratified round 17 (US-014), user statement US-012 (2026-09-16)
Verification:
  - TC-108 / VP-T002-documented-format-preserved (documented format byte-unchanged at 2N writes)
  - TC-107 / VP-T002-default-encoding (no format configured produces the documented layout)
  - TC-102 / VP-T002-collapsed-artifact (one `<format>.log`, no documented files, on the collapsed path)

---

Override: OI:buster_ripper.dump/dump_turn/sequential_diff_chain
Old: EI:buster_ripper.dump/dump_turn/sequential_diff_chain
New: PI:T002/dump_format/one_write_path_per_turn
Decision: preserve
Reason: Session direction `refine`. The per-turn-diff property survives; the write-time clause changes. The turn index and the previous rendered text advance at **append** time, never at flush time, so a flush never resets the chain. For a collapsed format each turn's record derives from the previous turn's rendered text in arrival order; the append-time sequencing is format-independent. The clause reads per selected format — the documented format's chain still advances when its turn is written and its diff files are unchanged.
Approved: yes
Session: 02.00-2026-09-13-message-dump-bulk-flush--9bf7afff5573
Source: session=02.00-2026-09-13-message-dump-bulk-flush--9bf7afff5573, plan-summary.md §Approved Override Register entry 2; user ratified round 17 (US-014)
Verification:
  - TC-103 / VP-T002-change-readable (per format)
  - TC-102 / VP-T002-collapsed-artifact (arrival order)

---

Override: OI:buster_ripper.dump/dump_turn/turn_zero_is_full_body
Old: EI:buster_ripper.dump/dump_turn/turn_zero_is_full_body
New: PI:T002/dump_format/format_is_per_artifact
Decision: preserve
Reason: Session direction `refine`. Turn 0's full normalized JSON body **and** its rendered rendering are present in the session artifact once flushed, never only a diff — in every shipped collapsed format. "Full body" means both the raw normalized JSON body and its rendering. The documented format's `turn_000.json` already satisfies this verbatim and is preserved, not refined.
Approved: yes
Session: 02.00-2026-09-13-message-dump-bulk-flush--9bf7afff5573
Source: session=02.00-2026-09-13-message-dump-bulk-flush--9bf7afff5573, plan-summary.md §Approved Override Register entry 3; user ratified round 17 (US-014)
Verification:
  - TC-104 / VP-T002-turn0-available (turn 0's raw body and rendering recoverable, per format)

---

## Planned Invariants

The family was renamed `PI:T002/artifact_encoding/*` → `PI:T002/dump_format/*` by decide 019: the set contains
`documented`, which is a layout rather than an encoding, and the misnomer would lead an implementer to build an
encoding-only set. Nothing approved was renamed — none of the earlier entries had been confirmed. `encoding` survives
only as the sub-axis a collapsed record's contents vary on.

**Statement lint applied at elevation.** Six statements were rephrased to their observable property, dropping location and
prescription clauses that a correct reimplementation could violate (`config.DUMP_FORMAT`/`config.DUMP_FORMATS`/`main.py`
references now live in the Entity layer, and "no increment removes it before…" — a prescription about future work — is
recorded in the override's `Reason:` instead). No rephrasing changes what the invariant requires.

Invariant: PI:T002/dump_format/format_is_per_artifact
Kind: structural
Strength: protected
Status: planned
Scope: buster_ripper.dump/dump_turn
Statement: One artifact carries exactly one configured dump format. The format is chosen once per process and never mutated, and a collapsed artifact's filename states the configured value.
Why: The format is the artifact's identity key together with the session id; a directory holding two formats would make the listing self-describing only by accident.
Source: session=02.00-2026-09-13-message-dump-bulk-flush--9bf7afff5573, plan-summary.md §Planned invariants; user confirmed as written round 17 (US-014)
Witness: TC-102 (VP-T002-collapsed-artifact)
Evidence:
  - buster_ripper/dump.py
  - .sessions/codify/02.00-2026-09-13-message-dump-bulk-flush--9bf7afff5573/qa-plan.md

---

Invariant: PI:T002/dump_format/format_does_not_change_write_count
Kind: behavioral
Strength: protected
Status: planned
Scope: buster_ripper.dump/dump_turn
Statement: The configured format changes what a record contains and which writer runs, never how many write-bearing operations a session performs for a given writer. A session of N turns flushed F times with a collapsed format selected costs F write-bearing operations, against 2N for the same N with the documented format selected.
Why: This is the session's headline reduction (G-001); a format that changed the write count would make the reduction a property of the encoding rather than of the buffering.
Source: session=02.00-2026-09-13-message-dump-bulk-flush--9bf7afff5573, plan-summary.md §Planned invariants; user confirmed as written round 17 (US-014)
Witness: TC-105 (VP-T002-write-count-reduced) — run per collapsed format
Evidence:
  - buster_ripper/dump.py:dump_turn
  - .sessions/codify/02.00-2026-09-13-message-dump-bulk-flush--9bf7afff5573/qa-plan.md

---

Invariant: PI:T002/dump_format/default_format_is_documented
Kind: behavioral
Strength: protected
Status: planned
Scope: buster_ripper.dump/dump_turn
Statement: With no format configured, the output is the documented per-turn layout at 2N write-bearing operations and no collapsed artifact is created. The collapsed formats are opt-in.
Why: The user's own requirement — old behavior stays unaffected until a new one is proved better and judged worth switching to. This is why the session's write reduction is opt-in, a consequence disclosed and accepted rather than a surprise.
Source: session=02.00-2026-09-13-message-dump-bulk-flush--9bf7afff5573, plan-summary.md §Planned invariants; user confirmed as written round 17 (US-014); grounded in US-012
Witness: TC-107 (VP-T002-default-encoding)
Evidence:
  - buster_ripper/config.py
  - .sessions/codify/02.00-2026-09-13-message-dump-bulk-flush--9bf7afff5573/qa-plan.md

---

Invariant: PI:T002/dump_format/format_set_is_closed
Kind: compatibility
Strength: normal
Status: planned
Scope: buster_ripper.dump/dump_turn
Statement: The accepted format set is exactly three values — documented, change-focused, self-contained. A configured value outside that set is a hard failure, never a silent fallback.
Why: Each shipped format becomes a documented format, so the size of the set is the irreversible part of this decision.
Source: session=02.00-2026-09-13-message-dump-bulk-flush--9bf7afff5573, plan-summary.md §Planned invariants; user confirmed as written round 17 (US-014)
Witness: tests/test_dump.py (T004 — the unknown-format guard test); the guard itself is T002's done-when
Evidence:
  - buster_ripper/dump.py:dump_turn
  - .sessions/codify/02.00-2026-09-13-message-dump-bulk-flush--9bf7afff5573/plan-summary.md

---

Invariant: PI:T002/dump_format/one_write_path_per_turn
Kind: structural
Strength: protected
Status: planned
Scope: buster_ripper.dump/dump_turn
Statement: For any configuration, a session of N turns performs either 2N or F write-bearing operations — never 2N + F.
Why: Derived constraint (F-13): retaining the documented writer while also writing the collapsed artifact yields 2N + F, which is worse than today while every test still passes. This is what forces the format branch to select which writer runs.
Source: session=02.00-2026-09-13-message-dump-bulk-flush--9bf7afff5573, plan-summary.md §Planned invariants; user confirmed as written round 17 (US-014)
Witness: TC-105 (VP-T002-write-count-reduced) paired with TC-109 (VP-T002-file-count-reduced)
Evidence:
  - buster_ripper/dump.py:dump_turn
  - .sessions/codify/02.00-2026-09-13-message-dump-bulk-flush--9bf7afff5573/qa-plan.md

---

Invariant: PI:T002/dump_format/documented_format_is_retained
Kind: compatibility
Strength: protected
Status: planned
Scope: buster_ripper.dump/dump_turn
Statement: The documented format remains a selectable value that produces today's three files with today's content and line format when selected, and remains the value produced when nothing is configured.
Why: The user called the removal of the documented writer premature and gated any displacement on a demonstration judged worth switching to. This is what keeps that gate real rather than rhetorical.
Source: session=02.00-2026-09-13-message-dump-bulk-flush--9bf7afff5573, plan-summary.md §Planned invariants; user confirmed as written round 17 (US-014); grounded in US-012
Witness: TC-108 (VP-T002-documented-format-preserved)
Evidence:
  - buster_ripper/dump.py:dump_turn
  - .sessions/codify/02.00-2026-09-13-message-dump-bulk-flush--9bf7afff5573/qa-plan.md

---

## Deferred at elevation (no registry entry yet)

The following Anchor Map rows name **authorized new symbols** that do not exist in the working tree yet. Per
the no-dangling-binding precedent, an `Entity:` whose `Binding:` cannot resolve is **deferred to the task's Yield**
rather than registered now — a dangling binding would fail the registry's existence check exactly as an unbound Anchor
Map row would. Each is registered at its task's Yield, once the symbol lands.

| Term | Planned binding | Register at |
|---|---|---|
| flush timer | `buster_ripper/dump.py` (the periodic loop body symbol, named by T003's extraction) | T003 Yield |
| flush trigger parameters | `buster_ripper/config.py:DUMP_FLUSH_INTERVAL`, `DUMP_FLUSH_BYTES` | T003 Yield |

Registered at T002 Yield (2026-09-22, Checkpoint 64 symbols landed): `bulk flush` → `E:T002/dump/bulk-flush`,
`dump format selector` → `E:T002/dump/dump-format-selector`, `dump summary line` → `E:T002/dump/dump-summary-line`,
`collapsed dump artifact` → `E:T002/dump/collapsed-dump-artifact`.

Registered at T003 Yield (2026-09-22, Checkpoint 66 symbols landed): `flush timer` → `E:T003/dump/flush-timer`
(`dump.py:poll_flush`), `flush trigger parameters` → `E:T003/dump/flush-trigger-parameters`
(`config.py:DUMP_FLUSH_INTERVAL`/`DUMP_FLUSH_BYTES` + `main.py`'s two callback-wired flags). Both rows above are
spent; the table retains them as evidence of the deferral→registration trail.

**Two Anchor Map rows are deliberately not entities.** `dump write buffer` names the same referent as
`E:T001/dump/session-state` (`pending`/`pending_bytes` are fields on that class, and the plan says so explicitly),
so a second entity for it would fail the uniqueness check — one handle per binding. `signal handling` names a native aiohttp
facility (`web_runner` → `GracefulExit` → app cleanup hooks), not a repository referent; it is a capability, not a WHAT
this registry can bind.
