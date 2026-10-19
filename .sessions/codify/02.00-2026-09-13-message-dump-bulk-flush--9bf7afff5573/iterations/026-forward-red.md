# 026 — forward-red (SCOPED RE-RUN)

## Orchestrator Prompt (verbatim)

> Session ID: 02.00-2026-09-13-message-dump-bulk-flush--9bf7afff5573
>
> Scoped re-dispatch — the one permitted re-run, per your own routing note in iterations/024-forward-red.md. Scope it ONLY to the rows whose artifacts were repaired. Do not re-run the full pass.
>
> The eight AMBIGUOUS rows were closed by decide in iterations/025-decide.md and the repairs applied by interview in round 19, committed as 5864051 (Checkpoint 52). Repaired rows and what changed:
> - G-1 — G-001's `Met when` gained a byte clause; VP-T002-write-count-reduced / TC-105 / T004 gained a written-bytes assertion. MS-002's gate gained a byte-volume observable.
> - G-2 — "(default)" deleted from G-006's change-focused arm.
> - G-3 — G-008(b) / MS-002 / T004 / TC-105 name the durable MS-002 comparison record.
> - G-4 — G-003's failure witness narrowed with an explicit carve-out for the settled append-time flush.
> - G-5 — CLOSED ON THE RECORDED FLOOR, not repaired. The 50% floor and `F = N` are deliberate; gating the shipped trigger values would convert Forward discretion into a plan commitment. Treat this as a disposition to verify, not a repair.
> - T-1 — the append condition moved to T002.
> - T-2 — `norecursedirs` named; the "or equivalent" list deleted. Note: `norecursedirs`'s ini value replaces pytest's defaults, so the repair must carry `.*`, `build`, `dist`, `node_modules`, `venv`, `*.egg`, `_darcs`, `CVS`, `{arch}` forward.
> - T-3 — "real-signal (subprocess) cases" deleted from T004; that scenario is TC-110's in tests/qa/.
>
> One residual decide raised and did not resolve: qa-plan.md:48's QA Gate Matrix `Evidence` cell still reads "the paired counts, recorded side by side (`G-008`(b))" — the phrasing that supported G-3's misreading. Decide judges it harmless now that MS-002's row names the durable comparison record, so it does not reopen G-3. You are the arbiter for that cell: say whether it still admits the misreading or not.
>
> Report which rows are genuinely closed and which still admit a misreading. If any still admit one, name it precisely. The assumption gate is unblocked unless you find a surviving ambiguity.

## Scope and method

Clean-room held: read only `plan-summary.md`, `handoff-context.md`, `qa-plan.md`, `roadmap.md` and the code
(`dump.py`, `config.py`, `main.py`, `server.py`, `pyproject.toml`, `serve_buster_ripper.sh`) plus the installed
pytest source. No decide/interview/collect iteration report was read. No full pass was re-run: the attack is scoped
to the eight rows named above and the one cell put to arbitration.

Artifact state at the time of this pass differs from the dispatch premise. HEAD moved during the pass:

- `5864051` (Checkpoint 52) — the repair batch, where the dispatch's `qa-plan.md:48` quote was accurate.
- `6700914` (Checkpoint 53, "Align the TC-105 matrix cell with its scenario contract") — landed while this
  dispatch was in flight. It edited exactly that cell plus `handoff-context.md` and `iterations/020-interview.md`.

`plan-summary.md` was unchanged since `5864051` for the whole first pass. Two repairs landed while this dispatch was in
flight, and **both are verified below against the text on disk, not against the version this pass first read**:

- `6700914` (Checkpoint 53) — `qa-plan.md`'s `TC-105` gate-matrix `Evidence` cell, committed.
- `R15` — `plan-summary.md:240`, `VP-T002-write-count-reduced`'s `Observable behavior`, applied uncommitted at
  20:23 (HEAD still `6700914`; `git diff HEAD` shows exactly that one line changed in `plan-summary.md`). `decide-red`
  retracted its earlier disposition: `R6`'s authorization did name that sentence and reworded its `T004` copy, and this
  row holds the same sentence verbatim apart from the `(018)` tag — so this is `R6` reaching its second surface, not a
  widening.

**Net effect on this pass's verdict: the one AMBIGUOUS row is closed by R15, so the pass returns `0 AMBIGUOUS`.** The
row's original verdict and the reasoning that produced it are kept below, marked as superseded, because the finding was
real against the text it was raised on.

## Verdict table

| # | Surface (`G-*` or task) | Plausible misreading | Excluded by | Verdict |
|---|---|---|---|---|
| 1 | `G-001` (G-1) | The count clause alone is the goal, so an artifact that rewrites the whole file per flush satisfies it: write count `= F`, every test green, while bytes written grow `O(N²)` (witness B) | `plan-summary.md:66` `Met when` — "the artifact is append-only: each flush's write carries only that flush's pending batch, so the session's bytes written grow with `F`, not `N²` (witness B is excluded by this clause, not only by the count)"; carried into `:240` (`Observable behavior` + `Evidence needed`: "per-write payload bytes"), `qa-plan.md:48`/`:117`, `:183` (T004) and `:220` (MS-002) | **Excluded — closed.** The clause is falsifiable by an observable the gate now collects (per-write payload bytes), not by prose alone |
| 2 | `G-006` (G-2) | *change-focused* is the default encoding, so a reader may take the default run to produce the collapsed file and the `documented` arm to be the non-default one | `plan-summary.md:71` — the arm now reads "*change-focused*: the clause holds **as written**", with no "(default)"; `:115` and `:153` fix `config.DUMP_FORMAT = "documented"`; `:27`/`:98` record the retraction explicitly; grep for a `change-focused`+`default` co-occurrence returns only retractions and the `documented` default | **Excluded — closed** |
| 3 | `G-008`(b) (G-3) | The paired in-test runs (documented vs collapsed at the same N) **are** the comparison `G-008`(b) requires, so Forward builds the paired assertion and never authors the durable record; the `roadmap.md` switch trigger is then inputless | `plan-summary.md:73` `Met when` (b) — "as a durable, human-readable record … **not merely an in-test assertion**"; `:220` (MS-002), `:184` (T004), `:240` (the VP row itself, after `R15`), `qa-plan.md:32`/`:48`/`:113`/`:117`, `roadmap.md:56-59` all name the record | **Excluded — closed.** Raised as AMBIGUOUS on the pre-`R15` text of `:240` (see below); the repaired row now names the record, so no surface carries the reading |
| 4 | `G-003` (G-4) | The settled append-time flush's latency is itself a `G-003` failure, so Forward must bound or offload it | `plan-summary.md:68` — the witness is breakage only ("a 5xx, a dropped connection, or a body the proxy altered"), the carve-out names the flush "by design (`T003`, no `await`)", and the boundary states "**Not a latency bound** … no clause of this plan bounds request latency" | **Excluded — closed.** Latency is not a failure; a latency so large it *breaks* a request is a witness on its own terms, which is consistent rather than contradictory |
| 5 | `G-001` floor (G-5) | The 50% floor and `F = N` are a weak gate, so Forward should add a flush-cadence or trigger-value assertion to strengthen it | `plan-summary.md:66` boundary — "**The 50% floor is this goal's count axis and `F = N` is a pass by design** — the goal gates the *reduction*, not the flush cadence; the buffering cadence is gated by `VP-T001-writes-deferred` and by `TC-100`'s deferral assertion"; `:51-55` states the floor and `F ≤ N` as arithmetic. No VP asserts a trigger value (`VP-T003-interval-flush` and `VP-T002-volume-flush` *inject* their parameters) | **Excluded — closed on the recorded floor.** The disposition is verified, not merely asserted: nothing in the task list or VP matrix gates `DUMP_FLUSH_INTERVAL`/`DUMP_FLUSH_BYTES`'s values, so `FK-A`/`FK-B`'s Forward discretion is intact |
| 6 | `T001`/`T002` (T-1) | The append is format-conditioned in `T001`'s own increment, so `T001` must reference `config.DUMP_FORMAT`, which `T002` owns — the dependency cycle T-1 was raised about | `plan-summary.md:146` — "**in this increment the append is unconditional** — no trigger exists yet, so nothing is drained; **`T002` adds the condition that makes it collapsed-path-only**"; `:156` carries the condition in `T002`'s ADDED list; `:169` gives `T002`'s dependency as `T001` | **Closed as a task-entry conflict** — the two task entries no longer disagree. One summary-surface restatement survives: see footnote F-1 |
| 7 | `T002` (`pyproject.toml` clause) (T-2) | The exclusion can be delivered by any of the three named mechanisms, so an `addopts` marker or a directory-level skip is acceptable | `plan-summary.md:165` — "**`norecursedirs`**, whose ini value **replaces** pytest's default list (`_pytest/main.py:213-222`), so the configured value must retain the default patterns (`.*`, `build`, `dist`, `node_modules`, `venv`, `*.egg`, `_darcs`, `CVS`, `{arch}`) alongside `qa`"; the "or equivalent" list is deleted (grep → zero hits). Verified in the installed pytest: the ini value is not merged with `addini`'s default (`main.py:213-222`), and `fnmatch_ex` is applied to the directory path (`:456-458`), so `qa` matches `tests/qa` | **Excluded — closed.** The repair is actionable as written and the carry-forward requirement is correct |
| 8 | `T004` (T-3) | The real-signal (subprocess) shutdown scenario belongs in the hermetic `tests/test_dump.py` | `plan-summary.md:183` — the claim is deleted (grep for `real-signal (subprocess) cases` → zero hits) and replaced by "the real-signal shutdown scenario is `TC-110`'s, under `tests/qa/` at `MS-002`"; `qa-plan.md:53` (TC-110) and `:167` carry it there | **Excluded — closed** |
| 9 | `qa-plan.md:48` `Evidence` cell (arbiter call) | The paired counts recorded side by side are the `G-008`(b) comparison, so the durable record need not be authored | The cell now reads "the instrumented write counter (call count and payload bytes) + two directory listings, recorded as a pair **+ the `MS-002` Yield Phase's comparison record (a durable, human-readable artifact)**" — the record is named as a distinct required element | **Excluded — closed.** Repaired at `6700914` while this dispatch was in flight; the dispatch's quote was accurate at `5864051` and is stale |

## G-3 — the AMBIGUOUS row, and its closure by R15

**As raised (pre-`R15` text).** `plan-summary.md:240`, `VP-T002-write-count-reduced`'s `Observable behavior`, ended:

> **The pair of runs (documented vs collapsed at the same N) IS the comparison `G-008`(b) requires (018)**

That is the pre-repair phrasing, and it contradicted the repaired `G-008`(b) one row earlier in the same document: the
comparison is "**a durable, human-readable record … not merely an in-test assertion**" (`plan-summary.md:73`). This is the
**restroking** family — the same referent (the `G-008`(b) comparison) rendered twice, a concise form and a detailed form,
and the two had drifted.

Why it was material, not a nitpick:

- The row is a **validation point**, and a VP row is what a scenario author reads first. The scenario contract and the
  gate-matrix cell for `TC-105` were both re-pointed at the record (R3, and R3's second surface at `6700914`); this row was
  the one that still said the pair alone suffices.
- The **roadmap switch trigger consumes the record** (`roadmap.md:56-59`, `US-012`). A Forward agent reading this row
  builds the paired assertion, sees `TC-105` pass, and produces no durable record — leaving the switch entry with no input.
  That is `G-008` witness W3's shape ("'better' is asserted, never measured") reached with every test green.
- It conflicted with a plan element, not with a preference: `G-008`(b)'s `Met when`, `MS-002`'s gate, `T004`'s ADDED list
  and three `qa-plan.md` surfaces all say the record is required.

`R2` had landed the byte clause and `per-write payload bytes` on this row but not the comparison clause, so the repair reached
one clause of the row and not the other — which is why the row was inside this pass's scope.

**Verified after `R15` (working tree, 20:23, uncommitted at HEAD `6700914`).** The sentence now reads:

> **The pair of runs (documented vs collapsed at the same N) is the measurement `G-008`(b) requires (018), and the `MS-002` Yield Phase records it as the durable comparison record**

Checks run against the text on disk, not from the dispatch:

- The `(018)` tag is kept, and the wording mirrors `T004`'s already-repaired sentence at `:184` — one blueprint element,
  with the prose pointed at the record rather than both surfaces reworded into a new agreement.
- `grep "IS the comparison"` across `plan-summary.md`, `qa-plan.md` and `roadmap.md` → **zero hits**.
- `grep "paired counts"` → **zero hits**.
- The only remaining `recorded side by side` hits are the two repaired `MS-002` rows (`plan-summary.md:220`, `qa-plan.md:32`),
  and in both the phrase's object is explicitly the record.
- `git diff HEAD` for `plan-summary.md` is exactly that one line — the other seven rows' artifacts were not touched by `R15`.

**Verdict: closed.** No surface admits the G-3 misreading.

## Footnotes (Forward discretion, yield-reviewable)

- **F-1 — the F-21 restatement on two summary surfaces.** `plan-summary.md:21` and `handoff-context.md:197` both read
  "**F-21** `T001`'s append is format-conditioned", while `T001`'s own entry (`:146`) says the append is **unconditional in
  this increment** and that `T002` adds the condition. Both readings reconcile to the same final code (the append is conditioned
  once `T002` lands), so the gate returns no on all four questions: no user-visible difference, no invariant or Anchor Map
  binding touched, one task boundary to correct, and the same artifact either way — hence a footnote, not an AMBIGUOUS row.
  **Recommended reading**: `T001`'s entry governs — implement the append unconditionally in `T001`; the condition lands with
  `T002`. Not blocking; if Decide prefers symmetry with G-3's surviving surface, the fix is the same one-line class.

## Rows checked and found genuinely closed

All nine surfaces are closed, none conditionally:

| Row | Closed by |
|---|---|
| G-1 | the append-only byte clause, carried into the VP's `Observable behavior` + `Evidence needed`, `TC-105` (both surfaces), `T004` and `MS-002` |
| G-2 | "(default)" absent from `G-006`'s change-forward arm; the default is `documented` everywhere it is stated |
| G-3 | `G-008`(b), `MS-002` ×2, `T004`, `TC-105` ×2, `roadmap.md`'s trigger — **and `plan-summary.md:240` by `R15`** |
| G-4 | breakage-only failure witness with the append-time-flush carve-out and the no-latency-bound boundary |
| G-5 | the recorded floor, verified as a disposition: no VP or task gates the trigger values, so `FK-A`/`FK-B` discretion is intact |
| T-1 | the append condition lives in `T002`'s ADDED list only; `T001`'s entry states its own increment is unconditional |
| T-2 | `norecursedirs` named, "or equivalent" deleted, carry-forward of pytest's default patterns stated and correct |
| T-3 | "real-signal (subprocess) cases" deleted and re-attributed to `TC-110` under `tests/qa/` |
| `qa-plan.md:48` (arbiter cell) | `6700914` — the durable record is named as a distinct required element |

**No surface still admits any of the eight misreadings.** The sole footnote (F-1) is non-blocking and does not gate forward.

## Routing

`0 AMBIGUOUS` → the assumption gate is authorized; Forward is not yet authorized. The one footnote (F-1) is
yield-reviewable and does not block.
