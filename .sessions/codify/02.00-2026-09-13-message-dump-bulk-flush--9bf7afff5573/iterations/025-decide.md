# 025 — decide (RED-FINDINGS mode)

## Orchestrator Prompt (verbatim)

> Session ID: 02.00-2026-09-13-message-dump-bulk-flush--9bf7afff5573
>
> Red-findings mode. forward-red returned 8 AMBIGUOUS rows in iterations/024-forward-red.md; per the skill's routing the assumption gate is NOT authorized until these close. Read the full report for the row detail and resolve all eight: G-1, G-2, G-3, G-4, G-5, T-1, T-2, T-3.
>
> Also in the report: 5 rows excluded with citations and 4 discretion footnotes (F-1..F-4). The exclusions need no action; the footnotes are forward discretion and yield-reviewable — confirm or correct that disposition rather than treating them as open rows.
>
> Constraints:
> - US-013 is in force: the user delegated small, reversible, non-data-losing decisions to the agents. Do NOT surface a question for any row whose resolution is reversible in one commit and loses no data. Name explicitly, for any row you cannot close, the single fact that forces it to the user.
> - The plan is approved and stamped. Repair artifacts so they stop admitting the misreading; do not silently re-decide the design. Where a repair changes what a gate can observe (G-1 and T-3 look like they will), record that as a plan-surface change rather than an editorial fix.
> - G-2 is stale pre-019 text contradicting the settled `documented` default; treat it as a correction, not a new decision.
> - G-5 is flagged weakest — close it on the recorded floor or repair it, and say which.
>
> Update handoff-context.md and the affected artifacts, then report: which rows closed, how, and whether any row genuinely requires the user.

## Mode and scope

`Mode: red-findings`. Read: `plan-summary.md` (incl. §Anchor Map, §Goals and Constraints, §Goal Definition Matrix, §Task List, §Milestone Matrix, §Validation Point Matrix), `handoff-context.md`, `qa-plan.md`, `roadmap.md`, `decisions.md` (`US-012`/`US-013`/`US-014`), and the code the rows bind to (`buster_ripper/dump.py`, `config.py`, `server.py`, `routes/messages.py`, `serve_buster_ripper.sh`, `pyproject.toml`, `README.md:140-165`) plus the installed `pytest 9.0.2` source. No prior decide/organize/interview iteration report was reopened except where a row required the recorded authority (`019`'s closure gate, `022`'s repair).

**Ownership note (skill constraint, recorded deliberately).** `red-findings` mode states: *"Do not edit the approved plan or invariant registry in this mode… Interview applies approved plan repairs and is the sole writer for every invariant-registry change"*, and *"do not edit `plan-summary.md`"*. The dispatch asked for "the affected artifacts" to be updated. The plan-surface repairs below are therefore **authorized exactly, not applied**: they are handed to `interview` as one batch, and only `handoff-context.md` is updated by this pass. **Nothing here re-decides a design**: every repair restates a property the approved artifacts already carry, or deletes a claim the plan itself has already retracted.

## Closure packets

### Packet 1 — G-1

```text
CLOSURE: absorbed
Finding: G-1 (iterations/024-forward-red.md, goal-shape rows)
Required outcome: A `flush_all` that rewrites the whole artifact on every flush — G-001 witness B,
  write count F but bytes written O(N²) — must fail at least one declared gate, not merely contradict plan prose.
Authority:
  - G-001's own `Failure witness` cell names B as a failure: "**B**: the flush is one write, but it
    **rewrites** the whole artifact each time, so bytes written grow `O(N²)`" (plan-summary.md:66)
  - Anchor Map `collapsed dump artifact`: "**Append-only** — a rewrite-per-flush variant is excluded by
    `FK-002`'s presented option text and by `G-001` witness B" (plan-summary.md:121)
  - T002's ADDED `flush_all()` clause: "**appends** them to that session's artifact in **one** write-bearing
    file operation" (plan-summary.md:160)
  - qa-plan.md TC-105 `Forbidden result` ALREADY names witness B: "a rewrite-per-flush whose bytes grow
    `O(N²)` (witness B)" (qa-plan.md:114)
  - TC-102 `Lifecycle`: "append-only — the artifact is never rewritten" (qa-plan.md:106)
Admissible candidates:
  (a) a byte-volume clause in G-001's `Met when` (the goal-shape field Red named)
  (b) a written-bytes assertion added to VP-T002-write-count-reduced's cells and to TC-105's cells
  (c) a new VP/TC dedicated to the byte axis
Rejected candidates:
  (c) -> contradicts the plan's own F-3 ("keep one write-count VP"; VP-T002-single-write-per-flush was
        retired and folded into VP-T002-write-count-reduced, plan-summary.md:245) and is strictly broader
        than (a)+(b) for the same observable (closure step 3)
  (a) alone -> leaves the named `Forbidden result` unobservable; (b) alone -> leaves the goal row silent on
        its own failure witness. They are one remediation, not two options.
Selected closure: (a)+(b) as ONE repair. G-001's `Met when` gains the byte clause; VP-T002-write-count-reduced's
  observable-behavior and `Evidence needed` cells gain the written-bytes assertion; TC-105's `Visible result` and
  `Evidence source` gain it; T004's write-count ADDED clause names it.
Risk shape: unchanged — append-only is already required by T002 + the Anchor Map + G-001's witness B. The repair
  adds no behavior, no referent, no file and no Anchor Map row; it makes an existing requirement observable.
Deferred additive work: none
Owning route: interview (plan repair) -> forward T004 (authors the assertion in the write-count test)
Question required: no
```

**Plan-surface change: YES.** It changes what `MS-002`'s gate can observe — TC-105's `Forbidden result` already named witness B, but neither its `Evidence source` (the instrumented write counter + directory listings) nor any other cell measured bytes.

---

### Packet 2 — G-2

```text
CLOSURE: absorbed
Finding: G-2 (iterations/024-forward-red.md)
Required outcome: No approved artifact may assert that `change-focused` is the default.
Authority:
  - T002's ADDED: `config.DUMP_FORMAT: str = "documented"` (plan-summary.md:153)
  - PI:T002/dump_format/default_format_is_documented — hard_requirement (plan-summary.md:307)
  - VP-T002-default-encoding's finalized clause: "the output is the **documented layout**" (plan-summary.md:241)
  - handoff-context.md:135's retraction: "'the change-focused encoding is the default' is **false**" (018/019)
  - US-012's default clause; `OI:` entry 1's stamped statement ("**The default is `documented`**")
Admissible candidates: delete "(default)" from G-006's `change-focused` arm.
Rejected candidates:
  - reword to "change-focused (not the default)" -> adds words where the default is already carried by three
    other cells (T002, the PI:, VP-T002-default-encoding); not the smallest complete remediation (closure step 4)
  - leave it -> preserves a claim the plan has itself retracted (closure step 3: preserves a verified defect)
Selected closure: delete "(default)" from G-006's `Met when`. The blueprint element that carries the default is
  PI:T002/dump_format/default_format_is_documented; the goal row needs no replacement clause.
Risk shape: unchanged.
Deferred additive work: none
Owning route: interview (plan repair)
Question required: no
```

**This is a correction, not a new decision** — as the dispatch states. It is an editorial deletion; no gate observes it differently.

---

### Packet 3 — G-3

```text
CLOSURE: absorbed
Finding: G-3 (iterations/024-forward-red.md)
Required outcome: G-008(b)'s comparison must be an input the roadmap switch trigger can consume. The trigger is
  "the user's judgment of the `MS-002` comparison evidence" (roadmap.md:56-58); a user cannot judge a pytest
  invocation that scrolled past.
Authority:
  - US-012: "Old behavior should stay unaffected until we prove a new one is better and it's worth switching to"
  - roadmap.md:55-58 — the switch entry's `Trigger` names the comparison evidence as its input
  - G-008(b)'s `Met when`: "a comparison of the collapsed path against the documented path exists at `MS-002`,
    on named measures **M1** … and **M2**" (plan-summary.md:73)
  - MS-002's `Complete when`: "the two counts … are **recorded side by side**" (plan-summary.md:219)
  - TC-105's `Visible result`: "recorded side by side" (qa-plan.md:113)
Admissible candidates: a durable, human-readable comparison record produced at MS-002 by the MS-002 Yield Phase.
Rejected candidates:
  - "the comparison IS the paired TC-105/TC-108 runs" -> leaves the roadmap trigger inputless; US-012's second
    clause ("until we prove…") becomes unsatisfiable. Closure step 3: knowingly leaves a required gate red.
  - an Anchor Map `Mode: NEW` row for the comparison artifact -> the creation gate authorizes code referents
    (commands, handlers, functions, types, modules, classes, widgets, stores, protocols, registries, configuration
    surfaces); a QA evidence record is none of those, and adding a row would authorize an artifact that does not
    exist in code. This is Red's preferred reading; it is NOT adopted as evidence.
  - placement sub-variants (session artifact vs roadmap.md vs the MS-002 yield record) -> same observable and same
    contractual outcome; mechanism, not outcome (closure step 2).
Selected closure: name the location AND the owner — the MS-002 Yield Phase's comparison record, a durable
  human-readable artifact. Named in G-008(b)'s `Met when`, MS-002's `Complete when`, T004's comparison clause, and
  qa-plan.md's TC-105 cells.
Risk shape: unchanged — no behavior is added; an already-declared requirement becomes observable.
Deferred additive work: none
Owning route: interview (plan repair) -> forward-qa QAP-MS002 (authors the record)
Question required: no
```

**Plan-surface change: YES.** It changes what a gate produces and what the roadmap trigger consumes.

---

### Packet 4 — G-4

```text
CLOSURE: absorbed
Finding: G-4 (iterations/024-forward-red.md)
Required outcome: G-003's failure witness must not classify the settled append-time flush as a goal failure.
Authority (all settled and approved):
  - G-005's `Met when`: the threshold check fires at append time inside `dump_turn` (plan-summary.md:70)
  - T003's ADDED: "the append-time volume check in `dump_turn` — after the append, flush when
    `state.pending_bytes >= config.DUMP_FLUSH_BYTES`" (plan-summary.md:173)
  - Anchor Map `bulk flush`: "contains **no `await`**" (plan-summary.md:112)
  - Anchor Map `dump entry point`: "Also hosts the append-time volume check (G-005, collapsed path only)" (plan-summary.md:109)
  - source: routes/messages.py:57 calls `dump_turn(body)` synchronously inside the async handler — no await, no offload
  - G-004's own boundary ALREADY disclaims a latency bound: "not a request-latency bound" (plan-summary.md:69)
Admissible candidates: keep the settled design; sharpen G-003's witness so "delayed" means breakage, not added latency.
Rejected candidates:
  - move the flush off the request path -> violates G-005's `Met when` and T003's settled ADDED text (closure step 3:
    violates an approved decision). This is the reading G-003's current wording admits, and it is the one to exclude.
Selected closure: G-003's `Failure witness` becomes a *failure* — a 5xx, a dropped connection, or a body the proxy
  altered — with an explicit carve-out that the append-time flush G-005 requires is synchronous by design and is not a
  delay of this kind; G-003's `Boundary / non-goals` gains the same no-latency-bound clause G-004 already carries.
Risk shape: unchanged — the design is not touched; the artifact stops admitting the reading that would have re-opened it.
Deferred additive work: none
Owning route: interview (plan repair)
Question required: no
```

---

### Packet 5 — G-5 (weakest row)

```text
CLOSURE: absorbed
Finding: G-5 (iterations/024-forward-red.md)
Required outcome: Either the shipped configuration is shown to deliver a bulk flush, or the floor is confirmed as the
  goal's count axis and a trigger that fires on every turn is explicitly a pass by design.
Authority:
  - §Goals and Constraints: "the reduction is `1 − F/(2N)` — **floor 50%** at `F = N`, zero unreachable (an empty
    flush writes nothing, so `F ≤ N`)" (plan-summary.md:51-55) — deliberate and explicit
  - G-001's `Met when`: "**50% ≤ reduction < 100%** for every achievable `F`" (plan-summary.md:66) — deliberate
  - `FK-A`/`FK-B` are NOT askable; the trigger VALUES are Forward discretion (plan-summary.md:23;
    handoff-context.md:116-118)
  - TC-100 (qa-plan.md:75) runs the collapsed path with **no flush parameter passed** and asserts the deferral —
    so "no scenario runs the defaults" is not accurate for the deferral property
Admissible candidates: close on the recorded floor.
Rejected candidates:
  - a `Met when` clause tying the shipped defaults to `F ≪ N` -> converts `FK-A`/`FK-B`'s Forward-discretion values
    into plan commitments; broader than the smaller complete remediation (closure step 3)
  - a new VP/TC exercising the shipped defaults -> same reason, plus F-3's one-write-VP rule
Selected closure: **CLOSED ON THE RECORDED FLOOR — not repaired into a new gate.** One clarification is added so the
  artifact stops admitting the misreading: G-001's `Boundary / non-goals` states that the 50% floor IS the goal's count
  axis, that `F = N` is a pass *by design*, and that the buffering cadence is gated by VP-T001-writes-deferred.
Risk shape: unchanged.
Deferred additive work: none
Owning route: interview (plan repair, clarification only)
Question required: no
```

**Honest residual, recorded rather than buried.** With the shipped `DUMP_FLUSH_BYTES = 4 MiB`, a session whose individual turns render above 4 MiB produces `F = N`; the plan calls that a pass. Closing on the floor is the agent-decidable reading (reversible in one commit, loses no data, contradicts neither `US-012` nor a protected invariant — `US-013`'s delegation covers it). *Tightening* the floor would contradict an approved goal row and is therefore not the agents' to make — but no settled authority requires it either, so **no fork is owed**. If the user ever wants the shipped defaults to guarantee `F ≪ N`, that is a goal-scope change for a later session, not a Red-findings repair.

---

### Packet 6 — T-1

```text
CLOSURE: absorbed
Finding: T-1 (iterations/024-forward-red.md, task rows)
Required outcome: The append condition has exactly one authoring task, and no task depends on a global another task adds.
Authority:
  - T002's ADDED claims `config.DUMP_FORMATS` / `config.DUMP_FORMAT` (plan-summary.md:153)
  - `config.py` has NO `DUMP_FORMAT` today (read in this pass; only `DUMP_DIR`, `KV_POLL_INTERVAL`, …)
  - T001 "Depends on: —" (plan-summary.md:150); T002 "Depends on: T001" (plan-summary.md:168)
Admissible candidates:
  (i) T001 appends unconditionally; T002 adds the condition. The condition is stated exactly once, in T002.
Rejected candidates:
  - T001 authors the globals itself -> duplicates T002's ADDED item; two owners for one artifact
    (closure step 3: violates approved ownership)
  - T001 keeps the condition and depends on T002 -> T002 depends on T001 -> dependency cycle (closure step 3)
  - reorder T001/T002 -> T002's dispatch appends to the buffer T001 adds, so T002 still depends on T001; same cycle
Selected closure: move the conditioning clause from T001's ADDED to T002's ADDED. In its own increment T001's append is
  unconditional, drains nothing (`flush_all` has no caller until T003), and stays green, additive and unit-testable exactly as
  its REMOVED clause already says ("the buffer is populated but not yet drained", plan-summary.md:149).
Risk shape: unchanged — the end-of-MS-001 behavior is identical; only the authoring task changes. Transient and in-milestone:
  between T001 and T002 the buffer accumulates on the documented path in memory only (a list append is not a write-bearing file
  operation), so neither TC-107 nor TC-108 observes it.
Deferred additive work: none
Owning route: interview (plan repair)
Question required: no
```

---

### Packet 7 — T-2

```text
CLOSURE: absorbed
Finding: T-2 (iterations/024-forward-red.md)
Required outcome: A bare `uv run pytest` must neither collect NOR import `tests/qa/`; the hermetic default suite stays green
  and network-free (qa-plan.md:170), and MS-003's gate IS that bare run (plan-summary.md:220).
Authority: qa-plan.md §Phase Requirements (:170); MS-003 (:220); T002's ADDED 022 clause (:164); `.agents/skills/qa-project/SKILL.md` §Conventions.
Mechanism ranking — verified against the INSTALLED pytest 9.0.2 source in this pass:
  - `_pytest/main.py:425` `pytest_ignore_collect` is the single hook. `--ignore=` is evaluated at `:433-438`;
    `norecursedirs` at `:455-458` (and only `if collection_path.is_dir()`). BOTH prevent collection and therefore import.
  - `Dir.collect` (`:552-573`) skips `pytest_ignore_collect` for initial paths and their parents
    (`isinitpath(..., with_parents=True)`), so BOTH are gate-safe: an explicitly selected `tests/qa/...` gate still collects.
    023's verification ("pytest 9.0.2 skips `pytest_ignore_collect` for initial paths and their parents") therefore covers
    `norecursedirs` as well — it is evidence about the hook, not about one mechanism.
  - `norecursedirs` is registered `addini(..., type="args", default=[...])` (`:213-222`) and an ini value is NOT merged with the
    default, so a value of `["qa"]` alone drops `.*`, `build`, `dist`, `node_modules`, `venv`, `*.egg`, `_darcs`, `CVS`, `{arch}`.
    (`.venv` is still caught by the separate pyvenv.cfg check at `:451-453`.)
  - an `addopts` marker (`-m "not qa"`) prevents EXECUTION only: the module is collected and imported, then deselected —
    inadmissible against a constraint that names network-free.
Admissible candidates: `norecursedirs`; `--ignore=tests/qa` in `addopts`.
Rejected candidates:
  - `addopts` marker -> fails the named constraint (import still happens; qa-plan.md:170 names network-free)
  - `--ignore=tests/qa` -> equivalent on the constraint axis, but the value is resolved relative to the invocation dir
    (`:435` `absolutepath`), so it silently stops excluding when pytest runs from elsewhere — a silent failure mode on
    exactly the axis being guarded, where `norecursedirs` is basename-matched and location-independent.
Selected closure: name `norecursedirs` and DELETE "(or an equivalent `addopts` marker / directory-level skip)"; and state that the
  ini value REPLACES pytest's default `norecursedirs` list, so the configured value must retain the default patterns alongside `qa`.
Risk shape: unchanged — one config key; no new file, no new dependency, no new test.
Deferred additive work: none
Owning route: interview (plan repair) -> forward T002 (authors the key)
Question required: no
```

---

### Packet 8 — T-3

```text
CLOSURE: absorbed
Finding: T-3 (iterations/024-forward-red.md)
Required outcome: The hermetic default suite stays green and network-free (qa-plan.md:170) AND the real-signal shutdown
  coverage exists at the tier that can host it.
Authority:
  - qa-plan.md §Phase Requirements (:170): "the `tests/qa/` gates are selected explicitly and are not part of it"
  - TC-110: tier "E2E (real signal)", scenario "subprocess + real signal + post-exit listing + exit code" (qa-plan.md:53)
  - MS-003's gate is the bare `uv run pytest` (plan-summary.md:220)
  - T002's ADDED 022 clause keeps `tests/qa/` out of that bare run (plan-summary.md:164)
  - source: `server.py:41-46` `on_startup` calls `backend.detect_backend(..., backend_url=config.UPSTREAM, ...)`;
    `config.py:15` `UPSTREAM = "http://localhost:30000"` — a spawned app hits the network at startup.
Admissible candidates: T004's ADDED clause drops the "real-signal (subprocess) cases" claim from `tests/test_dump.py`; the
  shutdown path's in-process coverage (`on_cleanup`) stays in the unit list; the real-signal scenario remains TC-110's, at
  MS-002 / QAP-MS002 under `tests/qa/`.
Rejected candidates:
  - spawning the real app from `tests/test_dump.py` -> breaks the constraint MS-003 gates (closure step 3)
  - leaving the phrase as written -> same defect, and the same behavior is then claimed by two tiers
Selected closure: delete "real-signal (subprocess) cases" from T004's `tests/test_dump.py` ADDED clause and point at TC-110
  (MS-002, `tests/qa/`) as the real-signal gate.
Risk shape: unchanged — no coverage is lost: TC-110 keeps the real-signal scenario end to end, and T004's `on_cleanup`
  in-process coverage stays in the unit list.
Deferred additive work: none
Owning route: interview (plan repair)
Question required: no
```

**Plan-surface change: YES.** It changes what the hermetic suite (MS-003's gate) contains.

---

## Footnote disposition (F-1…F-4) — CONFIRMED, none corrected

All four are **forward discretion, yield-reviewable**, and each is **resolved by settled plan authority** rather than left open. No repair is owed; the disposition is confirmed as authored.

| # | Disposition | Why confirmed (authority verified in this pass) |
|---|---|---|
| F-1 | **CONFIRM** — one value only: `config.DUMP_FORMAT`; `encoding` is the sub-axis a collapsed record's *contents* vary on | The term is settled by 019 (`PI:T002/artifact_encoding/*` → `PI:T002/dump_format/*`) and T002 states it outright: "The **term** is `format`, not `encoding`" (plan-summary.md:115). The Anchor Map carries **no** encoding-selector row, so a reader who tried to add `config.DUMP_ENCODING` would be creating a referent the creation gate does not authorize. `VP-T002-default-encoding`'s handle is retained deliberately (handle-stability rule). A reader who follows T002 gets it right; nothing to repair |
| F-2 | **CONFIRM** — spawn the timer unconditionally; it no-ops on the documented path | T002's ADDED `flush_all()` clause ("no-ops when `config.DUMP_DIR is None` or nothing is pending", plan-summary.md:160); the Anchor Map's `flush timer` row ("No-ops on the documented path (018, F-15)", :113); `spawn_kv_cache_monitor` is the shape peer. Unconditional spawning also keeps `on_cleanup` uniform: it always has a task to cancel |
| F-3 | **CONFIRM** — create the artifact lazily; never `touch` it when the session dir is made | The write unit excludes only directory creation (plan-summary.md:239; handoff-context.md:168), and `VP-T002-write-count-reduced` asserts `= F`. An eager create makes the collapsed total `F + 1`. `session_dir.mkdir` already happens at state creation (`dump.py:72`) and is the excluded directory operation; the artifact itself must not be pre-created |
| F-4 | **CONFIRM** — per-turn `ValueError` from the dispatch inside `dump_turn`, caught by `routes/messages.py`'s swallow-all; **no** startup check | `T002`'s ADDED states it exactly (plan-summary.md:155); `main.py`'s `click.Choice` already rejects a bad value at parse time; `T004` unit-tests the guard. A startup validation would add behavior no task lists **and** grow `main.py:main` — the already-violated capacity function (`P=21`, `L≈31`) whose closure 017 absorbed by keeping parameters out of it |

## Excluded rows (G-6, G-7, T-4, T-5, T-6) — no action, verified

Re-checked against the artifacts; all five exclusions hold and each cites a real blueprint element. **G-6** (the "does NOT meet this clause" wording) is closed by the same cell: "That is a disclosed consequence of the settled reading, **not a failure of the goal**" (plan-summary.md:66). **G-7** (traceback exit) is closed by T003's "wrapped so it cannot raise" and `TC-110`'s `Forbidden result` (:27). **T-4** (the in-repo precedent points the wrong way) is closed by the three explicit `callback` clauses + `CLOSURE: absorbed` (:58). **T-5** (binding the flush into `poll_kv_cache`) is closed by the Anchor Map's `Mode: NEW` timer row and `RB-002`'s "wrong concern" (:113, :137). **T-6** (README rewrite) is closed by T005's struck-through REMOVED clause marked "DROPPED (018), CONFIRMED (019)" (:193). No repair owed.

## Plan-surface changes (recorded as such, not as editorial fixes)

| Row | Surface | What changes for a gate |
|---|---|---|
| G-1 | `G-001` `Met when`, `VP-T002-write-count-reduced`, `TC-105`, `T004` | **MS-002's gate gains a byte-volume observable.** `TC-105`'s `Forbidden result` already named witness B; the repair makes it *falsifiable* — the instrumented write counter must record per-write payload bytes, and the assertion is that total bytes written track the pending batches, not the accumulated artifact. A rewrite-per-flush implementation fails a declared gate for the first time |
| G-3 | `G-008`(b) `Met when`, `MS-002` `Complete when`, `T004`, `qa-plan.md` TC-105 | **MS-002's gate gains a produced artifact.** The comparison becomes a durable record the roadmap switch trigger can consume, not only an in-test assertion. No Anchor Map row is added — a QA evidence record is not a code referent |
| T-3 | `T004`'s `tests/test_dump.py` ADDED clause | **MS-003's gate (the bare `uv run pytest`) loses a claimed member.** The real-signal coverage is re-attributed to TC-110 in `tests/qa/`, which is the tier that already owns it |

G-2, G-4, G-5, T-1 and T-2 are **editorial/task-boundary repairs**: no gate observes a different thing, no task moves, no dependency edge changes.

## Repairs authorized for `interview` (one batch — apply exactly these, nothing wider)

**R1 — `plan-summary.md` §Goal Definition Matrix, `G-001` row.**
- `Met when`: after "…**50% ≤ reduction < 100%** for every achievable `F` — for every shipped format." insert: "**and the artifact is append-only: each flush's write carries only that flush's pending batch, so the session's bytes written grow with `F`, not `N²` (witness B is excluded by this clause, not only by the count).**"
- `Boundary / non-goals`: append: "**The 50% floor is this goal's count axis and `F = N` is a pass by design** — the goal gates the *reduction*, not the flush cadence; the buffering cadence is gated by `VP-T001-writes-deferred` and by `TC-100`'s deferral assertion."

**R2 — `plan-summary.md` §Validation Point Matrix, `VP-T002-write-count-reduced`.**
- `Observable behavior`: after "…performs **exactly one** write-bearing operation." insert "**and the bytes written across the session grow with `F`, not `N²`** — each flush's write carries only that flush's pending batch, never the accumulated artifact (append-only, witness B)."
- `Evidence needed`: "instrumented write-call count + deferred-state observation + a known, injected `F`" → "instrumented write-call count **and per-write payload bytes** + deferred-state observation + a known, injected `F`".

**R3 — `qa-plan.md` TC-105.**
- `Visible result`: append "**and the bytes written across the session track the pending batches, not the accumulated artifact**".
- `Evidence source`: "the instrumented write counter + two directory listings, recorded as a pair." → "the instrumented write counter (**call count and payload bytes**) + two directory listings, recorded as a pair."
- `Visible result`: replace "**The paired runs are the comparison `G-008`(b) requires and are recorded side by side.**" with "**The paired runs are the measurement `G-008`(b) requires; the `MS-002` Yield Phase records them side by side in its comparison record.**"
- `Evidence source`: append " + the `MS-002` Yield Phase's comparison record (a durable, human-readable artifact)".

**R4 — `plan-summary.md` §Goal Definition Matrix, `G-008`(b) `Met when`.** "(b) a comparison of the collapsed path against the documented path exists at `MS-002`, on named measures **M1** … and **M2**" → "(b) a comparison of the collapsed path against the documented path exists at `MS-002` on named measures **M1** … and **M2**, **as a durable, human-readable record — the `MS-002` Yield Phase's comparison record, which is the input the roadmap switch trigger consumes — not merely an in-test assertion**".

**R5 — `plan-summary.md` §Milestone Matrix, `MS-002` `Complete when`.** "the two counts (collapsed `F` vs documented `2N` at the same N) are recorded side by side as the comparison `G-008`(b) requires" → "the two counts (collapsed `F` vs documented `2N` at the same N) are recorded side by side **in the `MS-002` Yield Phase's comparison record — a durable, human-readable artifact — as the comparison `G-008`(b) requires**". Also append to `MS-002`'s `Complete when`: "**and the bytes written across the session track the pending batches, not the accumulated artifact**".

**R6 — `plan-summary.md` §Task List, `T004`.** In the `VP-T002-write-count-reduced` ADDED clause, after "exactly one write-bearing operation per flush (K ≥ 2 pending turns)" insert "**and its payload bytes**"; and replace "the **pair** of runs (documented vs collapsed at the same N) **is** the comparison `G-008`(b) requires" with "the **pair** of runs (documented vs collapsed at the same N) is the **measurement** `G-008`(b) requires, and the `MS-002` Yield Phase records it as the durable comparison record"; and delete "real-signal (subprocess) cases;" from the `tests/test_dump.py` ADDED list, adding "; the real-signal shutdown scenario is `TC-110`'s, under `tests/qa/` at `MS-002`".

**R7 — `plan-summary.md` §Goal Definition Matrix, `G-003` row.**
- `Met when`: "Existing suite green; dump failure still cannot fail a request" → "Existing suite green; dump failure still cannot fail a request; **and the append-time flush's added latency is not a failure of this goal**".
- `Failure witness`: "A request fails or is delayed because of dump/buffer/flush behavior" → "A request **fails** — a 5xx, a dropped connection, or a body the proxy altered — because of dump/buffer/flush behavior. **The append-time flush `G-005` requires is not a delay of this kind**: it runs synchronously inside `dump_turn` on the request path by design (`T003`, no `await`), and no clause of this plan bounds request latency."
- `Boundary / non-goals`: append "**Not a latency bound** — like `G-004`, this goal does not bound request latency; the settled append-time flush is inside this goal's carve-out."

**R8 — `plan-summary.md` §Goal Definition Matrix, `G-006` row.** `Met when`: delete "(default)" from "*change-focused* (default):".

**R9 — `plan-summary.md` §Task List, `T001`.** In the second ADDED clause, "**only when a collapsed format is selected (018, `F-21`)**. On the documented path the turn is written at append time exactly as today and **nothing is buffered**; an unconditional append plus a flush would yield `2N + F` writes (`F-13`)." → "**In this increment the append is unconditional** — no trigger exists yet, so nothing is drained; **`T002` adds the condition that makes it collapsed-path-only** (018, `F-21`)."

**R10 — `plan-summary.md` §Task List, `T002`.** Add to the ADDED list, immediately after the format-dispatch item: "**ADDED: the append to `state.pending` is conditioned on the collapsed format being selected (018, `F-21`)** — on the documented path the turn is written at append time exactly as today and **nothing is buffered**; an unconditional append plus a flush would yield `2N + F` writes (`F-13`)."

**R11 — `plan-summary.md` §Task List, `T002`, the 022 collection-exclusion clause.** "— `norecursedirs` (or an equivalent `addopts` marker / directory-level skip)" → "— **`norecursedirs`**, whose ini value **replaces** pytest's default list (`_pytest/main.py:213-222`), so the configured value must retain the default patterns (`.*`, `build`, `dist`, `node_modules`, `venv`, `*.egg`, `_darcs`, `CVS`, `{arch}`) alongside `qa`. An `addopts` marker would not deliver the constraint (it deselects after import, and the constraint names network-free); `--ignore=` is equivalent on that axis but resolves relative to the invocation dir."

**R12 — `qa-plan.md` §Milestone Gates, `MS-002` row.** Mirror R5's two clauses so the QA plan and the plan agree.

**R13 — `roadmap.md` §Switch the default dump format, `Trigger`.** Append: "The evidence is the `MS-002` Yield Phase's comparison record (a durable, human-readable artifact), not an in-test assertion." No clause is removed.

**R14 — `handoff-context.md`** — updated by this pass (see below). `T003`'s and `MS-002`'s existing clauses need no edit beyond R5/R6; `MS-001`'s "no flush is observable here" is correct and unchanged.

## Compaction Decision

- Prior same-phase files reviewed: `003-decide.md`, `008-decide.md`, `010-decide.md`, `011-decide.md`, `012-decide.md`, `014-decide.md`, `015-decide.md`, `017-decide.md`, `019-decide.md`, `022-decide.md`.
- Superseded: **none.**
- Not superseded: **all ten.** Every prior decide file covers a different slice (option generation, the `FK-003`…`FK-007` chain, the TLS cluster, the payload cluster's 018/019 finalization, the 021 completeness repair). This run is the `red-findings` closure of `024-forward-red.md`'s eight ambiguous rows — a slice no prior decide file touches.
- Compaction performed: **no.**
- If no: no prior decide report is superseded — this pass changes no decision any of them originated, and no `FK` was raised (so there is no fork content to make self-contained in `decisions.md`). Deleting a report that still holds unresolved planning state is forbidden; none of the ten qualifies for deletion.

## Deterministic Closure Ledger

No `Deterministic Closure Ledger` exists in this session (grepped across the session directory: zero hits). The eight entries above are the ledger content for this pass; `interview` lands them with the repairs, or the routing agent appends them to the session's ledger if one is created.

| # | Finding | Closure | Selected remediation | Plan-surface change | Question |
|---|---|---|---|---|---|
| 1 | G-1 | absorbed | `G-001` byte clause + written-bytes assertion in `VP-T002-write-count-reduced` / `TC-105` / `T004` | **yes** | no |
| 2 | G-2 | absorbed | delete "(default)" from `G-006`'s `change-focused` arm | no | no |
| 3 | G-3 | absorbed | name the durable `MS-002` comparison record in `G-008`(b), `MS-002`, `T004`, `TC-105` | **yes** | no |
| 4 | G-4 | absorbed | `G-003`'s failure witness = breakage, with the append-time-flush carve-out + no-latency-bound boundary | no | no |
| 5 | G-5 | absorbed | **closed on the recorded floor**; explicit `F = N` boundary statement | no | no |
| 6 | T-1 | absorbed | move the append-conditioning clause from `T001` to `T002` | no | no |
| 7 | T-2 | absorbed | name `norecursedirs`; delete the "or equivalent" list; add the default-replacement clause | no | no |
| 8 | T-3 | absorbed | delete "real-signal (subprocess) cases" from `T004`; point at `TC-110` | **yes** | no |

## Assumptions made in this pass

1. **`TC-105`'s `Forbidden result` is the plan's own declaration that witness B must be observable** — the repair supplies the missing evidence source rather than inventing a new requirement. Assumption, but grounded: the `Forbidden result` cell is a blueprint element and already names witness B.
2. **A QA evidence record is not a code referent** — so no Anchor Map `Mode: NEW` row is owed for G-3's comparison record. Assumption; the Anchor Map's creation gate enumerates code referents only (plan-summary.md:105, 129-130).
3. **`norecursedirs` and `--ignore=` are equivalent on the collection axis** — verified in the installed pytest 9.0.2 source (both inside `pytest_ignore_collect`), not assumed.
4. **Closing G-5 on the floor does not contradict `US-012` or a protected invariant** — `US-012` gates the *layout switch*, not the trigger values; the floor is an approved goal clause.

## No row requires the user

Every one of the eight rows resolves by a repair that is **reversible in one commit and loses no data**, and none touches the public CLI surface, security, an irreversible act, or a recorded user decision (`US-012`'s reading and `US-014`'s stamps are untouched; `OI:`/`PI:` statements are not edited). The escalation floor fits no clause, so **no `FK` was reserved, no question was asked, and none is owed**. The assumption gate remains the next dispatch once `interview` has applied the repairs.

**DECIDE-RED: 8 rows | 0 invalid | 8 deterministic | 0 user forks | 0 plan breaks**
