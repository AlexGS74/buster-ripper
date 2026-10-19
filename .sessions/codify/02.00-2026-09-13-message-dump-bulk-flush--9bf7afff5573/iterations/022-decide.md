# 022 — Decide (SCOPED REPLAN: the three `decide-completness` gap rows)

## Orchestrator Prompt (verbatim)

> Execute the decide-interactive skill for this session. Arguments: $ARGUMENTS
>
> Session ID: 02.00-2026-09-13-message-dump-bulk-flush--9bf7afff5573
>
> Planning is closed and the plan is marked implementable. Run your completeness check: determine whether perfect execution of the approved task set is sufficient to reach every declared goal, MVP outcome, and hard constraint.
>
> Scoped replan. `decide-completness` returned GAP with three rows, recorded in `iterations/021-decide-completness.md`. Repair the plan artifacts so that perfect execution of the approved task set reaches every declared goal, milestone, and validation point.
>
> Constraints:
> - `US-013` is in force: the user delegated small, reversible, non-data-losing decisions to the agents and does not want to be asked about them. Do NOT surface a question for any row whose resolution is reversible in one commit and loses no data. If a row genuinely cannot be resolved without the user, say so explicitly and name the single irreversible or public-surface fact that forces it.
> - Do not reopen settled decisions. The finalized task table (`T001`–`T006`), the `MS-*` milestones, the goal set `G-001`..`G-008`, and the invariant set are approved and stamped. Repair the artifacts that disagree with the Task List, or move the Task List only where a recorded user decision requires it.
> - R1 note: the three flush triggers were assigned to `T003` by the user's own recorded decision. Check whether the fix is artifact-side (matrix, VP observable-after, MS-001 complete-when) or task-side, and state which you chose and why.
>
> Then update `handoff-context.md` to reflect the repair. Do not mark the plan unimplementable unless a row actually requires it.

(`$ARGUMENTS` was empty. The dispatch carried the `decide-interactive`, `codify-agents`, `critical-thinker`, `codify-user-prompt`, and `codify-deterministic-closure` skill bodies.)

## Session Type / Mode

- **Continuing session**; **scoped replan** on the output of `decide-completness` (021). No `Mode:` line → the default reading would be OPTIONS, but the dispatch names its own shape explicitly ("Scoped replan. Repair the plan artifacts…"), and no option set is admissible: the user's delegation under `US-013` covers every row, so manufacturing choices would be presenting questions the user has already declined. This pass therefore runs **PLAN-SELECTED duties on the repaired slice only** — the milestone/VP matrices, the Goal Coverage Matrix rows, and the two task clauses the gaps require — and **OPTIONS duties for nothing**.
- **Scope fence.** The finalized task table (`T001`–`T006`), the `MS-*` milestone **identities**, the goal set (`G-001`..`G-008`), the invariant set, `FK-001`/`FK-002`'s resolutions, `US-012`, `US-013`, `US-014` and the `OI:` register are **not re-opened**. No `VP-*` handle is renamed. Only *content* cells that disagree with the Task List were corrected.
- **Session identity** unchanged (`02.00-2026-09-13-message-dump-bulk-flush` / `9bf7afff5573`); no `parent_session_uid`, so the parent-child contract rule does not apply.

## Input Absorption

| Input | Origin | Disposition |
|---|---|---|
| `R1` — `G-005`'s required path is `T003`'s, not `T002`'s; `MS-001`'s task set leaves `flush_all` with no caller | 021 (derived) | **Repaired artifact-side.** Task List untouched. See §R1 |
| `R2` — the "hermetic default suite stays green and network-free" constraint has no delivering task | 021 (derived) | **Repaired by adding the clause to `T002`.** See §R2 |
| `R3` — the collapsed format cannot be selected on the run path the README documents | 021 (derived) | **Repaired by adding the wiring to `T005`.** See §R3 |
| `FK-001.Impact.Plan` — names "Task List `T003`" and "Milestone Matrix `MS-002`" as the destinations of the trigger decision | decisions.md (user, round 3) | **Governing on R1's direction.** Cited in §R1 as the reason the Task List was not moved |
| `US-013` — agents decide the small, reversible, non-data-losing items | decisions.md (user, round 15) | Applied: all three rows are settled here; no `FK-` reserved |
| `US-012` — old behavior stays unaffected until the new one is proven better and judged worth switching to | decisions.md (user, round 14) | **Not re-opened.** Honored by R3's empty `DUMP_FORMAT` default, which keeps the no-flag script path byte-identical |
| The `OI:` register entries 1–3 and the six `PI:T002/dump_format/*` (round 17) | decisions.md (user) | Untouched — no statement in this pass contradicts a stamped invariant |
| 21 `Verification: … PENDING` claim rows | plan-summary §Factual claims | Untouched — the assumption gate's, not a completeness gap |

## R1 — the volume trigger: **artifact-side**, and this was the deliberate choice

**The finding.** `G-005`'s required path (append to `state.pending` → compare `pending_bytes` against the configured threshold → `flush_all()`) is delivered entirely by `T003`. `T002` adds the collapsed writer and the flush function, but **no trigger and no caller**: after `MS-001`'s own task set (`T001`, `T002`, `T005`) `flush_all` is dead code. Three artifacts pointed the other way — the Goal Coverage Matrix's `G-005 → T002` cell, `VP-T002-volume-flush`'s `Observable after: T002`, and `MS-001`'s "Complete when: turns reach disk only after a flush — a volume-triggered flush or a shutdown flush". With them in place, 6 of `MS-001`'s 11 validation points (`TC-101`–`TC-106`, `TC-109`) could not be exercised at the milestone that declares them, and `TC-100`'s "timer effectively disabled" clause had nothing to disable.

**Two repairs were available.**

| | Fix A — artifact-side | Fix B — task-side |
|---|---|---|
| Move | the 7 flush-dependent VPs and `MS-001`'s complete-when to `MS-002` | the volume check + `config.DUMP_FLUSH_BYTES` + `--dump-flush-bytes` from `T003` into `T002` |
| Task List | untouched | rewritten |
| Milestones | content corrected, identities kept | `MS-001` gains a flush; `MS-002` loses one |
| Against | `MS-001` no longer demonstrates the collapsed path end to end | relocates part of the trigger cluster against its recorded destination |

**Chosen: Fix A.** Three reasons, in order of weight.

1. **The dispatch's own constraint decides it.** *"Repair the artifacts that disagree with the Task List, or move the Task List only where a recorded user decision requires it."* No recorded user decision requires moving the Task List here. `FK-001.Impact.Plan` (round 3) names **both** *"Task List `T003` (gains timer coroutine + app-state key + cancellation + tests)"* **and** *"Milestone Matrix `MS-002`"* as the destinations of the trigger decision. Fix B would relocate half of that cluster back into `MS-001` — i.e. against the record.
2. **The volume check's `T003` placement is plan-of-record, not an accident.** 005-organize marked `T003` "enlarged — decide finalizes"; 008-decide recorded "`T003`'s trigger half" as UNCHANGED/settled; round 18 approved the table as implementable. It has been the approved shape across three passes.
3. **`MS-001` loses nothing it was ever able to demonstrate.** Its own name is the format selector and the documented path's preservation. The collapsed path's *observability* — a flush, an artifact on disk — was always `T003`'s, so `MS-002` is where that path becomes checkable. Fix A makes the milestone map say what the code already does.

⚠ **One correction to the premise as given.** The dispatch says "the three flush triggers were assigned to `T003` by the user's own recorded decision". That is true of the **timer** and the **shutdown** wiring (`FK-001`'s `Impact` names `T003`), but the **volume check** is `T003`'s by the approved task table and organize's enlargement — not by a user statement. The distinction does not change the outcome (the table is not decide's to rewrite either way), but the record should not attribute to the user a placement that came from the plan. Recorded here rather than in `decisions.md`, because no `FK-` block is edited by this pass.

**What moved.**

- Goal Coverage Matrix: `G-005`'s task cell → `T003`; `G-008`'s `(b)` cell → `MS-002`; `G-008`'s task list `+T003`.
- Goal Definition Matrix `G-008`: both `MS-001` → `MS-002`.
- Seven VPs (`Observable after` → `T003`, `Validate at` → `MS-002`, `QA author pass` → `QAP-MS002`): `VP-T002-volume-flush`, `VP-T001-session-isolation`, `VP-T002-collapsed-artifact`, `VP-T002-change-readable`, `VP-T002-turn0-available`, `VP-T002-write-count-reduced`, `VP-T002-file-count-reduced`. **Handles unchanged** — the stability rule; only the boundary cells moved.
- `MS-001`'s and `MS-002`'s "Complete when" and VP lists: `MS-001` now declares **no flush** (4 VPs: `VP-T001-writes-deferred`, `VP-T001-dump-failure-isolated`, `VP-T002-documented-format-preserved`, `VP-T002-default-encoding`); `MS-002` carries the collapsed path end to end (10 VPs).
- `VP-T001-writes-deferred` **stays at `MS-001`** — its assertion is an *absence* (no artifact before a flush), so it needs no trigger. Its injection clause was dropped because "below both thresholds and the timer effectively disabled" named `T003`'s parameters, which do not exist at `MS-001`.
- §Goals and Constraints, the derived-constraint paragraph, the `OI:` register's roadmap sentence, §Open Questions 2f(iii) and item 8, and the Anchor Map's `dump entry point` row (annotated as a **code location, not a task mapping** — which is exactly what the mis-read cell treated it as).
- `qa-plan.md`: the MS-001/MS-002 milestone rows, `TC-101`…`TC-106` and `TC-109`, both QA author passes, §Phase Requirements, and the scenario-contract headings (TC-100/TC-101 split; TC-108 split out to `MS-001`).
- `roadmap.md`: MVP floor clause 6 and the deferred switch entry's trigger.

## R2 — the hermetic-suite constraint: clause added to `T002`

**The finding.** The constraint is declared twice (`qa-plan.md` §Phase Requirements; `qa-project` §Conventions: gates "are selected explicitly (by path or by marker) and are not part of the hermetic default `uv run pytest` run") and `MS-003`'s gate **is** that default run. Selecting gates by path does not exclude them from a bare `pytest`: `pyproject.toml:17-19` carries no `testpaths`, no `norecursedirs` and no registered marker, and pytest's own `norecursedirs` default excludes only `.*`, `build`, `dist`, `node_modules`, `venv`. `tests/qa/` is therefore collected, and `MS-003`'s "existing suite green" executes network-dependent gates against `http://hyperipper.local:30001`.

**Repair.** `T002`'s ADDED list gains the collection exclusion (`norecursedirs`, or an equivalent registered marker / directory-level skip). The constraint is a property of the repository's test configuration, which is `T002`'s file set.

**Why `T002` and not `T004`** (the two candidates 021 named): `tests/qa/` is authored from `QAP-MS001` onward, and that pass runs **after every MS-001 task has passed local Yield**. The exclusion must therefore already be in place at the end of `MS-001`. `T004` lands at `MS-003` — one milestone too late — and would leave `T003`'s local Yield collecting `tests/qa/`. Recorded verbatim in the task so a later reader does not "simplify" it back to `T004`.

## R3 — the selection surface: wiring added to `T005`

**The finding.** MVP floor clause 2 requires the reduction be "reachable, witnessed, and documented". `T005` documents `--dump-format` in the README's dump section, whose **sole** invocation example is `DUMP_DIR=~/mllm/prompt-diffs bash serve_buster_ripper.sh` (`README.md:143`). `serve_buster_ripper.sh:33-41` forwards no `"$@"` and declares no `DUMP_FORMAT`, so the flag cannot be passed on the only run path the dump documentation shows. `T006` adds `SKIP_TLS_VERIFY` to this same script on exactly this reasoning (*"the script forwards no `"$@"`, so without it the flag is unreachable on the local-deployment path"*); no task did the equivalent for the dump cluster.

**Repair.** `T005` gains: `DUMP_FORMAT=${DUMP_FORMAT:-}` and `[[ -n "${DUMP_FORMAT}" ]] && EXTRA_FLAGS="${EXTRA_FLAGS} --dump-format ${DUMP_FORMAT}"` — the exact shape `DUMP_DIR` already uses, including the empty default, so the default path is untouched and `VP-T002-default-encoding`'s "no `--dump-format` passed" clause stays true **on the script path as well**. The README's dump section gains `DUMP_FORMAT=change-focused DUMP_DIR=… bash serve_buster_ripper.sh` beside the existing example.

**Home task: `T005`, not `T006`.** `T005` owns the README's dump section; `T006` owns the TLS cluster. Adding the dump cluster's script wiring to `T006` would mix clusters in one task and blur `T006`'s disposition.

**No VP.** Structural — the serve script has no validation point today, and the precedent for this row class is already recorded in §Behavior Disposition Matrix (T006's flag/README/script row). Adding one here would create a VP with no observation surface of its own.

## Fork Determination — none authored, none owed

Checked against the skill's escalation floor ("it is irreversible or expensive to reverse (data loss, public API surface, security)").

| Row | Reversible in one commit? | Data loss? | Public CLI / security surface? | Fork? |
|---|---|---|---|---|
| R1 | yes — the moved cells are plan text | no | no | **no** |
| R2 | yes — one `[tool.pytest.ini_options]` key | no | no | **no** |
| R3 | yes — two shell lines + one README line | no | no (`DUMP_FORMAT` defaults empty; the no-flag path is unchanged) | **no** |

No clause of the floor fits, so `US-013`'s delegation covers all three and the user is not asked. **No `FK-` block was reserved, authored or edited** in this pass; `decisions.md` is byte-identical to 021's state. `handoff-context.md` §Open Forks remains `None`.

## What This Pass Did Not Do

- Did not edit `decisions.md` — no `FK-`, `US-`, `PI:` or `OI:` record changes. The one factual correction to the dispatch's premise (§R1) is recorded here and in `handoff-context.md`.
- Did not rename any `VP-*` handle, renumber a `TC-*`, or change an `MS-*` identity.
- Did not touch the TLS cluster, the 21 `PENDING` claim rows, or `roadmap.md`'s three above-floor entries.
- Did not mark the plan unimplementable: no row required it. The approach is intact; the milestone map now says what the code does.
- Did not re-open any task, goal, milestone identity, `VP-*` handle or invariant. `MS-001`'s and `MS-002`'s *content* changed; their identities did not.
- Committed as the session checkpoint `f7b1c9f` — `plan-summary.md`, `qa-plan.md`, `roadmap.md`, `handoff-context.md`, `iterations/021` and `iterations/022`, and nothing else. ⚠ **The message does not carry the `Checkpoint N:` prefix** the session's earlier commits use (last was `Checkpoint 51`); the orchestrator may prefer to note or re-title it. No amend was made — a history rewrite needs its own authorization.

## Validation Performed

| Claim | How verified | Result |
|---|---|---|
| `pyproject.toml` carries no collection exclusion (R2's premise) | read of `pyproject.toml` `[tool.pytest.ini_options]` | confirmed — `asyncio_mode`, `addopts` only |
| `serve_buster_ripper.sh` forwards no `"$@"` and has no `DUMP_FORMAT` (R3's premise) | read of `serve_buster_ripper.sh:33-41` | confirmed |
| The README's dump section shows exactly one invocation, and it is the script (R3) | read of `README.md:137-159` | confirmed — `:143` |
| `T002`'s ADDED list contains no trigger, and nothing calls `flush_all` after MS-001 (R1) | read of the Task List; grep of the plan for `flush_all` callers | confirmed |
| `FK-001.Impact.Plan` names `T003` **and** `MS-002` as the trigger decision's destinations (R1's direction) | read of `decisions.md#fk-001` | confirmed |
| No `MS-001` validation point still requires a flush after the repair | re-read of `qa-plan.md`'s MS-001 row + `TC-001`/`TC-100`/`TC-107`/`TC-108` bodies | confirmed — `TC-100` and `VP-T001-writes-deferred` assert the *absence* of a flush |
| Every `VP-*` handle in the plan still resolves, and every moved VP's `Observable after` names a task that delivers its trigger | grep of `plan-summary.md` and `qa-plan.md` for the moved handles | confirmed |
| The remaining `MS-001` references are all intentional | grep, then read of each | confirmed — the milestone's own row, its 4 VPs, `TC-001`/`TC-100`/`TC-107`/`TC-108`, `QAP-MS001`, and `TC-105`'s body pointing at `TC-108`'s `MS-001` run |

## Next Step

1. **`forward-red`** — the chain is `decide-completness` → **`forward-red`** → the assumption gate → the first `forward`. This pass does not run it and does not authorize `forward`; the 21 `Verification: … PENDING` rows are still the assumption gate's.
2. **The first `forward`** should read `T002`'s and `T005`'s new clauses as load-bearing, not as notes: R2's exclusion must land **before** `QAP-MS001`, and R3's empty `DUMP_FORMAT` default is what keeps `VP-T002-default-encoding` true on the script path.
3. The checkpoint is `f7b1c9f` (see §What This Pass Did Not Do for the message-prefix caveat). Nothing further is owed by this pass.
