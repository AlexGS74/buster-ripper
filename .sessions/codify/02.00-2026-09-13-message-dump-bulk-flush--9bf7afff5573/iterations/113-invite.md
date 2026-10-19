# Iteration 113 — invite repair pass (two label repairs to the `FK-011` capsule)

## Orchestrator Prompt (verbatim)

> You are a subagent spawned by another session.
> Session ID: 02.00-2026-09-13-message-dump-bulk-flush--9bf7afff5573
>
> Re-dispatch for two label repairs only, per [112-sprint](112-sprint.md): the draft capsule in [111-interview](111-interview.md) fails the Decision Capsule Contract's Readable Handles rule in two places — Detail ¶ "Where it sits" carries the bare task range `T001`–`T022` with no labels, and Detail ¶ "Decision trail" carries `F-5` bare where its defining row's form is `F-5 · preserved-thinking default`. Repair only those two labels, keep the record, options, recommendation and every other field unchanged, and re-emit `INVITE-READY`.

## Session Type / Mode

- **Phase**: `invite` — **repair pass only**, scoped by [112-sprint](112-sprint.md) §Blocking Findings. Not a new planning round, not a queue refresh, no re-analysis.
- **Question Mode**: `deferred` (`plan-summary.md` §Session Metadata). Nothing is presented here; `interview-sprint` presents.
- **Session type**: continuing. No `parent_session_uid` in `session-identity.json` → no parent coordination contract.
- **Outcome**: **two labels repaired in place**, `FK-011 · preserved-thinking serve default`'s draft capsule now satisfies the Readable Handles rule at both named locations, and `INVITE-READY` is re-emitted.

## Context Re-established (self-check before work)

| Source | Read | Result |
|---|---|---|
| `session-identity.json` | yes | `session_id` / `session_uid` match the handle suffix `9bf7afff5573`; **no** `parent_session_uid` |
| [112-sprint](112-sprint.md) | yes | `SPRINT-BLOCKED` before presentation; exactly two defects, both owned by `invite`, both rendering defects in [111](111-interview.md) §Topic Queue entry 1 |
| [111-interview](111-interview.md) §Topic Queue entry 1 | yes | one entry, `FK-011 · preserved-thinking serve default`, **READY** (wave 1); complete draft capsule; the only defects are the two named |
| `decisions.md` §`FK-011` | yes | still the one `State: open` record; `Label: preserved-thinking serve default`; `Presented: not yet presented` |
| [057-codex-review](057-codex-review.md) §F row `F-5` | yes | first cell reads `F-5 preserved-thinking default` — the label source for defect 2 |
| `plan-summary.md` / `plan-completed.md` task rows | yes | `T001`…`T022` are all 22 defined task IDs with no gap — the count behind the rephrased sentence |

**Context conflicts detected: none.** The record, both options, the `none` recommendation and the `Not deciding` scope are unchanged; only two label renderings moved.

## Repairs Applied

Two edits, both inside the draft capsule at [111](111-interview.md) §Topic Queue entry 1, both rendering-only.

| # | Detail ¶ | Before | After | Label source |
|---|---|---|---|---|
| 1 | **Where it sits** | "The plan's `T001`–`T022` and the four milestones carry no preserved-thinking clause" | "The plan's twenty-two tasks and its four milestones carry no preserved-thinking clause" | The range is dropped entirely; the count is verified against the plan's own task rows (`T001`…`T022`, 22 defined IDs, no gap) |
| 2 | **Decision trail** | "Finding **F-5** in [057-codex-review](057-codex-review.md) §E" | "Finding **F-5 · preserved-thinking default** in [057-codex-review](057-codex-review.md) §E" | [057-codex-review](057-codex-review.md) §F row `F-5`, first cell `F-5 preserved-thinking default`; the capsule renders it with the contract's `·` separator, as it already does for `FK-011 · preserved-thinking serve default` and `T006 · …` |

**Why repair 1 rephrases instead of labeling the range**: a range cannot carry one label per member without turning one clause into twenty-two handles; 112 offered the handle-free rephrasing and the sentence keeps its exact meaning (the plan's whole task set and all four milestones carry no preserved-thinking clause). The two tasks the paragraph actually discusses — `T006 · upstream TLS certificate-verification opt-out` and `T012 · preserve environment values as single argv words in the serve script` — already carry their labels and are untouched.

**Why repair 2 copies the row rather than inventing a label**: the label text `preserved-thinking default` is verbatim from the defining row; only the separator matches the contract's rendered form. No new name is coined.

## Verification

| Claim | How verified | Result |
|---|---|---|
| Exactly two lines changed | `git diff -- iterations/111-interview.md` | 2 insertions, 2 deletions — the two lines above, nothing else |
| Record unchanged | `decisions.md` not opened for write | `FK-011` `State: open`, `Label`, `Type`, options, `Recommendation: none`, `Presented: not yet presented` — untouched |
| Options unchanged | diff of the card table and both `Option … in full` paragraphs | byte-identical |
| Recommendation unchanged | diff of the `**Recommended:**` line | byte-identical (`none`) |
| Every other capsule field unchanged | diff review of the full capsule (card, Why now, Today, Not deciding, Answer, Downstream, queue metadata) | unchanged |
| No bare task range remains in the capsule | `sed -n '176,211p'` + handle scan | `T001`–`T022` gone; `T006`/`T012` labeled; no other bare `T`/`F`/`MS`/`US` handle |
| The repaired `F-5` form matches its source | [057](057-codex-review.md) §F row `F-5` vs the capsule text | label verbatim; separator is the contract's `·` form |
| `twenty-two tasks` is accurate | task rows in `plan-summary.md` + `plan-completed.md` | all of `T001`…`T022` defined, no gap → 22 |
| No file written outside this session's directory | file inventory of this pass | [111](111-interview.md) (two lines), this report, one status line in `handoff-context.md` |

## What Was Not Done

- **No re-analysis**: no source re-read, no re-derivation, no admission or ranking touched.
- **No new FK, no `US-` ID, no reserved, filled, superseded or reused ID.**
- **No `Presented:` block** written in `decisions.md` §`FK-011`; presentation remains Sprint's.
- **No wave index** appended to [111](111-interview.md); no wave has reached the user.
- **No other capsule text, Landscape sentence, queue entry, plan, invariant, roadmap or QA file** was edited.

## Observations Reported, Not Repaired

1. **One further bare record handle remains in the same capsule**, outside this dispatch's authorized scope: Detail ¶ **Decision trail** reads "the assumption-decide pass that reserved `FK-011` on 2026-10-05". Under the Readable Handles rule that `FK-011` is a bare handle; the one-word repair would be `FK-011 · preserved-thinking serve default`. **Not applied**, because this dispatch authorized exactly two label repairs; reported to the orchestrator, which may fold it into the same re-dispatch or leave it as is — 112's contract check recorded only the two defects now repaired, and the capsule's card and answer key already render `FK-011` with its label.
2. **Landscape sentences outside the capsule** still carry bare handles (`T001`–`T022` at [111](111-interview.md) §1, `F-5` at §1) and the queue locator list is bare by design (it is a locator list, not capsule text). 112's contract check scoped itself to the capsule; this pass did not touch the Landscape, per its dispatch.
3. The stale "four whole-list argv equalities" in `FK-011`'s `If-wrong` was already corrected to **14** in the capsule at [111](111-interview.md) §Fork Fidelity; nothing owed.

## Questions Asked

**This round: 0.** Invite prepares and repairs the wave; it does not present and does not ask. No `FK-` or `US-` ID was reserved, filled, superseded or reused.

## Gates Found Unmet (reported, not cured)

**Count: 0 blocking.** The two blocking defects from [112](112-sprint.md) are repaired and re-verified above. The carried-forward "no Decide report for this round" belongs to the orchestrator and is unchanged by this pass.

## Next Step

1. **`interview-sprint`** re-validates the repaired capsule against the Decision Capsule Contract, persists the exact text as `FK-011 · preserved-thinking serve default`'s `Presented:` block, appends the wave index to [111](111-interview.md), and presents the single capsule.
2. On the labeled answer, `interview-reconcile` records it and routes the adhoc docs task (Option A) or the script + docs + witnesses + assumption gate (Option B).

INVITE-READY: iterations/111-interview.md | ready 1 | stubs 0
