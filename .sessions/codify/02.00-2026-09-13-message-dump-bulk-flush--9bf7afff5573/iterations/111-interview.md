# Iteration 111 — invite (a new planning round after the `MS-001` boundary close)

## Orchestrator Prompt (verbatim)

> You are a subagent spawned by another session.
> Session ID: 02.00-2026-09-13-message-dump-bulk-flush--9bf7afff5573
> Execute the invite-interactive skill for this session. Arguments: $ARGUMENTS

## Session Type / Mode

- **Phase**: `invite` — the first agent of this planning round, run **once**. `interview-sprint` continues the round by presenting the wave; this pass presents nothing and asks nothing.
- **Question Mode**: `deferred` (`plan-summary.md` §Session Metadata). `AskUserQuestion` is unavailable on this dispatch path (background subagent), so the capsule is returned in the report for the orchestrator to present unchanged.
- **Session type**: continuing. No `parent_session_uid` in `session-identity.json` → no parent coordination contract.
- **Round**: the first planning round opened after `MS-001`'s `BOUNDARY-FINALIZED` at [110-01-forward](110-01-forward.md); the last recorded interview round is the `FK-010` wave at [020](020-interview.md) §Round 21. This report does not assign a new round number — the session's earlier round labels are not a clean sequence and inventing one would misstate the record.
- **Scope of the round**: **one causal Fork**, `FK-011 · preserved-thinking serve default`. Nothing else is open: `decisions.md` holds exactly one record in `State: open` (verified below), and no planned task or milestone reaches it.

## Context Re-established (self-check before work)

| Source | Read | Result |
|---|---|---|
| `session-identity.json` | yes | `session_id` / `session_uid` match the handle suffix `9bf7afff5573`; **no** `parent_session_uid` |
| `handoff-context.md` | yes | `BOUNDARY-FINALIZED` at [110-01](110-01-forward.md); `T015`–`T022`, `T021.001`–`T021.003`, `T022.001` and `MS-001` are `resolved`; **one open Fork: `FK-011`** |
| `plan-summary.md` | yes | Plan finalized and fully delivered; `MS-001` boundary closed; one open Fork named in §Current Phase |
| `decisions.md` | yes | Exactly one `State: open` record — `FK-011`; every `US-` is `recorded` (highest `US-016`) |
| `roadmap.md` | yes | MVP floor met; deferred above-floor entries carry named triggers; none covers preserved thinking |
| `qa-plan.md` | yes | Four milestone gates all authored and approved; no validation point touches preserved thinking |
| `iterations/095-yield-red.md`, `106-yield-phase.md`, `110-01-forward.md` | yes | Closing Yield Red CLEAN; `MS-001` `SEMANTIC-APPROVE`; boundary finalized with one reported deviation (review-object equality), no open work owed |
| `iterations/057-codex-review.md` §E | yes | Finding **F-5**, the source of `FK-011` |
| `serve_buster_ripper.sh`, `buster_ripper/routes/messages.py`, `buster_ripper/main.py`, `README.md`, `tests/test_serve_script_argv.py` | yes | FK-011's evidence re-derived on current bytes (`HEAD 1f36d90`), not adopted from the report |
| `assumptions/compositions/seam-S6.md` | yes | 25 members, `VERDICT: SAFE`, basis `sha256:cb7c0a5c…`; `Invalidates-On` names line 30's `PRESERVED_THINKING` default moving off `1` |
| `invariants/buster_ripper/assumption-claims.md:743` | yes | `CO:serve-script-env-to-argv/env-var-flag-crossing`, `SAFE` |

**Context conflicts detected: none.** The one open record is the one the handoff names, the code still does what the record says, and no delivered milestone or accepted task is reopened by this round.

## QA Capability Preflight — PASSES

`.agents/skills/qa-project/SKILL.md` exists and declares every capability this round could need:

| Required capability | Declaration | Gap |
|---|---|---|
| application start and invocation | Start command row (`.venv/bin/buster-ripper --host … --upstream … --dump-dir …`) | none |
| test framework and location | pytest + pytest-asyncio; `tests/` hermetic, `tests/qa/` process-level | none |
| teardown and prerequisites | SIGTERM/SIGINT + wait for exit; `uv sync`, remote upstream reachable | none |
| primary introspective tier and optional smoke tier | process-level end-to-end; hermetic default suite | none |
| quiescence / readiness contract | readiness = poll the listening port or startup log; quiescence = wait for exit | none |
| deterministic data mode | both flush triggers injectable; fresh temp `--dump-dir` per gate | none |

**This round proposes no new validation point.** Option A is docs-and-record only. Option B is witnessed entirely inside the hermetic suite (`tests/test_serve_script_argv.py`) plus the assumption-registry composition — neither is a QA gate, and neither needs a capability the contract does not declare. No ratification FK and no infrastructure task is owed. **Preflight result: PASSES.**

## Strategic Landscape

### 1. Where the session stands

The session's four milestones are delivered and its plan is complete:

- `MS-001` (documented-format dump path preserved) — `SEMANTIC-APPROVE` at [106](106-yield-phase.md), closing Yield Red `YR-MS001-r4` **CLEAN** at [095](095-yield-red.md) §Adjudication, `Delivered` entry written, boundary `BOUNDARY-FINALIZED` at [110-01](110-01-forward.md).
- `MS-002` (collapsed-format flush path) — `SEMANTIC-APPROVE` at [069](069-yield-phase.md), comparison record at `evidence/ms002-comparison-record.txt`.
- `MS-003` (coverage/regression) and `MS-004` (upstream-TLS opt-out) — approved; `QAP-MS004` 6/6.
- Every planned task `T001`–`T022` and every adhoc remediation row (`T013`, `T014`, `T015`–`T022`, `T021.001`–`T021.003`, `T022.001`) is `resolved`.
- Three seam records are capped/deferred with named triggers (`S21` recomposition capped, `S22`/`S23` `SAFE` with record-only re-traces) — **not** reopened here.
- Six accepted risks stand with their own `Reported:` state — **not** reopened here.

Against that, exactly **one** record is `State: open`: `FK-011 · preserved-thinking serve default`. It was reserved by an assumption-decide pass on 2026-10-05 from the branch's codex review (finding **F-5**) and has never been presented. It gates **no task and no milestone** — the record's own `Affected tasks: none in this session`.

### 2. Goal and MVP floor

The session's goal is unchanged and already met: buffer captured messages in memory and flush the dump to disk in bulk (one write per flush instead of one per turn), flushing on Ctrl-C/SIGTERM, with the documented per-turn layout preserved and the collapsed formats opt-in (`US-012`), plus the upstream-TLS verification opt-out. The MVP floor in `roadmap.md` §MVP floor has six clauses, all delivered. **This round adds no goal, no task, and no milestone.** The only thing it can add is a posture record and — in one arm — a small behavior change.

### 3. The single causal Fork

`FK-011 · preserved-thinking serve default` (type **preference**, owner `assumption-decide`, reserved 2026-10-05T14:14:28Z). The record's question, unchanged:

> `serve_buster_ripper.sh:30` defaults `PRESERVED_THINKING` to `1`, so **every** run of the documented serve path injects `chat_template_kwargs={enable_thinking: true, clear_thinking: false}` into each forwarded `/v1/messages` request, while `--preserved-thinking` appears nowhere in `README.md`. Must the serve path's outbound sampling semantics stay default-on and become a documented, recorded choice — or must they default off and require an explicit opt-in per invocation?

**Why it is the user's**: both directions are legitimate product postures, the code does what it says, the suite is green, and no recorded authority ranks the two. No goal, no `US-*`, no active invariant and no task clause addresses preserved thinking. The record's own escalation-floor check reached the same result: two materially different admissible outcomes, no ranking authority → `question`.

**Recommendation in the record: none.** Evidence establishes what the script does and what each arm costs; it does not rank the postures. This report preserves that neutrality and offers no lean.

### 4. Option A — keep the default on, document and record it

- **What it does**: add `--preserved-thinking` and `PRESERVED_THINKING` (with the default stated) to `README.md`'s serve-script section, to the standard the `--dump-format` and `--skip-tls-verify` examples already meet; record the default-on posture as a decision.
- **Concrete gain**: the shipped default stops being undocumented-by-omission; a reader of the README can see that every serve-path request asks the model to preserve thinking, and can turn it off with `PRESERVED_THINKING=0`.
- **Concrete loss**: the serve path keeps a default that changes what every deployment asks the model to do, and it remains the default. A user who wanted a conservative default still has to opt out.
- **Preserved behavior**: `serve_buster_ripper.sh:30` and `:42` byte-identical; the argv word set unchanged; the 25-member `CO:serve-script-env-to-argv/env-var-flag-crossing` composition stays current at `sha256:cb7c0a5c…`; all 14 whole-list argv equalities in `tests/test_serve_script_argv.py` stay green untouched; the injection at `messages.py:46-47` untouched.
- **Risk**: low — a documentation omission persists in no direction (it is closed), and nothing runtime-visible moves. The residual risk is that "documented" is mistaken for "endorsed by a ranked authority": the record is the only durable evidence, and it must say the posture is a choice, not an accident.
- **Reversibility**: trivial — one docs change and one decision record; reverting is a docs edit. No code, script or test byte moves.

### 5. Option B — default off, require an opt-in per invocation

- **What it does**: change `serve_buster_ripper.sh:30` to `PRESERVED_THINKING=${PRESERVED_THINKING:-0}`, document the flag and the env var, and make thinking preservation an explicit request.
- **Concrete gain**: nothing is injected unless the caller asks; the serve path stops silently changing what every deployment requests, and the default becomes the conservative one.
- **Concrete loss**: every current deployment stops preserving thinking after upgrade until it sets `PRESERVED_THINKING=1` — a silent behavior change for anyone relying on today's default.
- **Preserved behavior**: the flag stays available; the injection code is unchanged; `PRESERVED_THINKING=1` reproduces today's argv exactly; the `PRESERVED_THINKING="0"` arm already witnessed at `tests/test_serve_script_argv.py:217` keeps passing.
- **Risk**: medium — one default value moves, and the movement is user-visible. The record's `If-wrong` names the debt exactly: line 30's default moving off `1` fires the `CO:serve-script-env-to-argv/env-var-flag-crossing` composition's own `Invalidates-On`, so a fresh member re-verification plus a **25-member** `seam-S6` recomposition are owed before that composition may be cited as current; and `tests/test_serve_script_argv.py`'s `_THINKING` constant (line 62) becomes conditional, which row 201's `Invalidates-On` names as the trigger for re-verifying every pinned argv equality (14 whole-list equalities on current bytes).
- **Reversibility**: cheap in bytes, not free in evidence — reverting line 30 restores today's behavior, but the composition and witness re-verification must be re-paid on the way back if the change is reverted after being recorded.

### 6. Side-by-side

| Axis | Option A — keep default on, document | Option B — default off, require opt-in |
|---|---|---|
| Observable gain | The default becomes a written, recorded choice; a reader can see and override it | Nothing is injected unless the caller asks; the default is conservative |
| Observable loss | The default still asks every deployment to preserve thinking | Every current deployment stops preserving thinking until it opts in |
| Behavior that stays | Script, argv word set, injection site, all pinned argv equalities | The flag, the injection site, `PRESERVED_THINKING=1`, the `="0"` arm |
| Risk | Low: docs + record only | Medium: one default moves, two verification surfaces fire |
| Reversibility | Trivial | Cheap to revert, expensive to re-prove |
| Code/script/test bytes | Zero | One script line + README + test constant/expectations |
| Composition debt | None | Fresh member re-verification + 25-member `seam-S6` recomposition |
| Recorded authority | `decisions.md` records the posture | `decisions.md` records the posture, and the composition record is re-based |

### 7. Implementation sequence and task outcomes

No planned task exists for either arm, so the answer creates one adhoc remediation row and no milestone.

**Option A — sequence**

1. `authority-registrar` records the answer and the default-on posture in `decisions.md`; no invariant is created.
2. One adhoc docs task extends `README.md`'s serve-script section: the flag, the env var, the default, and how to turn it off — to the standard the `--dump-format` and `--skip-tls-verify` examples already meet.
3. Done when `README.md` names both the flag and `PRESERVED_THINKING` with the `1` default and the `0` off-switch, and the decision record states the posture is chosen. Zero code, script and test bytes move; the hermetic suite is unchanged.

**Option B — sequence**

1. `authority-registrar` records the answer in `decisions.md`.
2. Change `serve_buster_ripper.sh:30`'s default to `0` and extend `README.md` with the flag, the env var, and the opt-in requirement.
3. Make `tests/test_serve_script_argv.py`'s `_THINKING` (line 62) conditional and re-pin every whole-list argv equality that assumed the default-on arm; the `PRESERVED_THINKING="0"` arm at line 217 already proves the suppression shape.
4. The assumption gate re-verifies the affected `CO:serve-script-env-to-argv/env-var-flag-crossing` members and recomposes all **25** `seam-S6` members on the new bytes.
5. Done when the script defaults off, the README documents the opt-in, every pinned argv equality reflects the new default, the composition is re-verified, and the hermetic suite is green.

**What neither arm does**: no milestone, no validation point, no QA author pass, no `roadmap.md` entry, no new invariant, and no change to the dump formats, the flush policy, the TLS posture, the header-parity or image-surface remediations, or any delivered task.

### 8. Validation points, milestone constituents, QA author passes

- **Option A**: no validation point. The change is prose; the existing hermetic suite is the only check, and it does not move.
- **Option B**: no new validation point either. The evidence is the existing hermetic `tests/test_serve_script_argv.py` witnesses plus the assumption-registry composition. No `MS-*` milestone is created or reopened; `MS-001`–`MS-004` stay delivered. No `QAP-*` author pass is owed, because no process-level or integration behavior changes.
- **Milestone constituents**: none. This round's answer belongs to no milestone.

### 9. Scoped invariant preservation

- **Option A — preserve.** `CO:serve-script-env-to-argv/env-var-flag-crossing` stays current at its recorded basis `sha256:cb7c0a5c82e5ac1b5f33738a61b582d925be4502a51a6ed1484e2ab415b60631`; no refinement, override, deprecation or removal. The eight `PI:*` planned invariants cover `dump_format` (six) and `upstream_tls` (two) only — none covers preserved thinking, so neither arm creates or edits a registry entry.
- **Option B — the same composition is disturbed.** Its `Invalidates-On` explicitly names "line 30's `PRESERVED_THINKING` default moving off `1` or line 42's guard changing". The record owes a fresh member re-verification and a 25-member `seam-S6` recomposition before it may be cited as current. No invariant **statement** is edited by either arm; the registry gains nothing.
- **No `OI:` entry, no `PI:` entry, no approved-override register entry** is touched by this round.

### 10. What this round does not re-open

Delivered milestones and `resolved` tasks; the three capped/deferred seam records (`S21` recomposition capped, `S22`/`S23` `SAFE`); the six accepted risks; the deferred default-format switch in `roadmap.md`; the QA harness contract (`US-009`/`US-010`); `FK-001`–`FK-010`; the `MS-001` boundary's reported review-object deviation; and the `US-012` default-format reading.

## User-Facing Landscape

**Where the session stands.** All four milestones are delivered and approved: the dump bulk-flush, the documented-format preservation, the collapsed formats, the flush triggers, the TLS opt-out, and the QA gates. `MS-001`'s boundary is closed and the session's plan is complete. One decision was left open because it is outside every task: the serve script's preserved-thinking default.

**The one thing left.** `serve_buster_ripper.sh` ships with preserved thinking **on** — every run of the documented serve path asks the model to keep its reasoning chain visible across turns — and `README.md` never mentions the flag or the variable. Both postures are legitimate; nothing recorded ranks them.

**Option A** keeps the default on and writes it down: the README gains the flag and the variable with the default, and the posture becomes a recorded choice. Nothing observable changes; every deployment keeps today's behavior.

**Option B** flips the default off and makes it an opt-in: nothing is injected unless the caller asks. Every current deployment stops preserving thinking after upgrade until it sets `PRESERVED_THINKING=1`, and the change owes re-verification of the serve-script composition and its pinned argv witnesses.

**What is not in question either way**: the `--preserved-thinking` flag stays available, the injection code is untouched, and no dump-format, flush, TLS, header or image behavior moves. **Cost of leaving it open**: the default keeps shipping undocumented — the status quo, not a new risk.

## Topic Queue

One entry per causal FK. `decisions.md` has exactly one `State: open` record; no extension-owned Fork exists.

### Entry 1 — `FK-011 · preserved-thinking serve default` — **READY** (wave 1)

- **FK and plain-language title**: `FK-011 · preserved-thinking serve default` — should the serve path keep preserved thinking on by default and become a documented, recorded choice, or default it off and require an explicit opt-in?
- **Wave candidate**: 1.
- **Dependency handles**: none. The answer changes no other queue entry, no task, and no milestone.
- **Predecessor / supersession handles**: none — this is the original record, not a successor. No prior FK asked this question; `FK-001`–`FK-010` are resolved or superseded and none mentions thinking.
- **Exact plan / source locators**:
  - `serve_buster_ripper.sh:30` (`PRESERVED_THINKING=${PRESERVED_THINKING:-1}`) and `:42` (`[[ "${PRESERVED_THINKING}" == "1" ]] && EXTRA_FLAGS+=(--preserved-thinking)`) — re-read on `HEAD 1f36d90`.
  - `buster_ripper/routes/messages.py:46-47` — the injection (`if config.PRESERVED_THINKING: _inject_preserved_thinking(body)`); the record cites the enclosing `:43-50` range.
  - `buster_ripper/main.py:107-109` — `--preserved-thinking`, whose help text is the only place the flag is named.
  - `README.md` (289 lines) — **zero** occurrences of `preserved.thinking` or `PRESERVED_THINKING`.
  - `tests/test_serve_script_argv.py:62` (`_THINKING`), `:150`, `:156`, `:161`, `:166`, `:217` (`PRESERVED_THINKING="0"` arm) — the pinned argv equalities.
  - `assumptions/compositions/seam-S6.md` — 25 members; `invariants/buster_ripper/assumption-claims.md:743` — the composition record; `invariants/index.md:302` — its index row.
  - `decisions.md` §`FK-011` — the owner-complete record.
- **Complete draft capsule**:

**FK-011 · preserved-thinking serve default**

**Decide:** Must the documented serve path keep preserved thinking on by default and become a documented, recorded choice — or default it off and require an explicit opt-in per invocation?

**Why now:** The branch's codex review found the serve path injecting preserved thinking into every forwarded `/v1/messages` request while the flag appears nowhere in `README.md`; `FK-011 · preserved-thinking serve default` is the session's only open record, and no planned task or milestone reaches it.

**Today:** `serve_buster_ripper.sh:30` reads `PRESERVED_THINKING=${PRESERVED_THINKING:-1}`; line 42 appends `--preserved-thinking` when it is `1`; `buster_ripper/routes/messages.py:46-47` injects `enable_thinking=true, clear_thinking=false` into every forwarded `/v1/messages` request; `README.md` never names the flag or the variable.

| Option | You get | You give up | To undo |
|---|---|---|---|
| A · keep default on, document it | Every deployment keeps today's behavior, and the default becomes a written, recorded choice in `README.md` | The serve path still asks every deployment to preserve thinking unless it opts out | Trivial — docs and a decision record only; no code, script or test byte moves |
| B · default off, require opt-in | Nothing is injected unless the caller asks; the serve path stops silently changing what every deployment requests | Every current deployment stops preserving thinking after upgrade until it sets `PRESERVED_THINKING=1`; owes the serve-script composition re-verification and its pinned argv witnesses | Revert line 30's default and re-pay the composition and witness debt |

**Recommended:** none: the recorded evidence establishes what the script does and what each arm costs, and does not rank the two postures.

**Not deciding:** The `--preserved-thinking` flag stays available either way; the injection at `messages.py:46-47` is not in question; no dump-format, flush-policy, TLS, header-parity or image-surface behavior moves.

**Answer:** FK-011: A | B | {your own wording}

**Detail**

**Where it sits:** Outside every planned task. The plan's twenty-two tasks and its four milestones carry no preserved-thinking clause; `T006 · upstream TLS certificate-verification opt-out` touches the serve script only through the `SKIP_TLS_VERIFY` boolean guard, and `T012 · preserve environment values as single argv words in the serve script` rewrote line 42's assembly mechanism without touching line 30's default. Answering this Fork opens one adhoc task and no milestone.

**Decision trail:** Finding **F-5 · preserved-thinking default** in [057-codex-review](057-codex-review.md) §E (`CLOSURE: question`, `Question required: yes`), re-derived by the assumption-decide pass that reserved `FK-011` on 2026-10-05 against `serve_buster_ripper.sh`, `buster_ripper/routes/messages.py` and `README.md`. The record is complete: question, two options, why-yours, if-wrong, affected tasks, recommendation `none`, `Presented: not yet presented`.

**Option A in full:** No script line moves; the serve script's argv word set is unchanged, so the `CO:serve-script-env-to-argv/env-var-flag-crossing` composition stays current at `sha256:cb7c0a5c…` and every pinned argv equality stays green untouched. The cost is the posture itself: preserved thinking remains the shipped default, now written down. Reversal is a docs edit plus a superseding decision record.

**Option B in full:** Line 30's default moves off `1`, which fires that composition's `Invalidates-On`; a fresh member re-verification and a 25-member `seam-S6` recomposition are owed before the composition may be cited as current. `tests/test_serve_script_argv.py`'s `_THINKING` constant becomes conditional, which row 201's `Invalidates-On` names as the trigger for re-verifying every pinned argv equality — 14 whole-list equalities on current bytes. The gain is that the default becomes conservative; the loss is that every current deployment changes behavior on upgrade.

**Downstream:** Nothing. No later queue entry exists, no planned task is gated, and no milestone verdict moves. The answer's only downstream consumer is the adhoc docs task (Option A) or the script + docs + witnesses task (Option B), plus the assumption gate for Option B.

- **Which later entries this answer can invalidate**: none.
- **Status**: **READY**.

## Fork Fidelity

| Requirement | Result |
|---|---|
| Verified change / question preserved | yes — the record's question is carried verbatim in substance: default-on vs default-off with an explicit opt-in |
| Source preserved | yes — [057-codex-review](057-codex-review.md) §E, re-derived from the code on `HEAD 1f36d90` |
| Displaced assumption preserved | yes — "preserved thinking is on by default and that is acceptable" is the unrecorded assumption the Fork exists to surface |
| Recommendation | **none** — preserved, and no lean is added |
| If-wrong cost | preserved in substance: A costs nothing observable; B moves a default, fires the composition's `Invalidates-On`, and owes the pinned-argv witness updates |
| Alternatives complete | yes — exactly two materially different admissible postures; removing the flag is a rejected candidate in the source packet and is not re-opened as an option |
| Hidden assumptions / contradictions | one **stale cost detail**, reported below, not resolved here |

**Reported to the FK owner, not resolved by Invite:** the record's `If-wrong` says "the four whole-list argv equalities in `tests/test_serve_script_argv.py` stay green untouched". On current bytes the file pins **14** whole-list equalities (lines 150, 156, 161, 166, 175, 182, 187, 192, 200, 207, 217, 251, 256, 262), because `T012 · preserve environment values as single argv words in the serve script` widened the witness after the record was written. The cost is **understated, not misdirected**: Option B still owes re-pinning every equality, and the ranking between the arms is unchanged. No option, recommendation, admission or `State:` field is changed by this observation.

## Validation Performed

| Claim | How verified | Result |
|---|---|---|
| Session identity / handle suffix | `session-identity.json` vs handle | matches (`9bf7afff5573`) |
| No parent coordination contract | `session-identity.json` | `parent_session_uid` absent — confirmed |
| Exactly one FK is `open` | every `State:` field in `decisions.md` | `FK-011` only — confirmed |
| `FK-011` is owner-complete | `decisions.md` §`FK-011` | question, owner, type, why-yours, trigger, source, if-wrong, affected tasks, two options, recommendation `none`, `Presented: not yet presented` — complete |
| The question is still true on current bytes | `serve_buster_ripper.sh:30`, `:42`; `messages.py:46-47`; `main.py:107-109`; `grep -ni "preserved.thinking\|PRESERVED_THINKING" README.md` | line 30 default `1`; line 42 guard appends; injection at `:46-47`; README **zero** hits — confirmed |
| Option A moves no code/script/test byte | read of `serve_buster_ripper.sh`, `README.md` scope in the record | confirmed — docs + record only |
| Option B's composition debt is real | `assumptions/compositions/seam-S6.md` (`Invalidates-On` names line 30's default) + `invariants/buster_ripper/assumption-claims.md:743` | 25 members, `SAFE`, basis `sha256:cb7c0a5c…` — confirmed |
| Option B's witness debt is real | `tests/test_serve_script_argv.py:62` `_THINKING`, `:150` unset-default arm | confirmed — `_THINKING` is unconditional today |
| No task or milestone is gated | `plan-summary.md` §Implementation Plan Summary + §Milestone Matrix; record's `Affected tasks: none` | confirmed — no row reaches it |
| No extension-owned Fork is open | `decisions.md` record inventory | confirmed — none |
| QA preflight capability coverage | `.agents/skills/qa-project/SKILL.md` declarations | all required capabilities declared — passes |
| No file was written outside this session's directory | file inventory of this pass | `iterations/111-interview.md` only |

## Dropped-Clause Sweep

**Not triggered.** This pass supersedes no definition: it adds one iteration report and changes no task, VP, goal, register entry, invariant statement, or roadmap row. The sweep's trigger is a wholesale replacement of a definition, and there is none.

## Questions Asked

**This round: 0.** Invite prepares the wave; it does not present and does not ask. `interview-sprint` presents the capsule. **No `FK-` or `US-` ID was reserved, filled, superseded or reused by this agent.**

## Gates Found Unmet (reported, not cured)

**Count reported to the orchestrator: `gates found unmet 1`.**

1. **No Decide report exists for this round.** The dispatch named only the skill, and `iterations/` holds no Decide report after [110-01](110-01-forward.md). Invite compiled the owner-complete `FK-011` record itself, which carries the admission disposition, options, evidence and costs; it did not re-author, re-admit or re-rank anything. Belongs to the orchestrator: if a fresh Decide pass is required for this round's shape, re-dispatch `decide` before Sprint. **Not cured.**

*(not a gate — informational)* The `If-wrong` witness count in `FK-011` is stale at "four"; the current file pins 14. Reported in §Fork Fidelity to the FK owner, substance and ranking unchanged.

## Next Step

1. **`interview-sprint`** persists and presents wave 1 — the single capsule above — and stops for one labeled answer. There is no user stop after Invite.
2. **`interview-reconcile`** records the answer, classifies its effects, and routes the adhoc task (Option A: docs + record; Option B: script + docs + witnesses + assumption gate).
3. **`authority-registrar`** projects the settled posture into `decisions.md`; no invariant is created.
4. If the answer reopens any delivered surface — it should not — that surface returns to `decide` before anything is stamped.

## Wave Index (appended by `interview-sprint`)

| Wave | Order | FK | Dependencies | Presented (UTC) | Capsule persisted | Report |
|---|---|---|---|---|---|---|
| 1 | 1 of 1 | `FK-011 · preserved-thinking serve default` | none | 2026-10-18T03:58:36Z | `decisions.md` §`FK-011` `Presented:` | [114-sprint](114-sprint.md) |

Wave 1 stops for one labeled answer. No further queue entry exists; `decisions.md` holds no other `State: open` record.
