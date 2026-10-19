# Iteration 072-01 — Forward (Phase-Boundary Evidence, MS-001)

## Orchestrator Prompt (verbatim)

```text
You are a subagent spawned by another session.
Session ID: 02.00-2026-09-13-message-dump-bulk-flush--9bf7afff5573
Task: T005
Mode: phase-boundary-evidence
Milestone: MS-001
Target base: main
Target object: 72e3a1c6861bf0d8de6317adf52289e5f6cfa4e1
Review object: dbd3072403cc5014b19be229725dbc974337658a
Close: milestone
```

Plus the dispatch's boundary statement: MS-001's gate `SEMANTIC-APPROVE` (zero findings) at
[071-yield-phase](071-yield-phase.md) was followed the same day by an external `codex-review` against
`main` @ `72e3a1c6…` which returned **NOT clean** — 3 × P1 + 1 × P2, all classified real — absorbed by the
closure-packet routing into `T015`/`T016` (ledgers `DC-CR2-F-1-r1`…`DC-CR2-F-4-r1`, packet copy at
[`evidence/codex-review-post-071-packet.txt`](../evidence/codex-review-post-071-packet.txt)), with `CR2-X-1`
returned to the collector lane as `malformed` and `CR2-X-2` accepted as inert wording. Persist that state
review-preserving at the Review object before the repair route advances `HEAD`; run no production code, alter none.
The record must also carry honestly: the review's headline names head `2be9783` (Checkpoint 141) while it executed
against the tree at `dbd3072` (Checkpoint 162) and reported a `407 passed` baseline matching that tree.

## Increment Brief

- Task and goal: persist the **not-yet-passed MS-001 milestone boundary** — 071's `SEMANTIC-APPROVE` gate plus the
  same-day CR2 external review and the closure-packet routing it triggered — as one review-preserving evidence
  checkpoint **before** the owning repair route (`T015`/`T016`) advances `HEAD`, and put the review's stale
  head-label on the record as a label defect rather than letting it stand. No implementation in this mode.
- Anchors touched: **none** — evidence mode adds no feature behavior; no Anchor Map row is exercised.
- Will NOT create: any new referent, plan task, QA gate, or invariant record (triad rule — the registry is
  Forward's to never move). `T015`/`T016` already exist as rows; this pass adds no task and reopens no scope.
- Reuse evidence: none applicable (no `Mode: NEW` surface approached; the creation gate is not reached).
- Inventions: none.

## Pre-Flight Checks

- **Plan approval gate: PASSED** — planning CLOSED at round 18 ([020-interview](020-interview.md), `US-015`); the
  session-level approval covers boundary invocations, the same ruling [033-01](033-01-forward.md),
  [036-01](036-01-forward.md) and [058-01](058-01-forward.md) applied.
- **QA contract consistency**: `qa-plan.md` exists. `MS-001`'s gate was authored at
  [070-forward-qa](070-forward-qa.md) and executed independently by this boundary's Yield Phase
  ([071](071-yield-phase.md) §Check 6/7 — `QAP-MS001` 10/10, four `VP-*` handles `pass`). This mode authors no gate
  test, inspects no gate test, and **executes no harness and no test command** — 070/071's figures are inherited
  evidence on the same tree (no-re-run rule), and the dispatch forbids running production code here.
- **Task-specific planned context**: the dispatch labels `Task: T005`, the milestone's owning-task label (same
  shape as 058-01's `Task: T006`). §Planned Task Context carries no live `T005` block — T005 was delivered and
  accepted long before this boundary. The `T015`/`T016` blocks this dispatch names belong to the **next
  implementation Forward**, the owning repair route; they are persisted untouched, neither rewritten nor executed.
- **Global invariant registry read**: not performed beyond scope confirmation — nothing is implemented, so no
  narrowed scope exists to read for. `git status --porcelain -- invariants` → empty: 8 `planned`, 0 `active`.
- **Dependency claims**: none encoded — no production or test line is written in this mode.
- **Pre-question gate**: no question, `ESCALATION`, `BLOCKED-APPROVAL` or `PLAN-GAP` arose → **no closure packet
  owed by this pass**. `FK-011` is the routing pass's open fork (Interview's to present), recorded and untouched
  here; `CR2-X-1`'s `malformed` return belongs to the collector lane, and this mode neither re-adjudicates it nor
  converts it into a question.

## Evidence-Mode Execution Record

Mode rule applied: *verify all dirty files are limited to session/plan/review routing artifacts, then create one
review-preserving checkpoint; do not require Yield Red CLEAN, do not alter findings, do not implement a repair, do
not mark the phase accepted.*

**Step 1 — snapshot identity.** Branch `refactor/package-structure`; HEAD
`dbd3072403cc5014b19be229725dbc974337658a` (Checkpoint 162) — **equal to the prompt's Review object**, so this
checkpoint's first parent is the reviewed commit. Target base re-read from the prompt:
`git rev-parse main` = `72e3a1c6861bf0d8de6317adf52289e5f6cfa4e1` = the prompt's Target object;
`git merge-base --is-ancestor 72e3a1c6… HEAD` → **true**; `git rev-list --count main..HEAD` → **191 ahead /
0 behind**. The base sync is settled upstream ([054-01](054-01-forward.md) `BASE-SYNC: unchanged`, re-verified by
[069](069-yield-phase.md) and [071](071-yield-phase.md) §Invocation Contract Verification); this mode integrates
nothing and merges nothing.

**Step 2 — dirty-file census.** `git status --porcelain` at read time returned five paths, all under
`.sessions/codify/02.00-2026-09-13-message-dump-bulk-flush--9bf7afff5573/` — the CR2 routing pass's output:

| # | Path | State | Classification | Verified |
|---|---|---|---|---|
| 1 | `evidence/codex-review-post-071-packet.txt` | untracked | review evidence — the producer packet, persisted verbatim by the routing pass with its own recheck notes appended (91 lines) | file read in full |
| 2 | `handoff-context.md` | modified | routing — §Current Status CR2 block, the `T015`/`T016` §Planned Task Context blocks, §Deterministic Closure Ledger `DC-CR2-F-1-r1`…`DC-CR2-F-4-r1`, §Closure-packet routing, §Accepted risks | full diff read (+252 lines) |
| 3 | `plan-summary.md` | modified | plan routing — `T015`/`T016` rows (`pending`) + their §Planned Task Context pointer stubs | full diff read (+18) |
| 4 | `quality-backlog.md` | modified | routing — `QB:routes-chat/post-commit-second-head` pointer refresh, `QB:cli-surface/…` amended for `CR2-X-2`, new `QB:routes-chat/raw-bytes-passthrough-when-unfixed` | full diff read |
| 5 | `roadmap.md` | modified | routing — §Chat route's post-commit second head: pointers refreshed, CR2 repro appended as confirmatory evidence, reviewer's prioritization flag recorded | full diff read |

**Result: PASS.** Two independent scopes prove no repair is mixed into the evidence:
`git status --porcelain -- . ':(exclude).sessions'` → **empty**, and
`git diff --stat dbd3072 -- buster_ripper tests serve_buster_ripper.sh pyproject.toml README.md` → **empty**.
HEAD's production and test tree is the exact tree 071 certified; `invariants/` is clean. No file under
`buster_ripper/` or `tests/` is created, edited, deleted or run by this pass.

**Step 3 — the review's stale head-label, established as a label defect and not a verdict defect.**

The packet's headline names head `2be9783` (Checkpoint 141 — T013 retry 1's `fix(chat)` commit). The review
**executed against the tree at `dbd3072`** (Checkpoint 162), and the baseline it printed is what proves it:

| Figure | Which tree it belongs to | Where the session records it |
|---|---|---|
| review headline head `2be9783` | hermetic **374 passed, 94.20%** on that tree | [067-01-yield](067-01-yield.md):109 (independent full-suite run on the T013 retry-1 bytes) |
| review baseline **407 passed** | hermetic **407 passed, 95.75%** — the post-070 line: `c7420ce` (Checkpoint 161) plus `dbd3072`'s docs-only delta | [070-forward-qa](070-forward-qa.md):168; [071-yield-phase](071-yield-phase.md):436 |

The two counts cannot describe the same tree: `git diff --stat 2be9783..dbd3072 -- buster_ripper tests` →
**12 files, +1568 / −279**, with `dump.py`, `image.py`, `routes/count_tokens.py`, `tests/image_bodies.py`,
`tests/test_image.py`, `tests/test_image_stability.py` and `tests/qa/test_qap_ms001.py` among them (T014's wave and
070's `QAP-MS001` authoring both landed after Checkpoint 141; [068-01-yield](068-01-yield.md):28 records the
intermediate **406 passed**). And the tree at `dbd3072` differs from the one 071 measured only in session records:
`git diff --stat c7420ce..dbd3072` → 7 paths, **every one** under this session directory
(`071-yield-phase.md`, `decisions.md`, `handoff-context.md`, `plan-summary.md`, `qa-plan.md`,
`evidence/INDEX.md`, `evidence/ms001-gate-and-upstream-adjudication.txt`). A run reporting `407 passed` ran on
`dbd3072`.

Disposition — **the discrepancy affects the label, not the verdict**, recorded as such:

1. Every one of the six items was **re-established against the bytes at `dbd3072`** by the routing pass before it
   was dispositioned (all four ledger rows carry `Repository state: git-commit:dbd3072…`; the site map in the
   packet's §ROUTING-PASS RECHECK NOTES — `chat.py:207-211`/`:215-220`, `:152`/`:155-159`,
   `messages.py:45-52`/`:158-160`/`:291-293`, `passthrough.py:48-49`, `chat.py:174-188` with `prepare()` at `:184`
   — is `dbd3072`'s, not `2be9783`'s, and its line numbers are what `T015`/`T016` cite). All six hold there; none
   was already-closed.
2. A review that ran on the current tree and mislabeled which commit it ran on has produced findings **about the
   current tree**. Nothing in the batch is weakened by the label, and nothing in the label is a reason to re-run
   or re-verify the batch.
3. The producer's wording is **kept visible and unaltered** — the packet copy is checkpointed byte-identical
   (its provenance block at lines 6-8 already states the headline/re-verification split; this pass adds the
   baseline arithmetic that settles which side of it is the label). The correction lives in the routing record
   (§Closure-packet routing in `handoff-context.md`, and the chronicle entry below), which is where a later reader
   will look for it.

**Step 4 — mode prohibitions honored:**
- **Yield Red CLEAN not required, not asserted**: no `YIELD-RED: CLEAN` report exists for MS-001 and this
  checkpoint does not substitute for one. 071 states it outright: "phase acceptance, chronicle and parent
  completion remain Yield Red's."
- **Findings not altered**: 071's verdict, the CR2 packet copy, and all six dispositions are checkpointed
  byte-identical — zero edits to any of them. §Step 3's correction is an addition beside the record, not a
  change to it.
- **Repair not implemented**: `T015` and `T016` remain `pending`; no source, test or README line changed. The
  `407`-vs-`374` reconciliation was performed by reading recorded evidence, not by running a test command.
- **Phase not marked accepted**: MS-001's milestone-gate verdict stays 071's `SEMANTIC-APPROVE`, and the
  **boundary is blocked, not passed** — `T015`/`T016` are named prework of the close in both blueprints
  ("prerequisite of the **MS-001 boundary close** — land before Yield Red's final semantic pass over this
  batch"), and after them the assumptions recheck → **Yield Red** → `phase-boundary-finalize` still have not run.
- **History preserved**: one commit on top of `dbd3072`. No merge, no squash, no rewrite, nothing pushed.

**Step 5 — bookkeeping added by this mode (additive only):**
- This iteration report.
- One §Closure-packet routing bullet + one §Current Status bullet in `handoff-context.md` recording the
  label-not-verdict finding and the boundary's persisted state.
- One `evidence/INDEX.md` row for the packet copy (newest first; `Disposition` left empty — that column is the
  user's keep/remove ruling, see Observations).
- One chronicle entry. The tail stops at T014's acceptance, so [069](069-yield-phase.md)'s MS-002 re-verdict,
  [070](070-forward-qa.md)'s `QAP-MS001` authoring, [071](071-yield-phase.md)'s MS-001 gate and the CR2 batch and
  its routing left **no** chronicle record — context resumption today reads as "T014 accepted, seams owed", with
  no sign of either gate or of the blocker. The entry is factual session history, not a finding, and claims no
  acceptance: the same reasoning by which 058-01 appended the MS-004 boundary entry.
- One §Current Phase line in `plan-summary.md`: MS-001 remains **GATED** at 071, and is now additionally
  **BLOCKED** at the CR2 batch, with the evidence locator. Per supersession-by-rewrite this is written in place
  rather than stacked as a blockquote.

**Observations flagged, not acted on** (mode forbids repair):
1. `evidence/INDEX.md` rows added by earlier passes — including the 2026-10-06 MS-001 and MS-002 rows — carry text
   in the `Disposition` column, which is the user's keep/remove ruling and must stay empty. This is the third
   boundary to flag it (036-01, 058-01, here), so it is a recurring pattern rather than a one-off; the
   session-completion disposition sweep owns it. Left byte-identical.
2. Context-budget overruns in files this checkpoint persists: `handoff-context.md` is at 3017 lines (cap 200) and
   `plan-summary.md` at 459 (cap 400). The lossless trim into `handoff-archive.md` is its own operation (the
   Checkpoint 122 precedent) and is not this mode's.

## Failure-Path Evidence

| Operation / condition | Expected outcome and authority | Handling / observation evidence | Disposition |
|---|---|---|---|
| A dirty file outside session/plan/review scope (production, test, `invariants/`, README) | Mode violation → STOP, never mix repair with boundary evidence | `git status --porcelain -- . ':(exclude).sessions'` → empty; `git diff --stat dbd3072 -- buster_ripper tests serve_buster_ripper.sh pyproject.toml README.md` → empty; `git status --porcelain -- invariants` → empty | safe |
| HEAD ≠ the Review object the findings were written against | Checkpoint would mis-attribute the review | `git rev-parse HEAD` = `dbd3072…` = the prompt's Review object; parent check after commit | safe |
| Stale headline adopted as fact, or the batch's verdict silently re-litigated | The dispatch's honest-record requirement: record the discrepancy as a label defect, keep the verdict | §Step 3: `407` ↔ `dbd3072`, `374` ↔ `2be9783`, 12-file production delta between them, all six sites re-derived at `dbd3072` by the routing pass | safe |
| "Fixing" the label by editing the persisted packet copy | Findings are the producer's; this mode alters none | packet read, never written; the correction is an addition in the routing record | safe |
| Findings altered or dispositions re-adjudicated while persisting them | Mode forbids it | No edit to 071, the packet, `decisions.md`, `DC-CR2-F-*-r1`, the `CR2-X-1` malformed return or the `CR2-X-2` accepted risk | safe |
| Repair implemented, or the boundary marked passed, under evidence mode | Mode forbids both | `T015`/`T016` stay `pending`; this report states the boundary is blocked; `FK-011` stays `open` | not-applicable (path not taken) |
| Boundary mistaken for `phase-boundary-finalize` because MS-001 is the session's last milestone gate | Finalize requires Yield Red CLEAN; running it here would fabricate acceptance | Only `create-checkpoint` invoked; no `YIELD-RED: CLEAN` demanded or produced; finalize left to the orchestrator | safe |
| Target base not integrated, silently "evidenced" over a stale base | Boundary evidence must sit on an integrated branch | `merge-base --is-ancestor 72e3a1c6… HEAD` → true, 191 ahead / 0 behind; sync settled `unchanged` by 054/069/071 | safe |
| Any test or app command run "to check the baseline" | Dispatch forbids running production code; Forward never runs the full suite; no-re-run rule | Zero commands executed beyond Git reads; the `407`/`374`/`406` figures are cited from 070/071/068/067 | safe |
| `create-checkpoint` unavailable or failing | No raw `git add`/`git commit` fallback → BLOCKED report | Tool present; checkpoint succeeded (id below) | safe |
| Worktree not clean after the checkpoint | Boundary evidence must land fully persisted | Post-checkpoint `git status --porcelain` → empty (verified below) | safe |

Exclusions by family: test failures, LNCP violations and coverage regressions have no trigger — no file with
executable content changed in this mode; visual-demo capture has no trigger — no user-visible behavior produced;
the R1–R5 adversarial surfaces have no trigger — no parser, writer, capture→act span, payload field or identity key
is authored here. No policy was invented anywhere.

## Mandatory Check Lines

- **Backlog pickup: none match touch-set** — the touch-set is session/plan/review artifacts only; every unchecked
  item targets `buster_ripper/`, `tests/`, `qa-plan.md`, `README.md` or `invariants/`, none of which is touched
  here. The CR2 routing pass added `QB:routes-chat/raw-bytes-passthrough-when-unfixed` (owner: T015's touch-set)
  and amended two existing rows; all three ride the repair route, not this checkpoint. Reviewed, not skipped.
- **Capacity gate: not applicable — no additive production surface** (no production edit of any kind in this mode).
- **LNCP scan: not applicable — no files with executable content modified.**
- **Visual demo evidence: not applicable — no user-visible behavior produced.** One row was appended to
  `evidence/INDEX.md`, for the review packet this checkpoint persists.

## Checkpoint Record

- Files staged: the five verified routing/evidence artifacts above + this report + `chronicle.md` (entry appended)
  + `evidence/INDEX.md` (row) + `handoff-context.md` and `plan-summary.md` boundary-state lines.
- Suite evidence: inherited, not re-run — [071-yield-phase](071-yield-phase.md) measured hermetic
  **407 passed @ 95.75%** and `QAP-MS001` **10/10** on `c7420ce`, and `dbd3072`'s delta from `c7420ce` is
  session-artifacts only, so the tree 071 certified is the tree this checkpoint sits on. No command was executed
  by this pass.
- Result: **Checkpoint 163** — `codify git create-checkpoint`'s own auto-numbering (last checkpoint 162, + 1),
  first parent `dbd3072403cc5014b19be229725dbc974337658a` = the Review object. `git status --porcelain` **empty**
  afterwards; `invariants/` untouched; branch topology unchanged (one commit, no merge, no rewrite, nothing
  pushed).

## Durable Signal

- **Boundary state as of Checkpoint 163**: MS-001's **milestone gate is `SEMANTIC-APPROVE`, zero findings**
  (071 — all seven integration checks PASS on the cumulative diff `72e3a1c6…HEAD`, `QAP-MS001` 10/10 independently
  re-run, `US-009` c2 **HOLDS** with `Consequence: none`, `decisions.md` left with no `PENDING` verification row),
  and the **boundary is BLOCKED** — the same-day external review routed **4 packets `absorbed`**
  (`DC-CR2-F-1-r1`…`DC-CR2-F-4-r1`) into **`T015`** (chat.py response-header parity + raw-relay fall-through +
  leak-free 502; wave anchor) and **`T016`** (messages.py/passthrough.py preserved-thinking guard + static error
  bodies) as **one CR2 wave with one local Yield**, with `F-4` absorbed **record-lane only** (pointer refresh +
  confirmatory repro, ownership ruling explicitly rechecked and unchanged), **1 `malformed`** returned to the
  collector lane (`CR2-X-1` — retry-body egress needs a request-log probe and a field-authority claim, not a
  question), and **1 `accepted`** as inert wording (`CR2-X-2`, README architecture tree, owner
  `QB:cli-surface/undocumented-flags-and-stats-db` → `roadmap.md` §RM-005). `FK-011` stays the only open Fork.
- **The review's head-label is stale, its verdict is not** (§Step 3). Two consequences a later reader needs:
  (a) `T015`/`T016` cite the **`dbd3072` site map**, not the packet's headline commit, and any pointer re-check
  must resolve against post-T014 bytes; (b) the batch's baseline is **407**, so a repair branch that cannot
  reproduce 407 hermetic tests is not on this tree — it is on the 374-era tree or somewhere else entirely.
- **Nothing reopens accepted `02.00` scope.** All four findings sit in branch-added surface (the aiohttp
  migration's chat/messages relay arms and the `PRESERVED_THINKING` injection), outside the plan's `T001`–`T005`
  scope, which is why neither the red passes nor the invariants watched them; `T015`/`T016` gate no milestone.
- **Carried corrections the repair route must honor** (both from the routing pass, not this one):
  `CR2-F-2`'s "no witness can produce a non-parseable upstream answer" is true of `tests/upstream_stub.py` but not
  of `tests/test_routes_integration.py:239-243`, which does — and pins the synthetic-502 shape the closure
  discards, alongside the `upstream.json.assert_awaited_once()` pin at `:179-192`; both witnesses are **replaced,
  not skipped**, by T015's done-when. And `CR2-F-4`'s premise that the session record understated the harm as a
  "partial body without a terminator" matches no record — the roadmap entry and the backlog pointer already state
  the second-head shape, so the repro is confirmatory and only the pointer staleness lands.
- **Still owed at this boundary** after `T015`/`T016` land: the assumption recheck on the wave's bytes →
  **Yield Red**, which owns final semantic acceptance, the `Delivered` chronicle entry and parent completion →
  `phase-boundary-finalize`. `CR2-X-1` waits with Forward Collect / the assumption-collect lane, and it blocks
  nothing in this wave.
