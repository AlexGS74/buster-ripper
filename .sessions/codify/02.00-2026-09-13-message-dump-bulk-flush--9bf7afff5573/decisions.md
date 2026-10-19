# Decisions — 02.00-2026-09-13-message-dump-bulk-flush--9bf7afff5573

> ## Registry elevation — landing note (interview round 18, 2026-09-17)
>
> **Every `Registry: none — no invariant registry exists in this repository` verdict written before 2026-09-17 was
> accurate when written**: the repository had neither `invariants/` nor `.invariants/`, and the elevation step is a
> **final-approval** step that had not run. The final-approval gate in round 18 created the non-dot `invariants/` tree
> and registered this session's approved set by handle-keyed intersection
> ([020](iterations/020-interview.md) §Round 18 §Registry Elevation).
>
> **Records whose registry destination is now landed** (each annotated in place): [`US-001`](#us-001) (the dump cluster's
> `OI:`/`PI:` set), [`US-009`](#us-009) (the QA-harness `qa-project` row), [`US-011`](#us-011) and [`US-013`](#us-013)
> (the two `PI:T006/upstream_tls/*` and `OI:buster_ripper.server/on_startup/...`), [`US-012`](#us-012) (its gating clause
> is the source of `PI:T002/dump_format/{default_format_is_documented, documented_format_is_retained}`),
> [`US-014`](#us-014) (the round-17 ratification's ten items).
>
> **Records whose `Registry: none` remains correct** — they state no invariant: `FK-001`…`FK-009`, `US-002`…`US-008`,
> `US-010`. Their verdicts are left byte-unchanged; this note is what a reader needs to reconcile them with a registry
> that now exists.

## US-001

State: recorded
Owner: collect
Reserved at: 2026-09-13T18:40:19.277254+00:00
User words: "make this repository when it runs in message interception and dumping mode to do collection into memory and dump to the disk in bulk reducing number of writes we don't care about making this super robust and if for example shutdown will kill the in memory in flight messages it's okay so at the same time please properly handle ctrl c seek term if i'm not mistaken and do a flush of that dump on a disk before terminating the process"
Source timestamp: 2026-09-13 (CODIFY session 02.00 bootstrap; user statement relayed verbatim in orchestrator initial-goal field)
Normative choice: When buster-ripper runs in message interception/dumping mode (`--dump-dir`), captured messages must be collected in memory and written to disk in bulk, reducing the number of write operations. Durability of buffered in-flight messages is explicitly NOT a requirement: if a hard shutdown kills buffered in-memory messages, that is acceptable. Ctrl-C (SIGINT) and SIGTERM must be handled so that the buffered dump is flushed to disk before the process terminates.
Factual claims:
  - Claim: The repository currently has no SIGINT/SIGTERM handler of its own and performs no dump flush during shutdown.
    Verification: assumptions: claim-14, claim-15 HOLDS
    Evidence: assumptions/verdicts/claim-14.md, assumptions/verdicts/claim-15.md
    Consequence: none — the absence is the precondition the plan builds on: shutdown handling belongs to aiohttp's `web.run_app` (native, verified), and the shutdown flush is T003's new wiring in `on_cleanup` (`E:T003/server/shutdown-flush-seam`).
  - Claim: The current dump path writes to disk once per intercepted request (a turn file write plus an index.txt append).
    Verification: assumptions: claim-4 HOLDS
    Evidence: assumptions/verdicts/claim-4.md
    Consequence: none — the two write-bearing operations per turn are the baseline `2N` the plan's write-count goal (`G-001`, `PI:T002/dump_format/format_does_not_change_write_count`) is measured against, and the documented layout's own per-turn writes are preserved (`US-012`).
  - Claim: "seek term" as written by the user means SIGTERM.
    Verification: assumptions — discharged (round 18)
    Evidence: plan-summary.md §Topic and `G-002`/`VP-T003-shutdown-flush`, via the round-18 record ([020](iterations/020-interview.md) §Round 18)
    Consequence: none — an interpretation of the user's own words, not a blind-checkable proposition; discharged by the user's ratification of the plan whose Topic names SIGTERM verbatim and by the settled session vocabulary for flush.
Impact:
  Plan: landed (round 18) — the normative choice is carried by `plan-summary.md` §Goals and Constraints and by `G-001`…`G-008`; the tasks that realize it are `T001`–`T005`. Status flipped from `pending` by the final-approval gate ([020](iterations/020-interview.md) §Round 18).
  Registry: landed (round 18) — this statement is the source of the dump cluster's registered invariants: `OI:buster_ripper.dump/dump_turn/{output_layout, sequential_diff_chain, turn_zero_is_full_body}` (all `Decision: preserve`) and `PI:T002/dump_format/*` in `invariants/buster_ripper/dump/invariants.md`. Status flipped from `pending` when the registry was created by the final-approval gate ([020](iterations/020-interview.md) §Round 18 §Registry Elevation).
  Roadmap: landed — `roadmap.md` carries the MVP floor clauses this statement defines (buffered collection, bulk write, SIGINT/SIGTERM flush) and the above-floor deferred entry that gates the default-format switch on the user's judgment of `MS-001`'s evidence. Status flipped from `pending` by the final-approval gate ([020](iterations/020-interview.md) §Round 18).
Legacy source: none
Supersedes: —
Superseded-by: —

## FK-001

State: resolved
Owner: decide
Reserved at: 2026-09-13T18:56:10.195016+00:00
Type: preference
Question: When should the buffered dump be written to disk — only when the process shuts down gracefully, or also whenever the pending buffer reaches a turn threshold?
Why-yours: "it is irreversible or expensive to reverse (data loss, public API surface, security)"
Trigger:
  Changed: none — preference
  Source: buster_ripper/dump.py:104-106 (index.txt is opened in append mode once per turn and no write happens until a flush exists, so the trigger choice sets the entire loss window); README.md:146-162 (index.txt is documented as the reading interface for a live session)
  Replaces: none
  Was: none
If-wrong: A shutdown-only trigger means a SIGKILL, crash, or OOM loses the whole session's dump and leaves index.txt nonexistent for the session's entire life, which is a larger loss than the "in flight messages" the user waived in US-001. A threshold trigger adds durability machinery the user said they do not care about, and a few extra index writes. Both are cheap to change later — the trigger is one branch — but the loss window is a behavior users only notice after they have lost a session, which is why it is asked rather than guessed.
Presented:
  Final question: "When should the buffered dump be written to disk — only when the process shuts down gracefully, or also whenever the pending buffer reaches a turn threshold?"
  Options:
    - "Threshold + shutdown" — Flush the buffer whenever it reaches a pending-turn threshold (order of tens of turns), and again during the graceful shutdown that Ctrl-C / SIGTERM already triggers. Bounded loss window: at most the unflushed tail. index.txt exists and is readable while the proxy runs, so the documented reading workflow keeps working. More code than shutdown-only.
    - "Shutdown only" — Write nothing until a graceful shutdown. Fewest total writes (N + 1 for N turns) and zero disk I/O on the request path for the whole session, the maximum relief for a latency-sensitive proxy. But a SIGKILL / crash / OOM loses the entire session's dump — a larger loss than the "in flight messages" you already accepted — and index.txt does not exist at all while the proxy runs.
  Recommendation: none
  Timestamp: 2026-09-13T19:00:32Z
  Report: iterations/004-interview.md
Resolution:
  User words: "Okay, good points. We need to have both time-based and data volume-based flash frequencies. Do you think it's feasible to have both?"
  Source timestamp: 2026-09-13 (interview round 1; user answer relayed via the orchestrator's `User response:` field)
  Normative choice: The buffered dump is written to disk on BOTH a time interval and a data-volume threshold, in addition to the graceful-shutdown flush. Neither trigger replaces the other; the shutdown flush remains required. The presented option set ("threshold + shutdown" vs "shutdown only") was incomplete and the user's answer is a superset of its first position plus a time dimension.
  Factual claims: none
    # The user's message states a requirement and asks a question; it asserts no falsifiable
    # proposition. The feasibility answer it asked for is agent-originated and is recorded in
    # iterations/004-interview.md §Feasibility Answer with its source locators
    # (buster_ripper/server.py:25-34 and :60-72; buster_ripper/compaction.py:28-...).
    # The timer entity's Mode: NEW row still requires its own complete RB-* Reuse Evidence row
    # under the Reuse-Before-Build Evidence Gate.
Impact:
  Plan: plan-summary.md — trigger dial gains a third position (both triggers + shutdown); Options table, Goal Definition row G-001 (write-reduction criterion), Task List T003 (gains timer coroutine + app-state key + cancellation + tests), Milestone Matrix MS-002, Validation Point VP-T002-write-count-reduced (target re-derived in terms of flush count F), and the coverage headroom constraint all require replan. Dispatched as REPLAN-INCREMENT (flush-policy cluster).
  Registry: none — no invariant registry exists in this repository; the flush trigger policy is a policy choice, not an invariant.
  Roadmap: pending
Supersedes: —
Superseded-by: —

### Planning note (not a factual claim)

Decide's recorded call in `handoff-context.md` — *"Resolved from code: no background
flush timer is needed"* — is **superseded by user authority**, not falsified by
source. The reasoning behind it is visible in the landscape: `US-001`'s "we don't
care about making this super robust" was read as "the loss window does not
matter", so only *when to write* was considered, never *how stale a buffered turn
may get*. The requirement the user has now stated is about **staleness** (keeping
`index.txt` readable during a live session), which a durability framing cannot
express. Recorded here so no downstream agent re-derives the exclusion.

### Interaction the plan must now carry

The time trigger trades against the change's own stated purpose. For N turns, the
documented layout costs `N + F` writes versus `2N` today, where `F` is the number
of flushes; a collapsed layout costs `F`. A short interval drives `F` toward `N`
and the documented-layout reduction toward zero. The interval is therefore not a
free parameter — it sets both the staleness bound and how much of the write
reduction survives. This is why the interval and threshold values are routed to
decide for real option analysis rather than defaulted.

## FK-002

State: superseded
Owner: decide
Reserved at: 2026-09-13T18:56:18.521098+00:00
Type: ratification
Question: Preserving the documented per-turn dump layout caps the write reduction at about 50% (2N writes become N+1 for N turns). Do you accept that ceiling and keep the documented layout, or collapse the session dump into a single append-only file to reach roughly 100%?
Why-yours: "it is irreversible or expensive to reverse (data loss, public API surface, security)"
Trigger:
  Changed: Preserving the documented output layout caps the write reduction at ~50%, not ~100%. Only the per-turn index.txt append collapses; turn_NNN.diff files are one file per turn and cannot be merged without changing the layout.
  Source: README.md:146-162 (the layout is documented as the reading interface); buster_ripper/dump.py:82 and :95 (one write_text per turn file) and :104-106 (one index open+write+close per turn); iterations/002-organize.md §Goal-shape findings #1
  Replaces: "in bulk" was assumed to deliver a large write reduction while the documented layout is preserved.
  Was: plan-summary.md §Goal Definition Matrix, row G-001 (unquantified: "file-write operations < 2N")
If-wrong: Preserving the layout ships a change that delivers roughly half the reduction the user's words suggest they expect. Collapsing instead reaches ~100% but changes a documented output contract users already read and destroys the per-turn diff artifacts the tool exists for, requiring a README rewrite and an explicit override of the layout and diff-chain invariants — and this repo has no invariant registry to make that override checkable. The two outcomes are materially different products, not two sizes of the same one.
Presented:
  Final question: "Preserving the documented per-turn dump layout caps the write reduction at about 50% (2N writes become N+1 for N turns). Do you accept that ceiling and keep the documented layout, or collapse the session dump into a single append-only file to reach roughly 100%?"
  Options:
    - "Keep documented layout (recommended)" — turn_000.json, turn_NNN.diff and index.txt stay exactly as documented. Each flush writes the pending turn files and then appends the index once, so N turns go from 2N writes today to N + (number of flushes): at best N+1, i.e. roughly a 50% reduction. No README change, no change to the reading workflow, and the per-turn diff artifacts stay directly observable on disk. Choosing this accepts that the reduction is well under 100% and shrinks further the more often the buffer flushes.
    - "Collapse to one file" — append every pending turn record to a single per-session file: one write per flush regardless of turn count, so N turns go from 2N writes today to (number of flushes), up to roughly a 100% reduction. But the per-turn diff files and index.txt stop existing on disk, README.md:146-162 must be rewritten, diffs become a read-time concern, and two session invariants (EI:.../output_layout, EI:.../sequential_diff_chain) must be explicitly overridden via a new Approved Override Register entry plus a fresh RB-* Reuse Evidence row.
  Recommendation: preserve the documented layout and accept the ~50% ceiling
  Timestamp: 2026-09-13T19:08:02Z
  Report: iterations/004-interview.md
Resolution:
  User words: "Another good point, we need to change the flash format. So our bulks would be single write, not a bunch of writes which would kill the purpose and any benefit of saving SSD."
  Source timestamp: 2026-09-13 (interview round 2; user answer relayed verbatim via the orchestrator's `User response:` field)
  Normative choice: The dump payload format changes so that one bulk flush is ONE write. The documented per-turn layout is NOT preserved — `turn_000.json`, `turn_NNN.diff` and `index.txt` as separate per-turn files are displaced by a collapsed per-session artifact written once per flush. The user's stated reason is that a flush emitting a write per turn would defeat the purpose of the change (sparing the SSD). This is a choice against the recorded recommendation, not a falsification of it: the recorded basis for the recommendation (a preserved layout caps the reduction at ~50%) still holds; the user declined the ceiling.
  Factual claims:
    - Claim: Under the documented per-turn layout, one flush of K pending turns costs more than one file-write operation — one write per turn file plus the index append — so a preserved layout cannot deliver a single write per bulk.
      Verification: assumptions: claim-4 HOLDS
      Evidence: assumptions/verdicts/claim-4.md
      Consequence: none — the settled shape S-1 retains the documented layout as the default exactly because each turn costs its two writes there; the one-write-per-bulk reduction rides the collapsed formats (`2N` → `F`), as disclosed and ratified ([`US-014`](#us-014)(c)).
    - Claim: Reducing the number of file-write operations reduces SSD wear (the benefit the user's rationale invokes).
      Verification: assumptions — no row (not source-checkable in this repository)
      Evidence: iterations/007-collect.md (grounded research: wear is bytes × WAF, not write-call count; a count proof is a proxy proof)
      Consequence: none — the record is `superseded` by [`US-012`](#us-012), and `G-008`'s arithmetic rests on M1/M2 (M4 explicitly not the proof; M3 the real axis and unmeasurable in-repo); the switch trigger is the user's judgment of the `MS-002` comparison record.
    - Claim: "flash format" in the user's message means "flush format".
      Verification: assumptions — discharged (round 18)
      Evidence: plan-summary.md §Topic and `G-002`/`VP-T003-shutdown-flush`, via the round-18 record ([020](iterations/020-interview.md) §Round 18)
      Consequence: none — an interpretation of the user's own words, not a blind-checkable proposition; discharged by the user's ratification of the plan whose Topic names the flush ("Ctrl-C/SIGTERM before exit") and by the settled session vocabulary for format.
Impact:
  Plan: plan-summary.md — the payload/format cluster requires an INCREMENTAL replan (dispatched this turn as REPLAN-INCREMENT). Re-derived or displaced: §Options (Option 3 is the pick; the "recommend preserve" framing is retired), §Goals and Constraints (write-reduction arithmetic becomes `2N → F`), Goal Definition G-001 (outcome, "Met when", failure witness A/B), Anchor Map (the `dump index` row's contract and the `bulk flush` row's payload; new rows for the collapsed-format writer and its read path, each `Mode: NEW` requiring a complete `RB-*` row), §Reuse Evidence (RB-001 must be re-run for the collapsed format), §Task List (T002/T003 re-derived; a README rewrite becomes a task), Validation Point Matrix (VP-T002-layout-preserved and VP-T002-diff-chain are invalidated and need replacements), Behavior Disposition Matrix (B-001, B-002), §Open Questions item 2. `FK-A`'s `If-wrong` framing (`½·(1 − F/N)`, reaching zero at `F = N`) is documented-layout-specific and is now wrong — decide must revise it before `FK-A` is authored/presented.
  Registry: none — no invariant registry exists in this repository. The override of `EI:.../output_layout` and `EI:.../sequential_diff_chain` must be carried by a session-local Approved Override Register entry produced by the replan.
  Roadmap: pending
Supersedes: —
Superseded-by: US-012

### Supersession note (019-decide, 2026-09-16)

`Superseded-by` names a **`US-` record, not an `FK-`**, and that is deliberate: the successor came from the
user directly in interview round 14, no question was escalated, and no successor `FK-` was authored to hold it.
`US-012` is the record that actually supersedes the clause above.

- **What is superseded:** the `Resolution.Normative choice`'s **ordering** — "The documented per-turn layout is
  NOT preserved", i.e. the layout being displaced by the same change that introduces the collapsed artifact.
  `US-012` records the opposite ordering: the documented layout is not deleted until the collapsed format is
  demonstrated better and the user has judged it worth switching to.
- **What is NOT superseded:** everything else in this record. The `User words`, `Presented` text,
  `Factual claims` and `Impact` above are what the user said and were answered in round 2. A later revision
  does not erase them, and the superseding pass overwrote none of them.
- **The one clause that survives intact and is load-bearing elsewhere:** *one bulk flush is ONE write*. `US-012`
  does not retract it; it is the premise the payload shape is derived from, and it remains in force.
- **The residual obligation of `Impact.Roadmap: pending` is discharged by `roadmap.md`**, not here: the element
  this pick's ordering did not take — displacing the documented layout — is now **deferred, not dropped**, with a
  named owner and trigger in the roadmap entry *"Switch the default dump format"*.
- **The `### Revision under review` pointer below is retained** as the provenance of the revision; it is no
  longer the only guard against a reader applying the revised clause, since `State: superseded` is now visible
  at the top of the record.

### Provenance note (not a factual claim)

Two framing details are recorded so no downstream agent treats either as current:

1. The `Presented` text above still carries the pre-refresh arithmetic ("2N writes
   become N+1 for N turns"), while `iterations/004-interview.md` §Question
   Refinement: Payload layout describes a refreshed `N + F` / `F` framing. Both
   framings present the same two options and the user's choice is unaffected by
   the difference. Neither number is current: under the resolution above the
   collapsed layout costs `F` writes for N turns.
2. The recommendation ("preserve the documented layout") was not taken. The
   factual basis recorded for it is untouched; only the normative choice flipped.

### Revision under review (round 14 — the current authority on this question)

⚠ **The `Resolution.Normative choice` above is no longer the current authority.** On 2026-09-16 (interview
round 14) the user revised it, verbatim: *"Right, but deleting something before we implemented something
better is premature."* / *"Old behavior should stay unaffected until we prove a new one is better and it's
worth switching to."* The statement is recorded canonically as [`US-012`](#us-012).

What this changes and what it does not:

- **Changed:** the *ordering*. The documented per-turn layout must not be deleted by the change that
  introduces the collapsed artifact; the new behavior is implemented and proven better first, and only then
  does the old one get displaced. The `Resolution` above records the opposite ordering, so a reader must not
  apply it as current.
- **Unchanged:** the record itself. The `User words`, `Presented` text, `Factual claims` and `Impact` above
  are what the user said and were answered in round 2; a later revision does not erase them, and this pass
  overwrote nothing.
- **Recorded here now (019-decide):** `FK-002` is `State: superseded` with `Superseded-by: US-012` — see the
  §Supersession note above. `decide` ran the deterministic closure gate on the reading this revision left open
  and returned `CLOSURE: absorbed`, so **no successor FK was reserved or filled**; a question the gate absorbs
  is not a fork. This note therefore remains the *provenance* of the revision rather than its only guard.

### Successor question this answer creates (owned by decide, not authored here)

The resolution fixes the *requirement* (one write per bulk flush) but not the
*format*. Materially open, and user-visible: whether `index.txt` survives as a
file or is folded into the collapsed artifact; whether the per-turn unified
diffs remain readable at all and who reconstructs them at read time; whether the
artifact is append-only or rewritten per flush; and what replaces the documented
reading workflow (`README.md:146-162`). Per the Fork rules, Interview does not
allocate an ID for these — decide reserves and fills the successor FK(s), and
this record is their trigger.

## US-002

State: recorded
Owner: collect
Reserved at: 2026-09-13T20:07:12.590084+00:00
User words: "That's another good point. Can you do research while we're waiting for organize and find the best way to deal with issues like this to maximize SSD longevity? Because this kind of payload can be really harmful. One of the options we can use maybe is using a database. Those will be, how to say it, they can be still write heavy, but they will not write file by file."
Source timestamp: 2026-09-13 (collect research dispatch; user statement relayed verbatim in the orchestrator's user-context field)
Normative choice: Research (including external sources) the storage approaches that minimize SSD wear for this session's write-heavy message-dump payload — file-per-turn versus bulk-append versus a database (e.g. SQLite) — and report findings relevant to the current payload/flush design. Maximizing SSD longevity is the stated objective; a database is one candidate option to evaluate, not a settled choice. This is a research directive, not a change to the settled `FK-001`/`FK-002` decisions.
Factual claims:
  - Claim: "this kind of payload can be really harmful" — i.e. the message-dump write volume is large enough to materially shorten SSD life.
    Verification: assumptions — no row (not source-checkable in this repository)
    Evidence: iterations/007-collect.md (grounded research: wear is bytes × WAF; the write-call count is a proxy)
    Consequence: none — `G-008`'s arithmetic rests on M1/M2 and M4 is explicitly not the proof; the roadmap switch trigger is the user's judgment of the `MS-002` comparison record, not the wear propositions.
  - Claim: "they can be still write heavy, but they will not write file by file" — i.e. a database avoids per-file writes while remaining write-heavy.
    Verification: assumptions — no row (not source-checkable in this repository)
    Evidence: iterations/007-collect.md (grounded research: SQLite is not a wear win)
    Consequence: none — the database option was evaluated (iteration 007) and not adopted; `G-008`'s arithmetic rests on M1/M2, not on the database comparison.
Impact:
  Plan: landed (round 18) — findings feed the successor FK (`G-006`) and the payload/flush design in plan-summary.md §Options/§Goals and Constraints; no decision is changed by this record. Status flipped from `pending` by the final-approval gate ([020](iterations/020-interview.md) §Round 18).
  Registry: none — this record states no invariant; the registry created in round 18 holds only the dump-format and upstream-TLS properties.
  Roadmap: none
Legacy source: none
Supersedes: —
Superseded-by: —

## FK-003

State: superseded
Owner: decide
Reserved at: 2026-09-13T20:19:15.928143+00:00
Type: preference
Question: What should each turn's record in the collapsed dump artifact contain, so a reader can still locate a changed turn and see what changed about it?
Why-yours: "it is irreversible or expensive to reverse (data loss, public API surface, security)"
Trigger:
  Changed: `FK-002` fixed the requirement (one bulk flush is ONE write) and displaced the documented per-turn layout, but left undetermined what a record *contains*. `G-006` (the dump stays readable) is OPEN with four alternate readings, and no repository evidence excludes any of them.
  Source: README.md:134-135 ("useful for identifying remaining cache busters after normalization") and README.md:146-162 (the documented reading workflow); buster_ripper/dump.py:77-106 (what is computed per turn today — `_render_body` output at `:77`, the unified diff at `:87-93`, the summary line at `:84`/`:98-101`); iterations/006-organize.md §Successor Question Scope (the scope-first sweep that settled `index.txt`, append-only, and no-tool-required from the record, leaving this one open).
  Replaces: "displacing the documented layout leaves the dump's readable content unchanged."
  Was: plan-summary.md §Goal Definition Matrix row `G-006` (OPEN, alternate readings a–d); iterations/006-organize.md §Successor Question Scope.
If-wrong: The artifact is a **documented output surface** — `README.md:146-162` presents the dump's contents as the interface a human reads after a session, and `T005` rewrites that section, while the Approved Override Register records the replacement contract. So the answer fixes a user-read contract, not an internal detail: picking diffs when bodies were wanted leaves every turn's full text recoverable only by replaying the chain; picking bodies when diffs were wanted writes substantially more bytes per session, which is the exact quantity the user's stated SSD rationale (`FK-002`) targets; picking the machine record adds a viewer component this plan has never scoped and moves the documented purpose behind a tool. Each is cheap to change in code before release and another contract change after, which is why it is asked rather than defaulted.
Presented:
  Final question: "What should each turn's record in the collapsed dump artifact contain, so a reader can still locate a changed turn and see what changed about it?"
  Options:
    - "Per-turn diffs (change-focused)" — The artifact carries the same change information the tool produces today: turn 0 embeds the full request body, every later turn embeds the unified diff of the rendered request against the previous turn, and each turn contributes one summary line. Closest to today's reading experience and the fewest bytes written of the three. A turn's full text is recoverable only by replaying the diff chain from turn 0.
    - "Per-turn full bodies (self-contained)" — Every turn's record embeds the complete rendered request, so each record stands alone and any turn can be read directly with no replay. Writes substantially more bytes per session (every turn's rendered text instead of just its changes), and the change between two turns is visible only by comparing their records.
    - "Machine record + viewer tool" — One JSON object per turn, appended one per line, read with `jq`. The most compact and the most scriptable, but the documented `cat`/`grep`/`less` workflow stops working, so a small viewer/reconstruction command must be built and the README documents a tool instead of a format. Adds a new component to the codebase that this plan has not scoped.
  Recommendation: none
  Timestamp: 2026-09-13T20:19:15Z
  Report: iterations/008-decide.md (authored); iterations/009-interview.md (presented, 2026-09-13T20:32:14Z)
Resolution:
  User words: —
  Source timestamp: —
  Normative choice: —
  Factual claims: —
Impact:
  Plan: pending — PLAN-SELECTED writes the answer onto: §Goal Definition Matrix `G-006` (the chosen reading), §Anchor Map (the `collapsed dump artifact` row's record serializer; option 3 additionally adds a `Mode: NEW` viewer row requiring its own `RB-*` row), §Task List `T002` (record builder) and `T005` (README), §Validation Point Matrix (`VP-T002-change-readable`, `VP-T002-turn0-available`), §Behavior Disposition Matrix (B-001, B-002, B-007), §Approved Override Register (the three `OI:` replacement statements), §Invariant Reconciliation (the three `EI:` dispositions)
  Registry: none — no invariant registry exists in this repository
  Roadmap: pending
Supersedes: —
Superseded-by: FK-004

### Supersession note (not a factual claim)

Superseded **unresolved** — the user did not answer this question. They responded to its
presentation with three statements (`US-003`/`US-004`/`US-005`) that extend the option space
(a direction this set does not contain) and expose an undefined term ("full"). Per the Fork
rules a materially changed question is carried by a successor FK rather than by editing this
one's `Presented` text, which is preserved above exactly as it was shown. Nothing here was
answered, so nothing here is reversed; `FK-004` is this question's successor and records
`Supersedes: FK-003`.

## US-003

State: recorded
Owner: decide
Reserved at: 2026-09-13T20:56:37.564207+00:00
User words: "Would it be practical to keep current D format and just create a viewer in small tui?"
Source timestamp: 2026-09-13 (planning round 2, user response to the `FK-003` presentation; relayed verbatim in the orchestrator's user-provided-context field)
Normative choice: none — this is a question addressed to decide, not a stated preference. It asks decide to evaluate a candidate direction: keep the current dump content format and add a small TUI viewer, instead of choosing among `FK-003`'s three presented options. The candidate direction it names is what `FK-004` is authored from; no decision is made by this record.
Factual claims:
  - Claim: The current dump output format can be preserved while the settled bulk-flush change lands (the premise embedded in "keep current D format").
    Verification: plan (VP-T002-documented-format-preserved)
    Evidence: plan-summary.md §Options, settled by [019](iterations/019-decide.md) §Deterministic Closure Gate
    Consequence: none — realized by the settled shape S-1 (the documented layout retained as the default format, the collapsed formats added beside it); arbiter `VP-T002-documented-format-preserved`, validated at `MS-001`.
      # decide's reading this pass (iterations/010-decide.md §Answer 1): the claim HOLDS for the
      # artifact's *content* format (rendered-text sections, per-turn diffs, summary lines) and
      # DOES-NOT-HOLD for its *file layout* (turn_000.json / turn_NNN.diff / index.txt), which
      # `FK-002` displaces. The user chose `FK-002` knowing "our bulks would be single write, not a
      # bunch of writes", so the content reading is the grounded one; the file-layout reading would
      # re-open `FK-002`. Routed to the assumptions pass for an independent verdict.
  - Claim: "small tui" denotes a small terminal user interface viewer for the dump artifact, not a web UI or a long-running service.
    Verification: assumptions — moot (FK-007)
    Evidence: decisions.md#us-007, decisions.md#fk-007
    Consequence: none — the statement asked for a viewer; `FK-007` resolved to option 1 (the full-text encoding is the addition, no tool), so no tool ships and the term denotes nothing in the shipped plan.
Impact:
  Plan: pending — the candidate direction becomes option 3 of `FK-004`; the premise resolution is recorded in iterations/010-decide.md §Answer 1 and constrains that option's text
  Registry: none — no invariant registry exists in this repository
  Roadmap: pending
Legacy source: none
Supersedes: —
Superseded-by: —

## US-004

State: recorded
Owner: decide
Reserved at: 2026-09-13T20:57:02.586142+00:00
User words: "Could it be it depends on the definition of what full is?"
Source timestamp: 2026-09-13 (planning round 2, user response to the `FK-003` presentation; relayed verbatim in the orchestrator's user-provided-context field)
Normative choice: none — a question addressed to decide. It asks whether the trade-off between `FK-003`'s "per-turn diffs" and "per-turn full bodies" options depends on what the word "full" denotes.
Factual claims:
  - Claim: The diffs-versus-full-bodies trade-off depends on the definition of "full".
    Verification: assumptions — absorbed (F-6)
    Evidence: plan-summary.md §Notes F-6, settled by [019](iterations/019-decide.md)
    Consequence: none — "full" is explicitly defined in the plan: turn 0's raw normalized JSON body **and** its `_render_body` rendering, in every shipped format (`EI:.../turn_zero_is_full_body` refine; `VP-T002-turn0-available`).
      # decide's reading this pass (iterations/010-decide.md §Answer 2): it HOLDS for the
      # readability axis and only weakly for the byte axis. Today "FULL" means the *raw normalized
      # request body* (`dump.py:82` writes `json.dumps(body)` as turn_000.json) while every later
      # turn's record is a diff of the *rendered text* (`dump.py:77`, `:87-95`) — so the baseline and
      # the chain are already in two different representations, and hand-replaying the chain needs the
      # tool's own `_render_body`. Bytes differ by ~4% on the real dump (93,886 bytes raw JSON vs
      # 88.2 KB rendered for turn 0). Routed to the assumptions pass.
  - Claim: `_render_body` output is a lossy projection of the request body, so "full rendered text" and "full request body" are not interchangeable.
    Verification: assumptions: claim-9 HOLDS
    Evidence: assumptions/verdicts/claim-9.md
    Consequence: none
      # decide's reading this pass (iterations/010-decide.md §Answer 2): HOLDS from source —
      # `dump.py:27-58` emits only `[tools]`, `[system]` and `[message i / role]` sections, so every
      # other top-level body key (e.g. `model`, `max_tokens`, `temperature`, thinking config) is absent
      # from the rendered text. Routed to the assumptions pass.
Impact:
  Plan: pending — the definition of "full" is made explicit inside `FK-004`'s option text; `VP-T002-turn0-available` and `EI:.../turn_zero_is_full_body` acquire a definitional dependency on it
  Registry: none — no invariant registry exists in this repository
  Roadmap: pending
Legacy source: none
Supersedes: —
Superseded-by: —

## US-005

State: recorded
Owner: decide
Reserved at: 2026-09-13T20:57:02.799506+00:00
User words: "if it's possible, I would be interested in improving readability for a human."
Source timestamp: 2026-09-13 (planning round 2, user response to the `FK-003` presentation; relayed verbatim in the orchestrator's user-provided-context field)
Normative choice: Improving the readability of the dump artifact for a human reader is a desired outcome of this session, conditional on its being feasible. It is an outcome the user wants, not a selection among `FK-003`'s options: it excludes no option that keeps a human-readable artifact, and it does not by itself decide between change-focused and self-contained records, nor whether a viewer ships.
Factual claims: none
  # "if it's possible" is a condition on an agent-answerable feasibility question, not a factual
  # assertion. decide's answer is recorded in iterations/010-decide.md §Answer 1 (the viewer is
  # feasible: stdlib `curses` needs no new dependency, `difflib` is already imported at `dump.py:5`,
  # and the reconstruction is mechanical) and §Answer 3 (what it costs).
Impact:
  Plan: pending — constrains `FK-004`'s option text and, if a viewer option is chosen, re-derives `G-006`'s "without additional tooling" clause and adds a `Mode: NEW` viewer row plus its `RB-*` row
  Registry: none — no invariant registry exists in this repository
  Roadmap: pending
Legacy source: none
Supersedes: —
Superseded-by: —

## FK-004

State: superseded
Owner: decide
Reserved at: 2026-09-13T20:57:03.017002+00:00
Type: preference
Question: "What should a session's collapsed dump artifact hold, and how should a human read it — a change-focused log read with plain text tools, a self-contained log of every turn's full text, a change-focused log plus a small terminal viewer, or a machine record plus a viewer?"
Why-yours: "it is irreversible or expensive to reverse (data loss, public API surface, security)"
Trigger:
  Changed: The user answered `FK-003`'s presentation with three statements (`US-003`, `US-004`, `US-005`) rather than a pick. They (a) asked whether keeping the current dump format and adding a small TUI viewer is practical — a direction `FK-003`'s option set does not contain, because its only viewer option replaces the human-readable artifact with a machine record; (b) asked whether the diffs-versus-full-bodies trade-off depends on the definition of "full", which `FK-003`'s option text leaves undefined; and (c) stated a normative interest in improving human readability. Two additional facts were verified this pass and are not in `FK-003`'s trigger: the artifact's baseline and its diff chain are already in two different representations, and the plan already requires the raw body.
  Source: README.md:146-162 (the documented layout and "Reading the index"); buster_ripper/dump.py:82 (`turn_000.json` is `json.dumps(body)` — the **raw** normalized request body) versus dump.py:77,87-95 (every later turn is a unified diff of `_render_body(body)` output); buster_ripper/dump.py:27-58 (`_render_body` emits only `[tools]`, `[system]` and `[message i / role]` sections, so `model`, `max_tokens`, `temperature` and every other top-level key are absent from the rendering); plan-summary.md §Validation Point Matrix `VP-T002-turn0-available` ("Turn 0's full normalized request body is present in the artifact and recoverable from it"); command output over the real dump the user pointed at — `/Volumes/data/buster-dumps/c451c754ae4f` holds 137 turns, `turn_000.json` = 93,886 bytes, all `turn_*.diff` = 5,785,285 bytes, `index.txt` = 7,085 bytes, and the rendered text stays ≈1,770-2,160 lines across the session (`turn_001.diff` first hunk `@@ -1766,3 +1766,31 @@`; `turn_137.diff` last hunk `@@ -2141,6 +2141,19 @@`); `pyproject.toml:5-8` (dependencies are `aiohttp` + `click` only); `dump.py:5` (`difflib` already imported).
  Replaces: "`FK-003`'s three options exhaust the design space" and "a record's content can be chosen without deciding what 'full' means."
  Was: decisions.md §`FK-003` → `Presented` (three options; "full" undefined; the only viewer option is the machine record); iterations/009-interview.md §Topic 1 and §The one open question.
If-wrong: The artifact is a documented output surface — `README.md:146-162` presents its contents as the interface a human reads after a session, `T005` rewrites that section, and the Approved Override Register records the replacement contract. So the answer fixes a user-read contract: choosing a self-contained log when change-focus was wanted multiplies the bytes written per session by roughly 2.3× on the user's own dump (≈13 MB vs ≈5.8 MB over 137 turns) — the exact quantity the user's stated SSD rationale in `FK-002` targets; choosing change-focus when self-containment was wanted makes every turn's full text a replay of the chain; choosing a viewer spends a component this plan has never scoped and forces `G-006`'s "without additional tooling" clause to be restated, and it must be authorized and evidenced before it can be built. The record's content is cheap to change in code before release and a contract change after.
Presented:
  Final question: "What should a session's collapsed dump artifact hold, and how should a human read it?"
  Preamble: "Three things are fixed in every option below and are not part of the choice. (1) The artifact is one append-only file per session, written once per bulk flush — your `FK-002` choice; what changes here is what it holds and how you read it, not the write pattern. (2) 'Full' is now defined: the artifact keeps turn 0's complete normalized JSON request body — the same thing `turn_000.json` holds today, and what `VP-T002-turn0-available` already requires — and, because the per-turn changes are computed on the human-readable rendering, it also keeps that rendering of turn 0 as the baseline the first change is taken against. No option loses either one. (3) The numbers below are measured on the real dump you pointed at: 137 turns, 93,886 bytes of raw body for turn 0, and 5,785,285 bytes of change text."
  Options:
    - "Change-focused log, plain text tools" — The artifact holds one summary line per turn, turn 0's baseline, and for every later turn the change against the previous turn. You read it with `grep` and `less` on the one file, exactly the habit the README documents today. Smallest of the four: about 5.8 MB of change text on your real session. You see what changed at each turn directly. The one thing you cannot do is open turn K and read its whole text — you reconstruct it by replaying the chain from turn 0. Nothing new is built.
    - "Self-contained log, plain text tools" — The same one file and the same plain text tools, but every turn's record carries its complete text, so no turn needs a replay and any turn can be opened and read on its own. The change between two turns is no longer shown — you find it by comparing two records. Cost: roughly 2.3× the bytes, about 13 MB instead of 5.8 MB on your real session, which is the quantity your `FK-002` reasoning was about.
    - "Change-focused log + small terminal viewer" — This is the direction you asked about, and it is practical. The artifact holds exactly what the first option holds — the human-readable change-focused log, fewest bytes — and the plan additionally ships a small terminal viewer that lists the session's turns, shows a turn's reconstructed full text, and highlights what changed against the previous turn. It needs no new dependency: the standard library's `curses` module is already available and `difflib` is already imported at `dump.py:5`, so it is a new command in this repository, not a new stack. Cost: it is a component this plan has never scoped, so it must be authorized and evidenced before it can be built; the README documents a tool instead of a format; and the readability requirement as written ("a reader inspecting only the dump directory, with no additional tooling") has to be restated to admit the tool. If the interactive part is not wanted, a one-shot "print turn N" form is a strictly cheaper subset of the same option.
    - "Machine record + viewer" — One JSON object per turn, appended one per line, read with `jq` or with the viewer. The most compact and the most scriptable, and the easiest to aggregate across sessions. Cost: the artifact stops being human-readable, so `cat`, `grep` and `less` no longer work on it and the viewer becomes required rather than additive — this option carries the third option's new-component cost as well as losing what the third option keeps.
  Recommendation: none
  Timestamp: 2026-09-13T21:00:20Z
  Report: iterations/010-decide.md (authored, supersedes `FK-003`); not yet presented — authored this pass for presentation
Resolution:
  User words: —
  Source timestamp: —
  Normative choice: —
  Factual claims: —
Impact:
  Plan: pending — PLAN-SELECTED writes the answer onto: §Goal Definition Matrix `G-006` (the chosen reading, and for options 3/4 a re-derived "without additional tooling" clause), §Anchor Map (the `collapsed dump artifact` row's record serializer; options 3/4 additionally add a `Mode: NEW` viewer row requiring its own complete `RB-*` row, and `T005` becomes tool documentation), §Task List `T002` (record builder) and `T005` (README), §Validation Point Matrix (`VP-T002-change-readable`, `VP-T002-turn0-available`, and a viewer VP for options 3/4), §Behavior Disposition Matrix (B-001, B-002, B-007), §Approved Override Register (the three `OI:` replacement statements), §Invariant Reconciliation (the three `EI:` dispositions)
  Registry: none — no invariant registry exists in this repository
  Roadmap: pending — if options 3/4 are rejected, the viewer is a roadmap candidate rather than a plan element
Supersedes: FK-003
Superseded-by: FK-005

## US-006

State: recorded
Owner: decide
Reserved at: 2026-09-13T21:44:45.548805+00:00
User words: "Donald Dispatch, is it possible for us to make actual conversation be rebuilt, not full of it, but this turn of actual conversation being rebuilt in the log?"
Source timestamp: 2026-09-13 (planning round 2, user statement relayed verbatim in the orchestrator's `User Context` field; the most recent thing the user has seen is the `FK-003` presentation — `FK-004` was authored but never presented)
Normative choice: The artifact's readable content should be the actual conversation rather than the whole request, and it should not carry the whole request each turn — conditional on its being feasible. It selects none of `FK-004`'s four options.
Factual claims:
  - Claim: The actual conversation can be reconstructed from the collapsed artifact (the premise of "make actual conversation be rebuilt").
    Verification: assumptions: claim-10 HOLDS
    Evidence: assumptions/verdicts/claim-10.md
    Consequence: none
      # decide's reading this pass (iterations/011-decide.md §Answer 4): HOLDS for the artifact's *records* —
      # the conversation is the `[message i / role]` sections of the rendering (`dump.py:45-56`) and is what a
      # change-focused record already carries; DOES-NOT-HOLD for a single merged thread — the artifact's turns
      # are not one linear conversation (42 of 137 transitions have fewer messages than the previous turn).
      # Routed to the assumptions pass.
  - Claim: "not full of it" means the artifact should not store the whole request each turn (the change-focused form, not full bodies).
    Verification: assumptions — absorbed (US-007 / FK-007)
    Evidence: decisions.md#us-007, decisions.md#fk-007
    Consequence: none — absorbed by the settled format set: the change-focused record (kept, and the default) plus the self-contained record, chosen once per process; the contrast the statement drew is realized by the two collapsed values.
      # decide's reading this pass (iterations/011-decide.md §Answer 4): HOLDS as the only reading under which
      # the clause carries information — the statement's contrast is with `FK-003`'s "per-turn full bodies"
      # option, which the user had just seen. Routed to the assumptions pass.
  - Claim: The request's tool definitions and system prompt are constant within an artifact, so a change-focused log's records already carry only conversation changes.
    Verification: assumptions: claim-11 HOLDS
    Evidence: assumptions/verdicts/claim-11.md
    Consequence: none
      # decide's reading this pass (iterations/011-decide.md §Answer 4): HOLDS from source and from the real
      # dump — `session_id` keys on the system prompt (`session.py:93-98`), so `[system]` cannot change inside
      # one artifact, and 0 of the dump's 137 diffs change the `[tools]` or `[system]` section. Routed to the
      # assumptions pass.
Impact:
  Plan: pending — the feasibility answer and the constant-scaffolding fact reach `FK-005`'s preamble and option text (iterations/011-decide.md §Answer 4); no plan element is changed by this record
  Registry: none — no invariant registry exists in this repository
  Roadmap: pending — if the conversation-scoped *self-contained* form is wanted it is a named modification on `FK-005`'s second option, not a separate element
Legacy source: none
Supersedes: —
Superseded-by: —

## FK-005

State: superseded
Owner: decide
Reserved at: 2026-09-13T21:45:08.478638+00:00
Type: preference
Question: "What should a session's collapsed dump artifact hold, and how should a human read it — a change-focused log read with plain text tools, a self-contained log of every turn's full text, a change-focused log plus a small terminal viewer, or a machine record plus a viewer — given that the request's tool definitions and system prompt are constant inside one artifact, so a change-focused record already carries the conversation's changes and not the request's scaffolding?"
Why-yours: "it is irreversible or expensive to reverse (data loss, public API surface, security)"
Trigger:
  Changed: The user replied to the `FK-003` presentation with a fourth statement (`US-006`) rather than a pick: "is it possible for us to make actual conversation be rebuilt, not full of it, but this turn of actual conversation being rebuilt in the log?" It is a feasibility question about the artifact's *readable content* — whether the log can hold the conversation instead of the request — and its answer fixes what each record holds. Two facts verified this pass are in no prior trigger: (a) the request's `[tools]` and `[system]` sections never change inside an artifact — 0 of the real dump's 137 diffs touch either, and `[system]` cannot change by construction because the session is keyed on the system prompt — so a change-focused log's records already carry, in substance, only the conversation; (b) the artifact's turns are not one linear conversation — 42 of 137 transitions have fewer messages than the previous turn.
  Source: buster_ripper/session.py:93-98 (`session_id` = `sha256(system)[:12]` — the system prompt *is* the session key); buster_ripper/dump.py:27-58 (`_render_body` emits only `[tools]`, `[system]` and `[message i / role]` sections — the conversation is the message sections, `:45-56`); dump.py:77,87-95 (the diffs are taken on `_render_body` output); command output over the real dump `/Volumes/data/buster-dumps/c451c754ae4f` — 137 turns; `grep -lE '^[-+].*\[(tools|system)\]' turn_*.diff` → **0 of 137** diffs change either section; `grep -o '[0-9]* msgs' index.txt` → **42 of 137** turns have fewer messages than the previous turn (turn_003.diff adds messages 2–7, turn_004.diff removes 5–7, turn_005.diff re-adds 5–10); `jq -c` over turn_000.json → tools 8,851 bytes, system 13,223 bytes, messages 68,113 bytes, against 88.2 KB of rendered text; `pyproject.toml:5-8` (dependencies `aiohttp` + `click` only); `dump.py:5` (`difflib` already imported).
  Replaces: "the artifact's readable content is only a choice between diffs and full bodies; whether it holds the request or the conversation is not itself a choice, and the request's constant scaffolding is not known to be constant."
  Was: decisions.md §`FK-004` → `Presented` (four options, every one whole-request-rendered-text) and §`FK-004` → `Trigger`; iterations/010-decide.md §Answer 1–3.
If-wrong: The artifact is a documented output surface — `README.md:146-162` presents its contents as the interface a human reads after a session, `T005` rewrites that section, and the Approved Override Register records the replacement contract. So the answer fixes a user-read contract: choosing a self-contained log when change-focus was wanted multiplies the bytes written per session by roughly 2.3× on the user's own dump (≈13 MB vs ≈5.8 MB over 137 turns) — the quantity the user's stated SSD rationale in `FK-002` targets; choosing change-focus when self-containment was wanted makes every turn's full text a replay of the chain; choosing a viewer spends a component this plan has never scoped and forces `G-006`'s "without additional tooling" clause to be restated, and it must be authorized and evidenced before it can be built. The record's content is cheap to change in code before release and a contract change after.
Presented:
  Final question: "What should a session's collapsed dump artifact hold, and how should a human read it?"
  Preamble: "Four things are fixed in every option below and are not part of the choice. (1) The artifact is one append-only file per session, written once per bulk flush — your `FK-002` choice; what changes here is what it holds and how you read it, not the write pattern. (2) 'Full' is now defined: the artifact keeps turn 0's complete normalized JSON request body — the same thing `turn_000.json` holds today, and what `VP-T002-turn0-available` already requires — and, because the per-turn changes are computed on the human-readable rendering, it also keeps that rendering of turn 0 as the baseline the first change is taken against. No option loses either one. (3) The numbers below are measured on the real dump you pointed at: 137 turns, 93,886 bytes of raw body for turn 0, 5,785,285 bytes of change text, and a rendered turn 0 of 88.2 KB of which 8.8 KB is the tool definitions and 13.2 KB the system prompt. (4) Your last question — whether the actual conversation can be rebuilt in the log — has an answer that changes what the options mean, so it is stated here rather than deferred: the conversation *is* the `[message i / role]` sections of the rendering, and it is what the artifact already carries. The session is keyed on the system prompt, so the `[system]` section cannot change inside one artifact; and across all 137 changes in your real dump not one touches the tool definitions or the system prompt. A change-focused record is therefore already a conversation record — the request's constant scaffolding appears in it only if it changes, which is itself a cache buster worth seeing. Scoping a record to the conversation *alone* (lifting the constant tool definitions and system prompt out of it) changes almost nothing in the change-focused options and saves about a quarter of the bytes in the self-contained one; if you want that, it is a modification you can name on either option rather than a fifth choice."
  Options:
    - "Change-focused log, plain text tools" — The artifact holds one summary line per turn, turn 0's baseline, and for every later turn the change against the previous turn. You read it with `grep` and `less` on the one file, exactly the habit the README documents today. Smallest of the four: about 5.8 MB of change text on your real session. You see what changed at each turn directly. This is also, in substance, the conversation log your question asks for: the system section cannot change inside an artifact and the tool definitions did not change once in your 137 turns, so what these records carry is the conversation's changes, and the request's constant scaffolding shows up only if it changes. The one thing you cannot do is open turn K and read its whole text — you reconstruct it by replaying the chain from turn 0. Nothing new is built.
    - "Self-contained log, plain text tools" — The same one file and the same plain text tools, but every turn's record carries its complete text, so no turn needs a replay and any turn can be opened and read on its own. The change between two turns is no longer shown — you find it by comparing two records. Because that complete text is the whole rendering, each record repeats the tool definitions and the system prompt — about 22 KB of a 88.2 KB turn-0 rendering — even though neither changes inside an artifact; scoping the records to the conversation alone is a modification you can name on this option. Cost: roughly 2.3× the bytes, about 13 MB instead of 5.8 MB on your real session, which is the quantity your `FK-002` reasoning was about.
    - "Change-focused log + small terminal viewer" — This is the direction you asked about earlier, and it is practical. The artifact holds exactly what the first option holds — the human-readable change-focused log, fewest bytes, already the conversation's changes — and the plan additionally ships a small terminal viewer that lists the session's turns, shows a turn's reconstructed full text, and highlights what changed against the previous turn. It needs no new dependency: the standard library's `curses` module is already available and `difflib` is already imported at `dump.py:5`, so it is a new command in this repository, not a new stack. Cost: it is a component this plan has never scoped, so it must be authorized and evidenced before it can be built; the README documents a tool instead of a format; and the readability requirement as written ("a reader inspecting only the dump directory, with no additional tooling") has to be restated to admit the tool. If the interactive part is not wanted, a one-shot "print turn N" form is a strictly cheaper subset of the same option.
    - "Machine record + viewer" — One JSON object per turn, appended one per line, read with `jq` or with the viewer. The most compact and the most scriptable, and the easiest to aggregate across sessions. Cost: the artifact stops being human-readable, so `cat`, `grep` and `less` no longer work on it and the viewer becomes required rather than additive — this option carries the third option's new-component cost as well as losing what the third option keeps, and it holds the conversation only as a field you must extract before you can read it.
  Recommendation: none
  Timestamp: 2026-09-13T21:46:34Z
  Report: iterations/011-decide.md (authored, supersedes `FK-004`); not yet presented — authored this pass for presentation
Resolution:
  User words: —
  Source timestamp: —
  Normative choice: —
  Factual claims: —
Impact:
  Plan: pending — PLAN-SELECTED writes the answer onto: §Goal Definition Matrix `G-006` (the chosen reading, and for options 3/4 a re-derived "without additional tooling" clause), §Anchor Map (the `collapsed dump artifact` row's record serializer; options 3/4 additionally add a `Mode: NEW` viewer row requiring its own complete `RB-*` row, and `T005` becomes tool documentation), §Task List `T002` (record builder) and `T005` (README), §Validation Point Matrix (`VP-T002-change-readable`, `VP-T002-turn0-available`, and a viewer VP for options 3/4), §Behavior Disposition Matrix (B-001, B-002, B-007), §Approved Override Register (the three `OI:` replacement statements), §Invariant Reconciliation (the three `EI:` dispositions)
  Registry: none — no invariant registry exists in this repository
  Roadmap: pending — if options 3/4 are rejected, the viewer is a roadmap candidate rather than a plan element; if the conversation-scoped self-contained modification is wanted it is an option-2 clause, not a new element
Supersedes: FK-004
Superseded-by: FK-006

### Supersession note (not a factual claim)

**Superseded by `FK-006` in iteration 012.** `US-007` ("can we implment additional formats and
keep teh current?" / "I'm for switchable option") answered the direction this set was built on
rather than picking one of its four options, making the encoding a configuration choice and
retaining the current one — so the mutual exclusivity its question assumed no longer holds. Its
`Presented` text above is preserved exactly as written; it was never shown, so nothing the user
saw changes. The three options that survive are carried into `FK-006` with their costs; the
machine record survives there as part of a larger set, and its "most compact" wording is
corrected there (measured: 1.047× the plain-text change log, not smaller).

### Supersession note (not a factual claim)

`FK-004` was authored in iteration 010 but **never presented** to the user — its own
`Presented.Report` records "not yet presented". The user's fourth statement (`US-006`)
answers the direction the four options were built on rather than picking one, so the
presented set is repaired: the question gains the constant-scaffolding clause, the preamble
answers the conversation question, and each option states what it holds of the conversation
versus the request. `FK-004`'s `Presented` text is preserved above exactly as it was
written; nothing the user saw is changed, because nothing was shown. The four-option
structure, the `Recommendation: none`, and every option's identity are carried unchanged —
no option was added, removed, or re-ranked.

## US-007

State: recorded
Owner: decide
Reserved at: 2026-09-13T23:33:16.470289+00:00
User words: "can we implment additional formats and keep teh current?" — and, asked which reading was meant, "I'm for switchable option". (Both quoted exactly as received, including the original spelling.)
Source timestamp: 2026-09-13 (planning round 3; the question was relayed in the orchestrator's user-provided-context field and the clarification was given in response to the orchestrator asking which reading was meant. The user has seen the `FK-003` presentation; `FK-004` and `FK-005` were authored but never presented, so neither is a question this statement answers.)
Normative choice: The dump artifact's **record format is a configuration choice**, not a single fixed encoding, and the **current change-focused format is retained** rather than replaced. The formats are alternatives selected by configuration — one encoding per artifact — not several encodings written side by side. This is the part carried in the user's own words ("additional formats", "keep teh current", "switchable option").
  # The orchestrator's relay added: "The user's intent is that the artifact's record format be
  # selectable (one artifact per session, one write per flush, format chosen by configuration),
  # with the current change-focused format remaining available — not several formats written
  # simultaneously." That is orchestrator-worded gloss (context priority 5), not user words. Its
  # two substantive additions are treated as follows: "one artifact per session, one write per
  # flush" is not new authority — it is `FK-002`, already settled and independently recorded; the
  # "not several formats written simultaneously" clause is the only part that is neither user
  # words nor prior settled state, and it is treated as a **normative constraint on the design**
  # only because it is the reading under which "switchable option" carries information (a switch
  # selects one; writing all encodings at once is not a switch). If the user meant simultaneous
  # output, that is a different and much larger change and it would be corrected at presentation.
Factual claims:
  - Claim: The artifact's record encoding can be made a configuration choice without altering the settled one-artifact-per-session, one-write-per-flush contract (`FK-002`).
    Verification: plan (VP-T002-write-count-reduced + VP-T002-collapsed-artifact)
    Evidence: plan-summary.md §Task List T002 and §Anchor Map dump format selector, landed by decide 015 ([015](iterations/015-decide.md))
    Consequence: none — realized by T002's format branch: one collapsed artifact per `(sid, format)`, appended once per non-empty flush; the write count is format-independent (`PI:T002/dump_format/format_does_not_change_write_count`), arbitrated at `MS-002`.
      # decide's reading this pass (iterations/012-decide.md §Answer 5): HOLDS — the encoding is a
      # property of the record builder, and the flush already writes "the pending records" as one
      # operation regardless of what a record contains (`dump.py:72-101` is the whole record path).
      # Routed to the assumptions pass.
  - Claim: Additional record encodings can be implemented with the installed dependency set (`aiohttp`, `click`) and no new package.
    Verification: assumptions: claim-1, claim-2 HOLDS
    Evidence: assumptions/verdicts/claim-1.md, assumptions/verdicts/claim-2.md
    Consequence: none — every candidate record is `json` + `difflib` (already imported, `dump.py:4-5`) over the two-dependency runtime (`aiohttp` + `click`, `pyproject.toml:5-8`); no dependency is added.
      # decide's reading this pass (iterations/012-decide.md §Answer 5): HOLDS — every candidate
      # encoding is `json` + `difflib`, both already imported at `dump.py:5-7`; `pyproject.toml:5-8`
      # lists no serialization dependency to add. Routed to the assumptions pass.
  - Claim: Selecting a record encoding multiplies the bytes written per session by that encoding's factor — the self-contained encoding by about 2.3×, the machine record by about 1.047× — measured on the real dump.
    Verification: assumptions — carried provenance, not re-measured
    Evidence: iterations/012-decide.md §Answer 5 (the machine-record figure measured; the self-contained figure is `FK-005`'s recorded estimate)
    Consequence: none — measured provenance on the real dump, which is unmounted (`/Volumes/data`); M4 is explicitly not the proof (`G-008`(b) measures M1/M2), so the plan's arithmetic does not rest on these figures.
      # decide's reading this pass (iterations/012-decide.md §Answer 5): HOLDS for the two figures it
      # names — the machine record is measured (6,061,579 bytes of JSON-escaped change records against
      # 5,785,285 bytes of raw diff over 137 files, ratio 1.047); the self-contained figure is
      # `FK-005`'s recorded estimate (≈13 MB vs ≈5.79 MB, ≈2.3×) and is carried, not re-derived.
      # Routed to the assumptions pass.
Impact:
  Plan: pending — this record is the trigger for `FK-006` (reserved and filled this pass); it reaches plan-summary.md §Options, §Goal Definition Matrix `G-006`, §Open Questions 2b, §Anchor Map (the encoding becomes a configurable seam), §Task List `T002`/`T005`, §Validation Point Matrix, and §Approved Override Register (the `output_layout` replacement statement becomes per-encoding)
  Registry: none — no invariant registry exists in this repository
  Roadmap: pending — encodings not shipped become roadmap candidates rather than plan elements
Legacy source: none
Supersedes: —
Superseded-by: —

## FK-006

State: superseded
Owner: decide
Reserved at: 2026-09-13T23:33:27.773571+00:00
Type: preference
Question: "What should the switchable dump format offer — the current change-focused record and a self-contained full-text record alone, those two plus the small terminal viewer, or those two plus a machine record and the viewer?"
Why-yours: "it is irreversible or expensive to reverse (data loss, public API surface, security)"
Trigger:
  Changed: The user answered the format direction with a third planning-round statement (`US-007`) rather than a pick among `FK-005`'s four options: "can we implment additional formats and keep teh current?", clarified as "I'm for switchable option". This dissolves `FK-005`'s premise — that the four encodings are mutually exclusive and the user chooses one — because the user's answer is that the encoding becomes a **configuration choice** with the current one retained, which is not any of the four positions. Three facts were measured this pass and appear in no prior trigger: (a) a JSON-encoded change record is **1.047×** the plain-text change log, not smaller — 6,061,579 bytes against 5,785,285 over the real 137-turn dump — so `FK-005` option 4's "most compact" is **false**; (b) once the encoding is a per-process configuration choice the artifact **no longer announces its own encoding**, so a reader can no longer tell from `DUMP_DIR/<sid>/` what the artifact holds; (c) the write-count goal is untouched by the encoding — one artifact, one write per flush, `F` writes for N turns either way.
  Source: buster_ripper/dump.py:72-101 (`dump_turn` builds the entire record — summary line, turn-0 JSON body, per-turn `difflib.unified_diff` — as a branch on `state.turn`, so an encoding parameter is a branch on a configured value); buster_ripper/config.py:16 (`DUMP_DIR: Optional[Path] = None` — the mode switch this new switch parallels), `:44` (`KV_POLL_INTERVAL: int = 10` — the module-global shape a format global would follow); buster_ripper/main.py:23-24 (`--dump-dir` is the only dump CLI option today; there is no `--dump-*` format flag, so the flag is new surface); README.md:146-162 (the documented layout and "Reading the index" — the read surface a per-encoding README must cover); pyproject.toml:5-8 (dependencies `aiohttp` + `click` only); command output over the real dump `/Volumes/data/buster-dumps/c451c754ae4f` — 137 `turn_*.diff` files totalling 5,785,285 bytes, `turn_000.json` = 93,886 bytes, `index.txt` = 7,085 bytes, and `jq -Rs .` over every `turn_*.diff`, summed = **6,061,579** bytes (**1.047×** the raw diffs); plan-summary.md §Goal Definition Matrix `G-006` ("a reader inspecting only `DUMP_DIR/<sid>/` … without additional tooling") and §Goals and Constraints (the write-count reduction `1 − F/(2N)` with its 50% floor).
  Replaces: "`FK-005`'s question is the question — one mutually exclusive encoding per artifact, with 'keep the current format' being one of the four answers rather than a fixed constraint on all of them."
  Was: decisions.md §`FK-005` → `Presented` (four mutually exclusive options over one artifact, `Recommendation: none`, never presented) and §`FK-005` → `Trigger`; decisions.md §`US-007`.
If-wrong: The artifact is a documented output surface — `README.md:146-162` presents it as the interface a human reads after a session, `T005` rewrites that section, and the Approved Override Register records the replacement contract — so this answer fixes a read contract. Selecting the self-contained encoding when change-focus was wanted multiplies that session's written bytes by ≈2.3× (≈13 MB vs ≈5.79 MB over the real 137 turns), the quantity the user's `FK-002` rationale targets; shipping a machine record costs the human readability `US-005` asks for, for that encoding; shipping the viewer spends a component this plan has never scoped, adds a `Mode: NEW` anchor needing its own `RB-*` row, and forces `G-006`'s "without additional tooling" clause to be re-derived. Each encoding is cheap to change in code before release and a contract change after — the irreversible part is the **number** of encodings, because every one shipped becomes a documented format a user may come to depend on.
Presented:
  Final question: "What should the switchable dump format offer?"
  Preamble: "Four things are fixed and are not part of the choice. (1) Your answer settles **which encodings exist**, not how one is chosen: the format is a configuration choice made once when the process starts, like `--dump-dir` today, so a session's artifact carries exactly one encoding — not several written side by side. (2) 'Keep the current' is honoured as the **default**: the change-focused record is what you get with no configuration, so today's behaviour and today's reading workflow are unchanged for anyone who does not ask for the other. If you would rather a new encoding be the default, say so — it is a one-line change to this set. (3) None of this changes the write-count goal: one artifact per session, one write per flush, `F` writes for N turns, exactly as before. The byte cost depends on which encoding is *selected*, not on how many exist: measured on the dump you pointed at, the change-focused encoding is about 5.79 MB over 137 turns, the self-contained one about 13 MB (≈2.3×), and the machine record about 6.06 MB (1.047×). (4) The numbers for the change-focused and self-contained encodings are as measured and carried from the previous set; the machine-record figure is new this pass and it corrects an error — the earlier set called that encoding 'the most compact', and it is not: it is 1.047× the plain-text change log, i.e. marginally larger, and its value is that a script reads it without parsing diffs."
  Options:
    - "Two text formats, no viewer" — The format becomes a configuration choice between exactly two encodings, both plain text read with the tools the README already documents. The current change-focused record — a summary line per turn, turn 0's full body, and for every later turn the change against the previous turn — stays the default, so nothing about today's experience changes unless you ask for the other. The second encoding is the self-contained one: every turn's record carries its complete rendered text, so any turn opens and reads on its own with no replay from turn 0, at roughly 2.3× the bytes (about 13 MB instead of about 5.79 MB on your real 137-turn session) whenever it is selected. Both stay human-readable, `cat`/`grep`/`less` keep working on both, and no new component is built — the record builder simply learns the choice. Smallest realization of "additional formats, switchable, keep the current".
    - "Two text formats + small terminal viewer" — The same two encodings and the same default, plus the small terminal viewer you asked about earlier: it lists a session's turns, shows a turn's reconstructed full text, and highlights what changed against the previous turn. It needs no new dependency — the standard library's `curses` is already available and `difflib` is already imported at `dump.py:5` — so it is a new command in this repository, not a new stack. It gives navigation neither encoding gives on its own: with the change-focused encoding you replay the chain, and with the self-contained one you find a turn by scrolling. Cost: it is a component this plan has never scoped, so it must be authorized and evidenced before it can be built; the README then documents a tool as well as the formats; and `G-006`'s "a reader inspecting only the dump directory, without additional tooling" has to be restated to admit it. If the interactive part is not wanted, a one-shot "print turn N" form is a strictly cheaper subset of the same option.
    - "Two text formats + machine record + viewer" — Everything above, plus a third encoding: one JSON object per turn, one per line, for `jq` and for aggregating across sessions. Measured on your real dump, a JSON-encoded change record is **6,061,579 bytes against 5,785,285** for the plain-text change log — **1.047×**, marginally *larger*, not smaller; its value is that a script reads it without parsing diffs, not that it saves space. Cost: for that encoding the artifact stops being human-readable, so `cat`/`grep`/`less` do not work on it and `G-006`'s clause becomes encoding-dependent; and it is a third serializer, a third README section and a third test surface.
  Recommendation: none
  Timestamp: 2026-09-13T23:40:00Z
  Report: iterations/012-decide.md (authored; supersedes `FK-005`); not yet presented — authored this pass for presentation
Resolution:
  User words: —
  Source timestamp: —
  Normative choice: —
  Factual claims: —
Impact:
  Plan: pending — PLAN-SELECTED writes the answer onto: §Options (the encoding dial gains a position), §Goal Definition Matrix `G-006` (the chosen encoding set, and for the viewer options a re-derived "without additional tooling" clause — F-9 generalized by F-11), §Anchor Map (the `collapsed dump artifact` row's serializer becomes encoding-parameterized and the encoding selector becomes a `Mode: NEW` config row; the viewer options additionally add a `Mode: NEW` viewer row requiring its own complete `RB-*` row, and `T005` becomes tool documentation), §Task List `T002` (record builder takes the encoding), `T004` (per-encoding coverage) and `T005` (README documents each shipped encoding), §Validation Point Matrix (`VP-T002-collapsed-artifact` gains the "one encoding, and the reader can tell which" clause; `VP-T002-change-readable`, `VP-T002-turn0-available`, and a viewer VP for the viewer options), §Behavior Disposition Matrix (B-001, B-002, B-007), §Approved Override Register (the `output_layout` replacement statement becomes per-encoding), §Invariant Reconciliation (the three `EI:` dispositions, with `sequential_diff_chain`'s refine qualified per encoding)
  Registry: none — no invariant registry exists in this repository
  Roadmap: pending — any encoding not shipped is a roadmap candidate rather than a plan element; if the viewer is not shipped it is likewise a roadmap candidate
Supersedes: FK-005
Superseded-by: FK-007

### Supersession note (not a factual claim)

**Superseded by `FK-007` in iteration 014** — the same owner, `decide`, on the same cause,
and the repair the routing required. `FK-006` was **presented and answered**; the answer
(`US-008`) selected none of its three options and named no element, so no `Resolution` could
be written from it. `FK-006`'s `Presented` text is preserved above **exactly as written** and
nothing the user saw changes: `FK-007` does not edit this block, it repairs the *set* the next
presentation carries. The repair's diagnosis is that this set **fused two orthogonal axes** —
which encodings exist, and whether a human-facing tool is built — into three packages, so the
user answered the axes separately and the set could not absorb the answer. `FK-007` asks only
the human-facing axis, names the encodings as fixed context, and carries `FK-006`'s
`Recommendation: none`, `Type: preference`, `Why-yours` clause, the machine-record figure
correction (F-10), the 2.3× self-contained cost, and every option's cost clause. What does
**not** survive is the fusion itself and the assumption that the encoding-set membership is what
the user is choosing. One clause of `FK-006`'s preamble is **replaced rather than carried**:
its item (1), "Your answer settles **which encodings exist**, not how one is chosen", is false
for the successor, which asks about the human-facing element and states the encoding set as
fixed. See iteration 014's Dropped-Clause Sweep.

### Supersession note (not a factual claim)

`FK-005` was authored in iteration 011 but **never presented** to the user — its own
`Presented.Report` records "not yet presented — authored this pass for presentation", and no invite
or interview iteration exists between 011 and this pass. The user's third planning-round statement
(`US-007`) does not pick among its four options; it changes the question's shape by making the
encoding a configuration choice and retaining the current one, so the presented set is repaired once
more. `FK-005`'s `Presented` text is preserved above exactly as written; nothing the user saw is
changed, because nothing was shown. Three of the four `FK-005` options survive in substance —
change-focused (the default here), self-contained, and the viewer — and the machine record survives as
part of the third option rather than as a standalone one whose premise was that a viewer must be built
because the artifact is unreadable. `FK-005`'s question, which forced a single mutually exclusive
choice, does not survive, because `US-007` is the user's answer that there is no such single choice.

### Presentation outcome (not a factual claim)

`FK-006` was presented and answered in planning round 3. The user's response, verbatim: "Keep the
existing one as one option and add something more human-visible or human-friendly." It selects **none**
of the three presented options, so the FK's `Resolution` cannot be written from it and `State` stays
`open`. The statement is recorded losslessly as [`US-008`](decisions.md#us-008) (owner: interview). It
creates a materially new question — which element is the "more human-visible or human-friendly" one, and
whether the presented set already contains it — so per the Fork rules the FK returns to its owner
(`decide`) for durable repair: decide reserves and fills a successor FK carrying the repaired option
set, and marks this one `superseded`. The `Presented` block above is unaltered; nothing the user saw
changes.

## US-008

State: recorded
Owner: interview
Reserved at: 2026-09-14T13:14:33.597958+00:00
User words: "Keep the existing one as one option and add something more human-visible or human-friendly."
Source timestamp: 2026-09-14 (planning round 3, the user's response to the `FK-006` presentation, relayed verbatim in the dispatch's `User response:` field)
Normative choice: (a) The current change-focused record stays available as **one option** among the selectable dump encodings — kept, not replaced; and (b) the offered set must additionally carry **something more human-visible or human-friendly** than the plain change-focused text log. The statement answers `FK-006`'s question at the level of the question, and selects **none** of its three presented options: it names no encoding, it does not say whether the second plain-text encoding, the machine record, or the small terminal viewer is the "more human-visible or human-friendly" element, and it does not say whether that element is one the presented set already contains.
  # Scope of this record: only the two clauses above are carried as authority. The dispatch relayed
  # no orchestrator gloss this round, so nothing was added or subtracted by a relay.
  # Deliberately NOT carried as authority: whether "as one option" demotes the current encoding from
  # `FK-006`'s fixed default. The user did not say; the ambiguity is recorded, not resolved.
Factual claims:
  - Claim: An element more human-visible or human-friendly than the plain change-focused log can be added to the dump artifact using the installed dependency set, with no new package.
    Verification: assumptions: claim-1, claim-2 HOLDS
    Evidence: assumptions/verdicts/claim-1.md, assumptions/verdicts/claim-2.md
    Consequence: none — the self-contained addition (the `FK-007` pick) is built from the same installed set (`json` + `difflib` imported, `aiohttp` + `click` only); no new package.
      # decide's earlier readings, carried not re-derived: iterations/010-decide.md §Answer 1 (the
      # small terminal viewer is feasible — the standard library's `curses` needs no new dependency
      # and `difflib` is already imported at `dump.py:5`) and iterations/012-decide.md §Answer 5
      # (every candidate encoding is `json` + `difflib`; `pyproject.toml:5-8` adds nothing).
      # Routed to the assumptions pass.
  - Claim: The three options `FK-006` presented already contain a human-facing addition (the small terminal viewer, in options 2 and 3), and contain no encoding beyond the two plain-text ones and the machine record.
    Verification: assumptions — provenance (superseded presentation)
    Evidence: decisions.md#us-008, decisions.md#fk-006
    Consequence: none — the statement's presupposition concerned the superseded `FK-006` presentation; `FK-007` repaired the set and the pick (option 1) is made, so the viewer never ships and the question is closed.
      # This is the statement's presupposition, not an assertion: "add something more human-visible
      # or human-friendly" presupposes the presented set lacks it. Recorded because it is the fact
      # the repair turns on — decisions.md §`FK-006` → `Presented.Options` describes the small
      # terminal viewer in options 2 and 3, so if that viewer is what the user means, the repair is
      # a clarification rather than a new element. Routed to the assumptions pass.
Impact:
  Plan: pending — the statement is not a pick, so no plan element changes on it. Once decide authors the successor FK, its destinations are `FK-006`'s: §Options (the encoding dial), §Goal Definition Matrix `G-006`, §Anchor Map (the encoding selector row, plus a `Mode: NEW` viewer row with its own complete `RB-*` row if a viewer ships), §Task List `T002`/`T005`, §Validation Point Matrix, §Behavior Disposition Matrix (B-001, B-002, B-007), §Approved Override Register (the `output_layout` replacement statement, per encoding), §Invariant Reconciliation (the three `EI:` dispositions)
  Registry: none — no invariant registry exists in this repository
  Roadmap: pending — any encoding or tool not shipped is a roadmap candidate rather than a plan element
Legacy source: none
Supersedes: —
Superseded-by: —

## FK-007

State: resolved
Owner: decide
Reserved at: 2026-09-14T14:07:45.444993+00:00
Type: preference
Question: "What should the offered set add, so that a reader gets something more human-visible or human-friendly than the plain change-focused log?"
Why-yours: "it is irreversible or expensive to reverse (data loss, public API surface, security)"
Trigger:
  Changed: The user answered `FK-006` with a statement (`US-008`) rather than a pick: "Keep the existing one as one option and add something more human-visible or human-friendly." It selects **none** of `FK-006`'s three options and names **no element at all** — not an encoding, not a tool — so `FK-006`'s `Resolution` cannot be written and the question returns to its owner for repair. `FK-006`'s set **fused two orthogonal axes** into three packages ("which encodings exist" and "is a human-facing tool built"); iteration 012 §"Why three options and not four" had already recorded that "the viewer is orthogonal to the encoding set" and then packaged the viewer *inside* options 2 and 3. The user's sentence addresses the two axes **separately** — "keep the existing one as one option" speaks to the encoding axis, "add something more human-visible or human-friendly" speaks to the human-facing axis — so a repaired set must un-fuse them rather than re-present the same packages. Organize 013's Goal Shape Analysis left three readings `OPEN`: the element's identity, whether `US-008`(a) demotes the current encoding from the default, and whether a viewer is interactive or a one-shot form.
  Source: decisions.md §`US-008` → `Normative choice` (the two clauses, verbatim: "Keep the existing one as one option and add something more human-visible or human-friendly.") and §`FK-006` → `Presented.Options` (options 2 and 3 already contain the small terminal viewer, which is why "add" is ambiguous); iterations/013-organize.md §Goal Shape Analysis (the three `OPEN` rows) and §"Structural observation (for decide, not authority)" (the fusion diagnosis); iterations/012-decide.md §"Why three options and not four"; buster_ripper/dump.py:61-109 (`dump_turn` is the whole record path — one function, a `turn == 0` branch and a `turn ≥ 1` branch — so an encoding is a branch on a configured value, not a new mechanism); buster_ripper/dump.py:4 (`import difflib`; the `:5` cited in `FK-005`/`FK-006`/`US-003`/`US-005`/`US-007` and iterations 010–012 is `import json` — the claim those records rest on HOLDS, only the citation was wrong); buster_ripper/dump.py:84, `:98-101` (the per-turn summary-string construction exists **inline**; no symbol owns it); buster_ripper/config.py:16 (`DUMP_DIR: Optional[Path] = None` — the mode switch this encoding switch parallels) and `:44` (`KV_POLL_INTERVAL: int = 10` — the module-global shape); buster_ripper/main.py:23-24 (`--dump-dir` is the only dump CLI option today, so `--dump-format` is new surface); buster_ripper/main.py:62 and buster_ripper/server.py:99 (the only two entry points — a viewer command is a third); iterations/013-organize.md §"Prospective `RB-*` evidence for a viewer row" (name search `tui|viewer|curses|display|reconstruct|pager|navigat` over `*.py`/`*.md`/`*.sh` → **zero** matches; `_PyInit__curses`/`_PyInit__curses_panel` statically linked into the venv interpreter's `libpython3.13.dylib`, so `import curses` needs no new package); pyproject.toml:5-8 (dependencies `aiohttp` + `click` only); README.md:146-162 (the documented layout and "Reading the index" — the read surface `T005` rewrites); `ls /Volumes` → the real dump `/Volumes/data/buster-dumps/c451c754ae4f` is **not mounted** in this context, so every byte figure below is **carried, not re-measured**.
  Replaces: "`FK-006`'s question is the question — three packages, each a shipped *set* of encodings plus the viewer, with the human-facing element never named as the thing being offered."
  Was: decisions.md §`FK-006` → `Presented` (question, preamble, three options fusing the encoding set with the viewer, `Recommendation: none`) and §`FK-006` → `Trigger`; decisions.md §`US-008`; iterations/012-decide.md §Options Analysis.
If-wrong: The offered set is the plan's documented read surface — `README.md:146-162` presents it as the interface a human reads after a session and `T005` rewrites that section — so this answer fixes a read contract. Choosing "the full-text encoding is the addition" when a tool was wanted leaves the reader with the replay-or-scroll problem the instruction names; choosing a tool spends a component this plan has never scoped, adds a `Mode: NEW` anchor needing its own complete `RB-*` row and a third entry point beside the two existing `main`s, and forces `G-006`'s "without additional tooling" clause to be restated; choosing the one-shot form when browsing was wanted under-delivers the "human-visible" half the user named. Each is cheap to change in code before release and a documented-surface change after — the irreversible part is that every element shipped becomes something a user may come to depend on.
Presented:
  Final question: "What should the offered set add, so that a reader gets something more human-visible or human-friendly than the plain change-focused log?"
  Preamble: "Five things are fixed and are not part of the choice. (1) The artifact stays one append-only file per session, written once per non-empty bulk flush — your `FK-002` choice; this question changes what a reader is offered, not the write pattern, and the write-count goal is untouched (one artifact, one write per flush, `F` writes for N turns versus `2N` today). (2) Your answer keeps the change-focused record as one of the selectable encodings, and it also stays the **default**: with no configuration you get exactly today's artifact and today's reading workflow. That is `US-007`'s 'keep teh current', and the previous set stated the same default and invited you to change it, which you did not; if you want a different default, say so and it is a one-line change to this set. (3) The encodings are fixed here: the change-focused record (kept, default) and the self-contained full-text record — the 'additional format' you asked for. One encoding per artifact, chosen once when the process starts, like `--dump-dir` today. The machine record (one JSON object per turn, for `jq` and for aggregating across sessions) is **not** part of this question: it is the only encoding a script reads without parsing diffs, it measures 1.047× the plain-text log rather than smaller, and for it `cat`/`grep`/`less` stop working — name it if you want it and it rides the encoding set. (4) The previous set bundled the encodings and the human-facing element into three packages, and your answer addressed those two things separately, so this set asks only the half you named: which element is the 'more human-visible or human-friendly' one. If none of the three options below is the element you had in mind, name it — this set does not claim to be exhaustive. (5) The byte figures are carried from the previous set, not re-measured (the real 137-turn dump is not mounted in this context): change-focused about 5.79 MB over that session, self-contained about 13 MB (about 2.3×), machine record 6.06 MB (1.047×). With the encoding configurable the directory listing no longer announces which encoding an artifact holds, so the artifact's name will state it."
  Options:
    - "The full-text encoding is the addition — no new tool" — The offered set is two encodings and nothing else: the current change-focused record (kept, and still the default) plus a self-contained record carrying every turn's complete rendered text. Any turn opens and reads on its own, with no replay from turn 0; what changed between two turns you find by comparing them. This is the 'more human-friendly' half of your sentence, and it adds no component beyond the record builder. Cost: about 2.3× the bytes whenever the self-contained encoding is selected — the quantity your earlier SSD reasoning was about — and the change between turns is no longer shown in place.
    - "Add a small terminal viewer" — the two encodings above, plus a command that lists a session's turns, shows a turn's reconstructed full text, and highlights what changed against the previous turn. This is the 'more human-visible' half: it gives navigation neither encoding gives on its own — with the change-focused record you replay the chain, with the self-contained one you scroll to find a turn. It needs no new dependency: the standard library's `curses` is already available in this interpreter and `difflib` is already imported. Cost: it is a component this plan has never scoped, so it needs its own authorization and evidence; it is a third entry point in a repository that already has two `main`s; the README then documents a tool as well as the encodings; and the readability goal's 'without additional tooling' clause has to be restated to admit it. If you do not want the interactive part, the next option is the strictly cheaper subset of this one.
    - "Add a one-shot 'print turn N' command" — the two encodings, plus a non-interactive command that prints one turn's reconstructed text and its change against the previous turn, then exits. The same information on demand, with no interactive terminal state, no key handling, and far less surface to test. Cost: still a new command and a new README section, but smaller than the viewer — and it cannot browse a session.
  Recommendation: none
  Timestamp: 2026-09-14T14:12:00Z
  Report: iterations/014-decide.md (authored; supersedes `FK-006`); iterations/009-interview.md §Round 4 (presented, 2026-09-14T14:28:50Z)
Resolution:
  User words: "1"
  Source timestamp: 2026-09-14 (planning round 4; the user's answer to the `FK-007` presentation, relayed verbatim in the dispatch's `User response:` field. The value selects `Presented.Options` item 1 — the option list is the one this record carries, presented byte-identical and in order)
  Normative choice: The human-facing addition to the offered set is the **full-text encoding itself, and no new tool**. The dump's encodings are exactly two: the change-focused record (**kept, and still the default** — preamble item (2)'s one-clause correction path was not taken) and the **self-contained** record, in which every turn carries its complete rendered text so any turn opens and reads on its own with no replay from turn 0, and the change between two turns becomes a comparison the reader performs. **No terminal viewer and no one-shot `print turn N` command are built.** The **machine record is not added** — preamble item (3) framed it as outside the question with an opt-in "name it if you want it" path and the user did not name it. The ≈2.3× byte cost of the self-contained encoding is accepted **for that encoding when selected**; the default path's written volume is unchanged, so `FK-002`'s write-reduction outcome is untouched.
    # The pick is complete — nothing in the question is left open. Preamble (2) offered a different default and
    # the user did not take it; preamble (3) offered the machine record and the user did not name it; preamble (4)
    # invited naming an element outside the three options and the user named none. No follow-up question is owed,
    # and no sub-question is completed by inference.
    # This is a pick among presented options, not a statement — the value is bare, so no new `US-*` record is
    # created and no successor FK is required (contrast `FK-006`, whose answer selected no option).
  Factual claims: none
    # A bare option pick asserts no falsifiable proposition. The costs the chosen option carries (≈2.3× the bytes
    # for the self-contained encoding; no new package needed for it — `json` and `difflib` are already imported)
    # are **decide's** claims, already routed to the assumptions pass by `US-007` claim 3 and `US-008` claim 1 and
    # restated in `FK-007`'s preamble item (5) as **carried, not re-measured** (`/Volumes/data` unmounted). The
    # user's pick accepts that cost; it neither originates nor corroborates the figure. Same convention as `FK-001`.
Impact:
  Plan: landed (decide 015, PLAN-SELECTED — iterations/015-decide.md) — every destination below carries the pick:
    §Options (the encoding dial gained the **self-contained** position and the **no-tool** position; the human-facing dial
    closed), §Goal Definition Matrix `G-006` (F-11 resolved by the pick: the "without additional tooling" clause **holds as
    written** for the change-focused encoding and becomes a reader-performed comparison for the self-contained one — `Met when`
    written per encoding; **no restatement was needed because no tool ships**), §Anchor Map (the construction boundary as
    settled by decide 014 — `dump entry point` owns the **encoding branch**, `extend`; `collapsed dump artifact` owns the
    **artifact's name** (F-12) and the append target, `NEW`; `dump summary line` stays `extend` and encoding-independent —
    **plus the `Mode: NEW` `dump encoding selector` row** (`config.DUMP_FORMAT`/`DUMP_FORMATS` + `main.py:--dump-format`)
    **with its own complete `RB-004` row**; **no tool row was added**), §Task List `T002` (record builder takes the
    configured encoding; change-focused default; the encoding selector's config surface and CLI flag land in `T002` so the branch
    is configurable in the increment that first reads it), `T004` (per-encoding coverage) and `T005` (README documents **both**
    encodings and their selection; **no tool section**), §Validation Point Matrix (`VP-T002-default-encoding` **live**;
    `VP-T002-change-readable` and `VP-T002-turn0-available` qualified per encoding per F-11; `VP-T002-collapsed-artifact`
    gained F-12's name clause and is scoped to a **single** configured encoding — the artifact's identity is `(sid, encoding)`;
    `VP-T002-single-write-per-flush` **retired** and folded into `VP-T002-write-count-reduced` per F-3;
    `VP-T001-writes-deferred`/`VP-T001-session-isolation` corrected to `Observable after: T002`;
    **`VP-T002-human-facing` NOT created** — it was a candidate only for the tool options), §Behavior Disposition Matrix
    (B-001, B-002, B-007, plus the configured-encoding and encoding-in-the-name rows), §Approved Override Register (the
    `output_layout` replacement statement became per-encoding, with the artifact's name settled as
    `DUMP_DIR/<sid>/<encoding>.log`; `User Approved` remains **pending** — only the user may stamp it), §Invariant
    Reconciliation (the three `EI:` dispositions, with `sequential_diff_chain`'s refine qualified per encoding — F-11 — plus four
    `PI:` entries and the Invariant Conflict Matrix). `RB-003` is unchanged: there is still **no reader in code**, and the read
    path stays the human workflow (`README.md:146-162`). Full destination detail: iterations/014-decide.md §Options Analysis
    and iterations/015-decide.md §Approved Override Register.
  Registry: none — no invariant registry exists in this repository
  Roadmap: pending — `roadmap.md` is seeded at final approval. The two elements this pick did **not** take become
    roadmap entries **above the MVP floor**: (a) the **machine record** (one JSON object per turn, for `jq` and
    cross-session aggregation; measured 1.047× the plain-text log, so a convenience rather than a saving), and (b) a
    **human-facing tool** — the interactive terminal viewer, or its strictly cheaper one-shot `print turn N` subset
    (decide 014's Forward discretion: a **separate console script** in `pyproject.toml:[project.scripts]`, not a third
    module-level `main`). Neither is below the floor: the pick's own element (the self-contained encoding) is what satisfies
    `US-008`'s "more human-visible or human-friendly" instruction. The **conversation-scoped encoding variant** was
    **not** adopted and is **not** a third roadmap entry — the pick attached no modification and the encoding set is closed
    at two (decide 015 §Dropped-Clause Sweep).
Supersedes: FK-006
Superseded-by: —

## US-009

State: recorded
Owner: interview
Reserved at: 2026-09-14T22:23:44.164482+00:00
User words: "use http://hyperipper.local:30001 for testing there is nothing running locally for this to be tested atm."
  # SECOND UTTERANCE (user correction, 2026-10-17) — the user's correction of the endpoint above, quoted
  # verbatim and in order:
  "So the upstream should be 30,000, not 30,001. 30,001 is another Buster Reaper instance. But 30,000 is up right now."
  # Both utterances are preserved. The first is what the user said on 2026-09-14; the second is what the
  # user says now about the same clause and is the current authority on the endpoint. It corrects a recorded
  # fact about the environment, not a preference: `:30001` is not the model server, and the gates have been
  # pointed at a second proxy. It is not a new Fork — the user named both ports and said which is which, so
  # no option set exists to present.
Source timestamp: 2026-09-14 (the user's answer to the QA infrastructure setup topic — Topic 2 of this session's planning round 2 — relayed verbatim as user-provided context in the dispatch; verified absent from every file under the session directory before this record was written); correction 2026-10-17 (relayed verbatim on the registrar dispatch path as the user's correction of this record, naming both ports and which one is the upstream)
Normative choice: The upstream for QA/testing is the **remote endpoint `http://hyperipper.local:30001`** — the QA harness points the proxy's `--upstream` at it. No local upstream stub, and no `http://localhost:30000`. Scope is the **QA/verification path only**: it does not change the shipped default (`--upstream` default `http://localhost:30000`, `main.py:13`; `config.UPSTREAM`, `config.py:13`).
  # Scope of this record: only the endpoint clause above is carried as authority. The dispatch relayed the user's
  # words verbatim with no orchestrator gloss, so nothing was added or subtracted by a relay.
  # Deliberately NOT carried as authority: whether the harness may fall back to a stub when the remote host is
  # unreachable, and whether the remote host serves exactly the vLLM-compatible surface. The user did not say;
  # recorded as a pending factual claim, not resolved by inference.
  # Note for readers: 30001 is also the app's own `--port` default (`main.py:17`). The two are different hosts —
  # the proxy listens locally, the upstream is remote — so there is no port collision. Recorded so the number is
  # not misread as the listen port.
  # CORRECTION 2026-10-17 (user, second utterance) — the endpoint clause above is SUPERSEDED: the QA/testing
  # upstream is the remote endpoint **`http://hyperipper.local:30000`**, the vLLM-compatible model server the
  # proxy expects. `http://hyperipper.local:30001` is **another Buster Ripper instance** — a peer proxy on the same
  # host — not the model server, and it is not the gate upstream. Everything else in the clause above stands
  # unchanged: no local upstream stub, no `http://localhost:30000`, and the scope is the QA/verification path only
  # (the shipped defaults at `main.py:13` / `config.py:13` are untouched). The corrected clause is the operative
  # one; the clause above is retained as the 2026-09-14 record.
  #   # Why the number looked right: `:30001` is also the app's own `--port` default (`buster_ripper/main.py:70`;
  #   `main.py:17` when this record was written). A Buster Ripper instance started with its defaults listens on
  #   `:30001` and forwards to its own `:30000`, so a probe of `hyperipper.local:30001` reaches a proxy that
  #   answers with its own upstream's vLLM-shaped model list. The note above is right that the port is not this
  #   proxy's listen port; it did not ask what listens at that number on the remote host.
Factual claims:
  - Claim: Nothing is running locally to serve as the proxy's test upstream — no listener on the local upstream default `127.0.0.1:30000`, and none on `127.0.0.1:30001`.
    Verification: assumptions — host runtime state, stale by nature
    Evidence: iterations/009-interview.md §US-009 record (the bounded local observation at recording time)
    Consequence: none — the operative decision (the remote QA upstream) is the user's own `US-009` normative choice; the QA gates' remote-upstream dependency is declared in `.agents/skills/qa-project/SKILL.md` (a gate run fails if that host is unreachable).
  - Claim: `http://hyperipper.local:30001` is reachable from the host that runs the QA gates and serves the vLLM-compatible API the proxy expects.
    Verification: yield-phase:MS-001 **HOLDS** (2026-10-06, HEAD `c7420ce`)
    Evidence: `iterations/071-yield-phase.md` §Check 6 ("Upstream reachability, established first-party") and §Routing; raw transcript `evidence/ms001-gate-and-upstream-adjudication.txt` §3
      # Not source-checkable in this repository — an environment fact. MS-001 is the earliest milestone whose gates
      # need the upstream: VP-T001-writes-deferred, VP-T001-session-isolation, VP-T001-dump-failure-isolated, and all
      # VP-T002-*. If it DOES-NOT-HOLD, the consequence is that every MS-001 gate must re-scope to a stub or the
      # milestone cannot be gated at all.
      # ARBITERED HOLDS at MS-001's Yield Phase, first-party rather than inherited: the harness's own fail-closed
      # preflight require_upstream() passed in 0.05 s; GET /v1/models answered 200 with a vLLM-shaped model list
      # (owned_by "vllm"); and a real proxied POST /v1/messages returned the UPSTREAM's own typed 404 NotFoundError
      # ("The model `qa-gate-model` does not exist.") while creating that session's turn_000.json + index.txt — the
      # signature of a completed round trip, distinguishable from the proxy-local 502 "Cannot connect" the same
      # host's absence produced at 035. The DOES-NOT-HOLD branch (re-scope every MS-001 gate to a stub) does not
      # fire, so no re-scope is owed and US-009's normative choice stands unamended.
      # Scope note: this adjudicates the reachability-and-surface fact only. US-009's other claim (host runtime
      # state: "nothing is running locally") remains routed to `assumptions` and is not this milestone's to settle.
      # RE-CHECK 2026-10-08 at HEAD 83de78c (091-yield-phase): the claim DOES NOT HOLD on that snapshot's host
      # state. Port 30001 on 192.168.50.132 refused connections on 3 immediate probes, on all 43 periodic probes
      # the pass, while the same host answered icmp at 0.25 ms and accepted ports 22 and 30000 — the host is up,
      # the listener is not. All 8 endpoint-dependent QAP-MS001 gates errored on require_upstream() with zero
      # assertion failures, so the four MS-001 VP-* handles are recorded `unknown` and the boundary is
      # BLOCKED-UPSTREAM, not approved. The HOLDS cell above is left intact: it is a dated observation of
      # transient runtime state (this row's own annotation says "an environment fact"), not a correction —
      # 087 measured the endpoint live at 8dfa5af on 2026-10-07. Evidence: evidence/ms001-reboundary-091-gate-attempt.txt;
      # disposition: iterations/091-yield-phase.md §Routing + §Deterministic Closure. No stub fallback was taken:
      # this row's own note says fallback is deliberately not carried as authority, and the DOES-NOT-HOLD
      # consequence named below ("the milestone cannot be gated at all") is what fired, via the other branch.
      # CORRECTION 2026-10-17 (user, second utterance) — the claim above was true of `:30001` while the
      # peer proxy was listening (071), but `:30001` is the WRONG SUBJECT for the gate upstream: it is a
      # second Buster Ripper instance (a peer proxy) on 192.168.50.132, not the vLLM-compatible model
      # server. The user's correction, verbatim: "So the upstream should be 30,000, not 30,001. 30,001 is
      # another Buster Reaper instance. But 30,000 is up right now."
      # The earlier observations are re-read under this correction, not deleted:
      #   • 071's HOLDS (2026-10-06) measured the peer proxy: a reachable endpoint relaying a vLLM-shaped model
      #     list and its own upstream's typed 404. The evidence is real but NON-DISCRIMINATING — a peer proxy and
      #     the model server are indistinguishable by response shape, so it never established that `:30001` was the
      #     model server. The HOLDS cell above stands as that dated observation.
      #   • 091's RE-CHECK (2026-10-08) measured the same peer proxy while it was stopped. The "transient host
      #     state / the listener is not [up]" reading in that annotation is SUPERSEDED as a causal account: the host
      #     was up and the peer proxy was down — a peer lifecycle fact, not vLLM host flakiness. The
      #     BLOCKED-UPSTREAM verdict that annotation produced stands as that pass's verdict.
      # Corrected claim (operative): `http://hyperipper.local:30000` is reachable from the host that runs the QA
      #   gates and serves the vLLM-compatible API the proxy expects.
      #   Verification: yield-phase:MS-001 **HOLDS** (2026-10-17, HEAD `62f8d7c`) — arbitrated by
      #     [093-yield-phase](../iterations/093-yield-phase.md) §Check 6 + §Routing on the corrected endpoint.
      #     (Was `RE-RUN OWED` until this pass; the re-run it owed was executed and passed.)
      #   Evidence: [093](../iterations/093-yield-phase.md) §Check 6 + §Routing; raw transcript
      #     `evidence/ms001-reboundary-093-gates-30000.txt` — preflight `GET /v1/models` → `200` in 0.19 s
      #     with a vLLM-shaped list (`owned_by: "vllm"`, `root: "/model"`, `max_model_len`), then 26 gates
      #     completing proxied round trips through it (11 `QAP-MS001` + 15 `QAP-MS002`, each asserting the peer
      #     answered `< 500`, 32/32 with `QAP-MS004`'s own loopback peer); `:30001` still refuses (`curl`
      #     exit 7 — the peer proxy is stopped, which is consistent with, not contrary to, the correction).
      #     The four MS-001 VP-* handles are `pass` at that report's §Handle Ledger.
      #   Consequence: none — the owed act ("re-run the 32 process-level gates against `:30000`") ran and passed;
      #     no stub fallback was licensed and none was taken.
    Consequence: none
      # CORRECTION 2026-10-17 — `none` stands for the historical claim about `:30001`, whose DOES-NOT-HOLD
      # branch did not fire at 071. For the corrected claim about `:30000` the consequence is now `none` as well:
      # the 091 boundary's cause was a peer proxy, and the re-run the correction owed ran and passed at
      # [093](../iterations/093-yield-phase.md) (HEAD `62f8d7c`).
Impact:
  Plan: landed — `plan-summary.md` §Open Questions item 8 (the QA infrastructure topic): the upstream item moves from an open confirmation carrying decide's code-natural reading (**local stub**) to a **settled live remote upstream**, and the item now lists what remains open. Also landed: `iterations/009-interview.md` §Topic Queue §Topic 2 and §Round 7 (the re-derived presentation, with the stub-vs-live item closed). **Landed (round 18)**: `.agents/skills/qa-project/SKILL.md` §Harness Declarations row "Upstream" records this endpoint. No other plan element changes — no task, VP, milestone, or invariant names the upstream.
    # CORRECTION 2026-10-17 (registrar batch): `plan-summary.md` §Open Questions item 8 now carries a dated
    # correction naming `http://hyperipper.local:30000`; `.agents/skills/qa-project/SKILL.md` §Harness Declarations
    # rows "Start command" and "Upstream" are re-pointed with a dated note in the same batch. Still owed by their
    # lanes: `tests/qa/harness.py:5` (the module docstring) and `:39` (`UPSTREAM = "http://hyperipper.local:30001"`)
    # — `forward-qa`'s lane (QA author pass), test bytes; `qa-plan.md` §Overview and §Scenario Contracts (the
    # trigger commands and the MS-004 reachability note) — the same `forward-qa` lane; and `roadmap.md` §RM-008's
    # endpoint parenthetical — the roadmap owner's re-read (its observation was made through the peer proxy, which relays
    # response headers, so the ruling's substance is likely unaffected while its factual basis still names the wrong
    # endpoint). No task, VP, milestone or invariant names the upstream; that stands.
  Registry: none — this record states no invariant; the registry created in round 18 holds only the dump-format and upstream-TLS properties. The 2026-10-17 correction creates, displaces and weakens no invariant, so `invariants/` carries zero bytes.
  Roadmap: none — the correction names no roadmap impact. `roadmap.md` §RM-008 cites the endpoint as its factual basis; the re-read is the roadmap owner's act and is recorded as owed in §Plan above, not decided here.
Legacy source: none
Supersedes: —
Superseded-by: —

## US-010

State: recorded
Owner: interview
Reserved at: 2026-09-14T23:17:24.541303+00:00
User words: "This looks good to me."
  # The user's answer to Topic 2 (QA infrastructure setup), presented in round 7 (`iterations/009-interview.md`
  # §Round 7 §Re-derived Topic 2). The same message continued with a separate request about upstream TLS
  # verification; that clause is a different normative choice and is recorded separately as `US-011`. The full
  # message was: "This looks good to me. Only thing if it's not a big deal, if upstream will be with HTTPS, please
  # do verification of certificate on it. And have explicit skip verify, like usual, skip TLS verify flag to go
  # unverified on, let's say, self-signed certificates. We should keep security strong."
Source timestamp: 2026-09-14 (relayed verbatim as the user's answer to the QA infrastructure topic — Topic 2 of this
  session's planning round 2; delivered on the interview dispatch path, not as a `User response:` field)
Normative choice: The QA harness contract for this project is **confirmed as presented** — all six items of the
  round-7 Topic 2 presentation are approved, and `.agents/skills/qa-project/SKILL.md` records them:
  1. **Start command** — `.venv/bin/buster-ripper --host 127.0.0.1 --port <port> --upstream http://hyperipper.local:30001
     --dump-dir <tmpdir>`, with `--dump-format <encoding>` varied for the per-encoding gates.
  2. **Base URL / invocation** — `POST http://127.0.0.1:<port>/v1/messages`.
  3. **Test framework** — pytest + pytest-asyncio + pytest-aiohttp + pytest-mock (the `[dependency-groups] dev` set,
     `pyproject.toml`).
  4. **Test file location** — a new `tests/qa/` tree, separate from the mocked `tests/` suite.
  5. **Teardown** — send SIGTERM/SIGINT to the subprocess and wait for exit (the signal is the feature under test).
  6. **Deterministic flush triggers** — the harness fixes **both** trigger parameters at process start: the flush
     interval (`config.DUMP_FLUSH_INTERVAL`, via a CLI flag) and the volume threshold (`config.DUMP_FLUSH_BYTES`,
     via a CLI flag). Without them the timer gate and the volume gate are sleep-and-hope races.
  The upstream for the harness is the remote endpoint settled by `US-009` (`http://hyperipper.local:30001`).
  # Scope of this record: the harness contract above only. It does not settle the TLS-verification request, which
  # is recorded separately as US-011, and it does not extend the QA path's authority over the app's shipped
  # defaults (`--upstream` default, `config.UPSTREAM`).
  # The approval is of the presentation's content, not of a document: nothing in the six items changes when the
  # skill file is written. No item was qualified, narrowed, or amended by the user at approval time (amended
  # 2026-10-17: item 1's endpoint literal only — see the correction below).
  # CORRECTION 2026-10-17 (user, `US-009` second utterance) — item 1's literal `--upstream` value and the
  # sentence above it are re-pointed: the harness upstream is `http://hyperipper.local:30000`, the
  # vLLM-compatible model server. `:30001` is another Buster Ripper instance, not the upstream. The six
  # approved items otherwise stand exactly as approved, including item 1's structure (the start command, the
  # `--dump-format` variation, the flush flags), items 2–6, and the scope guard. This is the first amendment
  # to an item and it changes the endpoint literal only. Detail: `decisions.md` §US-009's 2026-10-17 correction.
Factual claims: none
Impact:
  Plan: landed — `plan-summary.md` §Open Questions item 8 records the harness contract as settled and
    `.agents/skills/qa-project/SKILL.md` is written from it (the `qa-project` capability gap that blocked final
    approval is closed). Also landed: `iterations/009-interview.md` §Round 8 (the answer recorded).
    # CORRECTION 2026-10-17 (registrar batch): the endpoint literal in both destinations is re-pointed to
    # `http://hyperipper.local:30000` (`plan-summary.md` item 8 carries a dated correction;
    # `.agents/skills/qa-project/SKILL.md` rows "Start command" and "Upstream" carry a dated note). No other
    # clause of the contract moves.
  Registry: none — no invariant registry exists in this repository
  Roadmap: none
Legacy source: none
Supersedes: —
Superseded-by: —

## US-011

State: recorded
Owner: interview
Reserved at: 2026-09-14T23:18:29.248845+00:00
User words: "Only thing if it's not a big deal, if upstream will be with HTTPS, please do verification of certificate
  on it. And have explicit skip verify, like usual, skip TLS verify flag to go unverified on, let's say, self-signed
  certificates. We should keep security strong."
  # Verbatim continuation of the same user message whose approving clause is recorded as `US-010`. The opening
  # clause ("This looks good to me") is the QA-harness approval and belongs to `US-010`; the request above is a
  # separate normative choice and is recorded here.
  # SECOND UTTERANCE (placement answer, interview round 9) — the answer to the round-8 placement question, recorded
  # on this same record rather than as a new `US-*`: the question was a scope/placement question on this record, not
  # an FK, and a second record for the same cause would split one causal record in two. Both utterances are
  # preserved verbatim.
  "So if it's already implemented on the positive path, there will be cases where we need to pass the skipTLS
  verify flag. We need to add it. It would be very relevant for local deployments."
Source timestamp: 2026-09-14 (relayed verbatim on the interview dispatch path as the user's answer to the QA
  infrastructure topic — Topic 2 of this session's planning round 2); placement answer 2026-09-16 (relayed verbatim
  on the interview dispatch path, round 9, as the answer to the `US-011` placement question presented in round 8)
Normative choice: When the proxy's upstream is an HTTPS endpoint, the proxy **verifies the upstream's TLS
  certificate**; and there is an **explicit opt-out flag** that turns verification off for self-signed
  certificates. Security posture stays strong by default — verification on unless explicitly skipped. The user
  named the shape ("skip TLS verify flag") but not its spelling; the flag's exact name, and whether the default is
  expressed as a config global or only as a CLI option, are not settled by these words.
  # PLACEMENT (round 9) — SETTLED by the second utterance: "We need to add it." The work is placed **in this
  # session**; the follow-up placement is not taken and no roadmap entry is owed for it. The plan therefore gains
  # the TLS cluster and a scoped `organize` → `decide` replan is owed before final approval.
  # Rationale, recorded as the user's stated priority (normative context, not a proposition about the code):
  # "It would be very relevant for local deployments." Carried as rationale; NOT promoted into a PENDING factual
  # claim, because nothing in this session can observe the deployments it refers to and an ownerless claim would
  # block the approval gate rather than inform it.
  # Scope of this record: the app's outbound TLS posture toward its upstream. The user's conditional ("if upstream
  # will be with HTTPS") is carried as stated: the requirement is about the HTTPS case. The settled QA upstream is
  # plain HTTP (`US-009`), so nothing in the QA path exercises it today — which is the replan's harness question
  # (`iterations/009-interview.md` §Round 9 §Replan Inputs item 4), not a gap in this record.
  # STILL NOT decided by this record, and deliberately not inferred: the flag's exact spelling; whether the default
  # is expressed as a config global, a CLI-only default, or both; whether skipping verification logs a warning;
  # whether the app's `--upstream` default changes (no change is owed — a scheme change is a value change,
  # `main.py:13`). Each is named as open in `iterations/009-interview.md` §Round 9 §Replan Inputs items 9–11.
Factual claims: none
  # The user asserted no proposition about the code. The code facts this request depends on are the interviewer's
  # own reads, recorded in `iterations/009-interview.md` §Round 8 §Source verification and re-verified unchanged in
  # §Round 9 §Source Re-verification (the shared `aiohttp.ClientSession` is created with aiohttp defaults and no TLS
  # option; no `ssl`/`tls`/`verify`/`cert` token exists anywhere in `buster_ripper/`), and they are not promoted to
  # user authority by being written here.
Impact:
  Plan: landed — placement **in this session** (round 9). `plan-summary.md` §Open Questions item 9 records the
    placement and §Current Phase records the owed replan. The TLS cluster gains a task with done-when criteria, bound
    Anchor Map rows (the flag is `Mode: NEW` and owes a complete `RB-*` row), at least one validation point covering
    both postures, a milestone mapping, and a goal row; those land with the scoped `organize` → `decide` replan
    dispatched by `REPLAN-INCREMENT` (round 9). **No existing task, VP, milestone, or invariant is invalidated** —
    the dump cluster is untouched.
  Registry: **landed (round 18)** — `none` was correct when written; the registry created by the final-approval gate
    carries this request's two properties as `PI:T006/upstream_tls/{verification_on_unless_explicitly_disabled,
    optout_is_process_wide}` and the preserve-provenance block
    `OI:buster_ripper.server/on_startup/shared_client_verifies_by_default` in
    `invariants/buster_ripper/server/invariants.md` ([020](iterations/020-interview.md) §Round 18 §Registry Elevation).
  Roadmap: none — the in-session placement is taken; no follow-up entry is owed for this request
Legacy source: none
Supersedes: —
Superseded-by: —

## FK-008

State: resolved
Owner: decide
Reserved at: 2026-09-16T16:01:38.869201+00:00
Type: preference
Question: When the `--skip-tls-verify` opt-out is passed, must the process emit a visible runtime record stating that upstream certificate verification is disabled — or is the flag itself the only signal?
Why-yours: "it is irreversible or expensive to reverse (data loss, public API surface, security)"
  # Binding, stated so the step is inspectable: the named category is `security`, and it is bound to the app's
  # public observable surface — what the process prints when a security control is deliberately turned off.
  # No data settles this: `US-011` names the shape of the flag ("explicit skip verify", "like usual, skip TLS
  # verify flag") and the posture ("We should keep security strong"), and the word "explicit" modifies the flag, not the
  # process output. Both readings are consistent with every word the user wrote.
Trigger:
  Changed: none — preference
  Source: `decisions.md#us-011` Normative choice — "Security posture stays strong by default — verification on unless explicitly skipped. The user named the shape ("skip TLS verify flag") but not its spelling" — and its closing note, "STILL NOT decided by this record ... whether skipping verification logs a warning"; plus `iterations/016-organize.md` §Goal Shape Analysis, third `G-007` row (Authority: `OPEN — user`) and §Deferred Questions item 1 ("the source words settle neither reading"). The repo's adjacent evidence points both ways: `main.py:107-115` already prints one startup summary naming the active modes, while `--verbose` (one of the six `is_flag=True` options) prints nothing about its own activation.
  Replaces: none — no prior belief was recorded; the row has been `OPEN` on user authority since `US-011` was recorded.
  Was: `iterations/016-organize.md` §Goal Shape Analysis, third `G-007` row — alternates "(i) 'explicit' means only *'a flag exists'*; (ii) 'explicit' means the skip is *visible at runtime* (a log line), so an operator cannot silently run insecure. **These classify differently**"
If-wrong: Low code cost either way — a startup log line is cheap to add and cheap to remove. What is not symmetric is what the operator can see. Under reading (i), an operator whose wrapper script carries `--skip-tls-verify` has no signal at all that verification is off, which is the outcome the user's "keep security strong" was guarding. Under reading (ii), an unrequested line appears in the startup output and must be documented in the README.
Presented:
  Final question: When `--skip-tls-verify` is passed, must the process say so at runtime?
  Options:
    1. No runtime record — the flag is the only signal. `VP-T006-optout-is-explicit` is NOT authored, and the startup summary is unchanged.
    2. A visible runtime record — the startup summary (`main.py:107-115`, which already names the active modes) states that upstream certificate verification is disabled. `VP-T006-optout-is-explicit` is authored and gates it.
  Recommendation: none
  Timestamp: 2026-09-16T16:23:07.411685+00:00 — interview round 10, presented; answer not yet received
  Report: `iterations/009-interview.md` §Round 10 (the presentation)
Resolution:
  User words: "I think option A, option 1 is acceptable because what's the point of saying like you're not very fine
    certificate. One way you cannot establish connection, other way it works. There is no confusion."
    # Relayed verbatim on the interview dispatch path as the `User response:` field for `Fork: FK-008` (round 11).
    # The user named the option twice ("option A", "option 1"). The FK's own option list is numbered 1/2, and
    # "option 1" names it exactly; "option A" reads as the same first-position reference, not as a second choice —
    # the two are not in conflict. The stated rationale is consistent with option 1 and inconsistent with option 2:
    # it argues a notice would add nothing, which is a reason not to emit one. Recorded as a mixed answer — the
    # choice is normative, the behaviour it rests on is a factual claim split out below.
  Source timestamp: 2026-09-16T17:34:43Z (interview round 11; the answer to the `FK-008` presentation stamped at
    2026-09-16T16:23:07.411685+00:00, round 10)
  Normative choice: **Option 1 — no runtime record.** Passing `--skip-tls-verify` emits nothing about the posture;
    the flag itself is the only signal. The startup summary (`main.py:107-115`) is unchanged, so the repo's
    "behavioural flags are announced" precedent is not extended to this one. `VP-T006-optout-is-explicit` is **NOT
    authored** — the handle is retired and must not be reused. The Behavior Disposition Matrix row "(new) skipping is
    visible at runtime" closes as **out of scope**. No third `PI:T006` exists. `G-007`'s third half is closed, and
    the goal's `Met when` is unaffected because the clause was never carried there.
    # The rationale is the user's own and is recorded as the reason, not as a proposition: a printed notice about the
    # certificate would add nothing, because the two postures are already distinguishable by outcome. "There is no
    # confusion" is a value judgement — it is carried here as the stated basis for the choice and is not a claim.
  Factual claims:
    - Claim: With no opt-out and an untrusted-certificate HTTPS upstream, a request through the proxy does not
        complete, and the failure is a certificate-verification error.
      Verification: yield-phase:MS-004 HOLDS (2026-10-05, HEAD 25c2b77)
      Evidence: `iterations/056-yield-phase.md` §Check 6 — `VP-T006-verify-on-by-default` / `TC-003` **pass**
        (`tests/qa/test_qap_ms004.py::test_tc003_untrusted_peer_refused_by_default`), with the first-party
        observation in `evidence/ms004-tls-posture-gate.txt` §2 "posture default" block: HTTP 502 carrying
        `SSLCertVerificationError / [SSL: CERTIFICATE_VERIFY_FAILED] … self-signed certificate`, the peer's
        completed-request set empty, 7 aborted handshakes, one connector built `ssl=True`, session carrying no
        `ssl` of its own, rc 0 with no traceback
      Consequence: none
    - Claim: With the opt-out passed against that same peer, the request completes.
      Verification: yield-phase:MS-004 HOLDS (2026-10-05, HEAD 25c2b77)
      Evidence: `iterations/056-yield-phase.md` §Check 6 — `VP-T006-optout-honored` / `TC-113` **pass**
        (`tests/qa/test_qap_ms004.py::test_tc113_same_peer_accepted_with_optout`), with the paired half in
        `evidence/ms004-tls-posture-gate.txt` §2 "posture optout": HTTP 200 carrying the peer's own
        `qa-ms004-peer-answered` marker, the peer recording the `POST /v1/messages`, zero handshake aborts,
        the same single connector now built `ssl=False`; `TC-114`'s paired runs confirm the flag is the only
        variable
      Consequence: none
      # Both claims are the user's own ("One way you cannot establish connection, other way it works"), stated as the
      # basis for the choice. User origin adds no evidentiary weight: they are falsifiable propositions about the
      # shipped behaviour and are routed to the milestone whose validation points already witness exactly this —
      # `VP-T006-verify-on-by-default` and `VP-T006-optout-honored`, both validated at `MS-004`. A supported HOLDS
      # from yield-phase replaces each `Verification` status and `Evidence` with the exact report locator.
Impact:
  Plan: landed — `plan-summary.md` §Goal Definition Matrix (`G-007`, third half: closed as "no runtime record",
    no longer `READY-BLOCKED`), §Validation Point Matrix (`VP-T006-optout-is-explicit` **NOT CREATED**; handle
    retired), §Behavior Disposition Matrix (row "(new) skipping is visible at runtime" → **out of scope**), §Planned
    invariants (the conditional third `PI:T006` is **not authored**), §Milestone Matrix (`MS-004` blocked-on note),
    §Current Phase, §Open Questions item 9.
  Registry: none — no invariant registry exists in this repository
  Roadmap: none
Supersedes: —
Superseded-by: —

## FK-009

State: resolved
Owner: decide
Reserved at: 2026-09-16T16:02:11.363680+00:00
Type: ratification
Question: The upstream-TLS verification point must exercise both postures against a peer whose certificate does not chain to a trusted root. May that peer be started locally by the gate, which amends the QA harness contract the user approved — or must the approved contract stand unamended, leaving the posture proven in the hermetic `tests/` suite with no `tests/qa/` gate?
Why-yours: "the choice would contradict a recorded user decision or a protected invariant"
  # Binding, stated so the step is inspectable: the recorded user decision is `US-010` (`decisions.md#us-010`), the
  # user's approval of the QA harness contract as presented. Its Upstream row reads "There is no local stub: **a gate run
  # fails if that host is unreachable**". A gate-local TLS peer under `tests/qa/` contradicts that sentence as written.
Trigger:
  Changed: the replan's proof-route selection lands on a gate that must negotiate TLS against an untrusted certificate (`iterations/016-organize.md` §Where the Testable Decision Lives, route D), while the approved harness contract declares no local stub exists for gate runs and declares its gate tier to be process-level end-to-end.
  Source: `.agents/skills/qa-project/SKILL.md` §Harness Declarations row "Upstream" — "`http://hyperipper.local:30001` — the live remote vLLM-compatible endpoint. There is no local stub: **a gate run fails if that host is unreachable**."; same file §Conventions "Hermetic default suite" ("The gates under `tests/qa/` need the remote upstream, real subprocesses, and real signals") and §Capability Declarations row "Harness tier" ("process-level end-to-end ... No in-process or mocked tier can prove the shutdown flush, the timer flush, or the volume flush"). Recorded approval: `decisions.md#us-010`.
    # CORRECTION 2026-10-17 — the quoted row's endpoint is corrected by `decisions.md#us-009`'s dated correction
    # to `http://hyperipper.local:30000`; the quoted wording above is preserved as the 2026-09-16 text. This fork's
    # TLS amendment is unaffected: it is scoped to the one gate's loopback peer and names no remote upstream.
  Replaces: "the approved harness contract tolerates a TLS gate's local peer as an addition, so no fork is owed"
  Was: `iterations/016-organize.md` §Goal Shape Analysis row `C-TLS-1` (Authority: `OPEN — user`), reading (i) — "the clause scopes the **dump gates'** upstream, so a TLS gate's local peer is an *addition* to the contract → no fork owed"
If-wrong: The code cost is low in both directions; the cost that is not symmetric is what the user approved. Option 1 edits a document the user approved without their word, and the next reader of `.agents/skills/qa-project/SKILL.md` cannot tell a deliberate stated amendment from an agent's drift. Option 2 leaves the strongest available evidence unbuilt: a gate-local peer is the only way the gate runs the real console script as a local deployment does, which is the case the user named ("very relevant for local deployments").
Presented:
  Final question: May the upstream-TLS gate start its own local self-signed HTTPS peer, amending the approved harness contract?
  Options:
    1. Yes — amend the contract. Its "Upstream" row records the TLS gate's loopback self-signed peer as a stated amendment, and `VP-T006-*` are validated at `MS-004` by a `QAP-MS004` gate under `tests/qa/` at the contract's process-level tier. The gate runs the real console script with `--upstream https://127.0.0.1:<peer-port>`, so the flag-to-posture chain is proven end to end and the gate is authored independently by `forward-qa`.
    2. No — leave the contract absolute. No `tests/qa/` gate is added for TLS and no peer is placed in that tree; the two postures are proven in the hermetic `tests/` suite against a loopback self-signed peer (`MS-004`'s QA author pass becomes `—`, the hermetic suite is the gate), and the flag-to-global link is proven by a CLI-invocation unit test with `web.run_app` patched.
  Recommendation: option 1 — the closure ranks it first. Both options prove the two postures, so correctness is equivalent; option 1 then wins on architectural integrity, because the milestone gate is authored independently by `forward-qa` instead of by the agent that implemented the connector, and it exercises the shipped console script rather than an in-process app. Option 2 is the smaller change, and it is the only one that leaves the approved contract untouched — which is exactly why it cannot be taken without your word.
  Timestamp: 2026-09-16T17:34:43Z — interview round 11, presented
  Report: `iterations/009-interview.md` §Round 11 (the presentation)
Resolution:
  User words: "pisk one I do not care, both ways are mInor"
    # Verbatim, typos included. This is a **delegation**, not an affirmative selection of option 1: the user
    # declined to choose between the two recorded options and authorized this agent to pick. The pick is
    # executed on the only ranked ground on record — the `Presented.Recommendation`, which this agent did not
    # author, did not alter, and added no lean to. Recorded here so no later reader can mistake the pick for
    # an agent answering the fork on the user's own behalf.
    # "both ways are minor" is carried as the user's own materiality judgment (a statement about their
    # priorities), not as a proposition about the two options, so it is not recorded as a factual claim. The
    # recorded `If-wrong` stands unaltered and is not falsified by it: `If-wrong` describes a legibility risk
    # to a future reader of the amended document, which is orthogonal to how much the user cares.
  Source timestamp: 2026-09-16 (relayed verbatim on the interview dispatch path as the user's answer to `FK-009`;
    the fork was presented 2026-09-16T17:34:43Z — round 11 — and re-presented unchanged in round 12)
  Normative choice: The user delegates the selection and states the difference is not material to them; the
    recorded recommendation governs, so **option 1** is taken — the upstream-TLS gate **may start its own
    loopback self-signed HTTPS peer**, and `.agents/skills/qa-project/SKILL.md` §Harness Declarations row
    "Upstream" gains the **stated amendment** scoped to that one gate. The three `VP-T006-*` validation points
    are validated at `MS-004` by the `QAP-MS004` author pass under `tests/qa/` at the contract's process-level
    tier, authored independently by `forward-qa`. Every other gate keeps the remote upstream: the amendment is an
    addition scoped to the TLS gate, not a change to the dump gates' upstream and not a change to the hermetic
    default suite. The user's delegation is the word that makes the amendment legible as deliberate rather than
    as drift, which is the asymmetry `If-wrong` named.
  Factual claims: none
Impact:
  Plan: landed — `plan-summary.md` §Milestone Matrix (`MS-004`'s QA author pass is `QAP-MS004`; the option-2
    fallback string is removed), §Validation Point Matrix (the three `VP-T006-*` rows' QA author pass is
    `QAP-MS004`), §Task List (`T006`'s proof clause now names the `tests/qa/` gate and the committed
    self-signed fixture as that gate's peer material), §Open Questions item 9 (recorded resolved). Also landed:
    `.agents/skills/qa-project/SKILL.md` §Harness Declarations row "Upstream" carries the stated amendment, and
    `iterations/009-interview.md` §Round 13 records the answer.
  Registry: none — no invariant registry exists in this repository
  Roadmap: none
Supersedes: —
Superseded-by: —

## US-012

State: recorded
Owner: interview
Reserved at: 2026-09-16T20:12:17.070561+00:00
User words: Two consecutive messages on the same point, quoted verbatim and in order:
  (1) "Right, but deleting something before we implemented something better is premature."
  (2) "Old behavior should stay unaffected until we prove a new one is better and it's worth switching to."
Source timestamp: 2026-09-16 (interview round 14; the user's answer to Topic 3's presentation in round 13, relayed verbatim in the dispatch's `User response:` field. The dispatch carried no `Fork:` field, so this is a user statement, not an answer to a supplied FK.)
Normative choice: The documented per-turn dump layout (`turn_000.json`, `turn_{N:03d}.diff`, `index.txt`) must not be deleted by the change that introduces the collapsed artifact. **Old behavior stays unaffected until the new behavior has been implemented, demonstrated to be better, and judged worth switching to.** The removal half of the change is therefore premature as planned, and the sequencing is the user's requirement: implement the new behavior, prove it is better, and only then displace the old one. This is the user's own revision of the payload decision they made in round 2 — `FK-002`'s `Resolution.Normative choice` recorded the opposite ordering (the documented layout "is NOT preserved"), and this statement is the current authority on that question. It does not by itself say whether the old layout stays the **default** or merely stays **available**, nor what "prove it is better" is measured by; those are new questions owned by `decide`.
  # Scope of this record: only the normative choice above is carried as authority. The dispatch relayed
  # no orchestrator gloss this round, so nothing was added or subtracted by a relay.
  # Deliberately NOT carried: any reading of "old behavior" beyond the documented dump layout, and any
  # extension of the stated principle to the session's other changes (notably the upstream-TLS work,
  # `T006`, whose observable default posture is unchanged). The user mentioned the dump layout and the
  # `FK-002` decision it rests on; extending the principle to a case they did not mention would be a new
  # decision, and it is not made here.
Factual claims: none — both messages are sequencing and value judgments about what should happen when
  ("is premature", "should stay unaffected", "worth switching to"), not falsifiable propositions about the
  code, the dependency set, or the plan's arithmetic. No claim row is manufactured for them.
Impact:
  Plan: `plan-summary.md` — this statement reopens the payload cluster and requires a scoped replan before
    final approval. Destinations: §Approved Override Register `OI:buster_ripper.dump/dump_turn/output_layout`
    (its `User approved` cell is **refused**, not stamped — recorded round 14); §Approved Override Register
    `OI:.../sequential_diff_chain` and `OI:.../turn_zero_is_full_body` (both describe the collapsed artifact's
    shape, so both become contingent on the replan); §Task List `T002` (its three `REMOVED` clauses — the
    `turn_000.json`, `turn_{N:03d}.diff` and `index.txt` write sites) and `T005` (its `REMOVED` clauses — the
    README layout block); §Goal Definition Matrix `G-001` (the write-reduction goal, its `Met when` and its
    two failure witnesses depend on whether the old write path stays live); §Options (Option 3's payload
    branch); §Anchor Map (`collapsed dump artifact` row's Mode); §Planned invariants
    (`PI:T002/artifact_encoding/*` — four entries); §Invariant Conflict Matrix; §Behavior Disposition Matrix
    (B-001, B-002); §Open Questions item 2b. The successor FK is `decide`'s to reserve and fill; no ID was
    allocated by this pass.
  Registry: **landed (round 18)** — `none` was correct when written (the registry did not exist then), but this
    statement's gating clause **is** an invariant source: the final-approval gate registered
    `PI:T002/dump_format/default_format_is_documented` and `PI:T002/dump_format/documented_format_is_retained`, both
    `Strength: protected`, sourced to this record, in `invariants/buster_ripper/dump/invariants.md`
    ([020](iterations/020-interview.md) §Round 18 §Registry Elevation). The original verdict text follows, unchanged:
    none — no invariant registry exists in this repository (`invariants/` and `.invariants/` both
    absent at the repo root).
  Roadmap: pending — if the documented layout is retained as a selectable option, or its removal is deferred
    to a later session, that deferral is a roadmap entry. `decide` owns the entry.
Legacy source: none
Supersedes: —
Superseded-by: —

## US-013

State: recorded
Owner: interview
Reserved at: 2026-09-16T22:06:30.327213+00:00
User words: "You know I'm getting so infuriated, why the f*cking we spend hours talking about the flag? Are you messing with me? It's not that complicated, make decisions for yourself, it's pretty straight forward."
Source timestamp: 2026-09-16 (interview round 15; the user's answer to Topic 3's remainder — the TLS half of the invariant alignment pass, presented in round 14 and re-presented in round 15. Relayed verbatim in the dispatch's `User response:` field. The dispatch carried no `Fork:` field, so this is a user statement, not an answer to a supplied FK.)
Normative choice: The user **delegates** the decision and directs the agents to decide rather than asking again. Concretely:
  (a) The TLS half of the invariant alignment pass is **ratified by that delegation** — `OI:buster_ripper.server/on_startup/shared_client_verifies_by_default` and the two `PI:T006/upstream_tls/*` entries. The dispositions stand exactly as decide 016/017 authored them; the user declined to review them and said so. The referent of "the flag" is the TLS `--skip-tls-verify` question this session spent rounds 10-14 on.
  (b) The user directs the agents to decide the **remaining agent-decidable items** in this planning pass rather than being asked. The process feedback is part of the choice: the interview loop's per-item questioning is what the user is refusing.
  (c) The directive does **not** repeal the escalation floor: a choice that would contradict a recorded user decision (notably [`US-012`](#us-012)) or a protected invariant is still not the agents' to make, and a delegation does not convert a factual claim into an established one. Nor is it a standing waiver for the whole session — it is scoped to this pass's remaining items.
  # Scope of this record: only the normative choice above is carried as authority. The dispatch relayed no
  # orchestrator gloss, so nothing was added or subtracted by a relay.
  # Deliberately NOT carried: any reading of the user's payload principle (`US-012`) as now extended to the
  # TLS cluster, or of this delegation as a repeal of `US-012`'s non-extension clause. The two statements have
  # different referents (the dump layout vs the TLS flag) and neither is generalized into the other.
Factual claims: none — "it's not that complicated" / "it's pretty straight forward" is the user's materiality
  judgment about the decision (the same shape as `FK-009`'s "both ways are mInor"), not a falsifiable proposition
  about the code, the dependency set, or the plan's arithmetic. Nothing in the session can disprove it, and no claim
  row is manufactured for it. The delegation itself is a normative act, not a proposition.
Impact:
  Plan: `plan-summary.md` — destinations:
    §Approved Override Register `OI:buster_ripper.server/on_startup/shared_client_verifies_by_default` — its
      `User approved` cell is stamped **`yes — by explicit delegation (US-013)`**, with the delegation provenance
      written out so no later reader mistakes it for an affirmative review of the statement (same shape as
      [`FK-009`](#fk-009)'s pick);
    §Planned invariants — the two `PI:T006/upstream_tls/*` entries are **confirmed** (statement and strength
      unchanged; a confirmation is not an edit); the four `PI:T002/artifact_encoding/*` entries remain **contingent**
      on the payload replan and are not confirmed by this delegation;
    §Approved Decisions, §Current Phase, §Session Metadata, §Open Questions item 9 — status updated;
    §Approved Override Register entries 1–3 — **unchanged**: entry 1 stays REFUSED (`US-012`), entries 2–3 stay
      contingent on the payload replan. The delegation does not reach them.
  Registry: **landed (round 18)** — `none` was correct when written: no invariant registry existed at that time
    (`invariants/` and `.invariants/` both absent at the repo root, re-verified in round 15). The final-approval gate in
    round 18 created the registry, and the two confirmed `PI:T006/upstream_tls/*` entries plus
    `OI:buster_ripper.server/on_startup/shared_client_verifies_by_default` are registered in
    `invariants/buster_ripper/server/invariants.md` ([020](iterations/020-interview.md) §Round 18 §Registry Elevation).
  Roadmap: none — this statement approves no deferral and defers no scope. The payload-cluster reroute's roadmap
    entry (if the documented layout is retained or its removal deferred) remains `decide`'s to write; 019 wrote it
    (`roadmap.md` §Switch the default dump format).
Legacy source: none
Supersedes: —
Superseded-by: —

## US-014

State: recorded
Owner: interview
Reserved at: 2026-09-16T23:55:38.049100+00:00
User words: "I approve"
Source timestamp: 2026-09-16 (interview round 17; the user's answer to Topic 1 — the **payload half of the invariant
  alignment pass** — presented by `invite` in round 16 as ONE question with three options, relayed verbatim in the
  dispatch's `User response:` field. The dispatch carried no `Fork:` field, so this is a user statement, not an answer
  to a supplied FK. The selected option is the presented option 1, *"Ratify all three entries and the six planned
  invariants as written (recommended)"* — the user's words are the affirmative, not the label.)
Normative choice: The user **ratifies the payload half of the invariant alignment pass**. Concretely:
  (a) `OI:buster_ripper.dump/dump_turn/output_layout`, `OI:buster_ripper.dump/dump_turn/sequential_diff_chain` and
      `OI:buster_ripper.dump/dump_turn/turn_zero_is_full_body` are stamped **`User approved: yes`**, with the
      dispositions and replacement statements exactly as decide 018/019 authored them — all three **`refine`**, and
      entry 1's statement carrying the **default clause** (`documented`).
  (b) The six `PI:T002/dump_format/*` planned invariants are **confirmed as written**, with the strengths as authored
      (`format_is_per_artifact`, `format_does_not_change_write_count`, `default_format_is_documented`,
      `one_write_path_per_turn`, `documented_format_is_retained` = hard_requirement; `format_set_is_closed` = preferred).
  (c) The **disclosed consequence is accepted**: under S-1 the session's write reduction is **opt-in** — a run that
      configures nothing gets today's files at today's `2N` writes. The user stamped the statement that carries it.
  (d) The round-14 **refusal is discharged** — what is stamped is entry 1's **rewritten** statement (the documented
      layout stays and is the default), not the removal-half statement [`US-012`](#us-012) refused.
  (e) The ratification is **scoped to the payload half**. It does not re-open, extend or restate `US-012`'s
      non-extension clause, and it does not touch the TLS half (stamped in round 15 by delegation, [`US-013`](#us-013)).
  # Scope of this record: only the normative choice above is carried as authority. The dispatch relayed no orchestrator
  # gloss, so nothing was added or subtracted by a relay. The word "approve" is read against the question actually
  # presented (the invariant alignment pass), not generalized into approval of the plan as a whole — that is a separate
  # question, still unasked, and this record is not evidence that it was asked or answered.
Factual claims: none — the ratified statements are plan and invariant artifacts, not new falsifiable propositions
  offered by the user. The factual content they carry is already owned by the plan's own records with their own
  verifiers (`VP-T002-documented-format-preserved`, `VP-T002-default-encoding`, `VP-T002-write-count-reduced`, the
  `EI:`/`PI:` records themselves); ratifying a statement neither creates nor discharges those claims, and no claim row
  is manufactured for the word "approve". The ratification itself is a normative act, not a proposition.
Impact:
  Plan: `plan-summary.md` — destinations:
    §Approved Override Register entries 1–3 — their `User approved` cells stamped **`yes`** with the round-17 provenance,
      and the section's stale refusal/contingency preamble replaced by the stamped result;
    §Planned invariants — the six `PI:T002/dump_format/*` entries **confirmed** (statement and strength unchanged; a
      confirmation is not an edit);
    §Interview Alignment — new block recording the completed pass (date, the ten items, the answer source, the two
      closed gates);
    §Approved Decisions, §Current Phase, §Session Metadata, §Open Questions item 2f — status updated;
    §Goals and Constraints — **unchanged**: the disclosed consequence was already written there and is now ratified
      rather than merely disclosed.
  Registry: **landed (round 18)** — `none` was correct when written: no invariant registry existed at that time
    (`invariants/` and `.invariants/` both absent at the repo root, re-verified in round 17). The elevation step is a
    **final-approval** step, and the final-approval gate that ran in round 18 created the registry: the ten items this
    record ratified are now registered in `invariants/` — `OI:buster_ripper.dump/dump_turn/{output_layout,
    sequential_diff_chain, turn_zero_is_full_body}` (all `Decision: preserve`, `Approved: yes`, sourced to this record)
    and the six `PI:T002/dump_format/*` in `invariants/buster_ripper/dump/invariants.md`. The ratification's own
    destination is therefore **landed**, not skipped ([020](iterations/020-interview.md) §Round 18 §Registry Elevation).
  Roadmap: none — this statement approves no deferral and defers no scope. The format-switch entry already exists in
    `roadmap.md`, sourced to [`US-012`](#us-012).
Legacy source: none
Supersedes: —
Superseded-by: —

## US-015

State: recorded
Owner: interview-migrate
Reserved at: 2026-09-27T21:58:14.586913+00:00
User words: "approve, begin"
Source timestamp: 2026-09-17 (interview round 18; the source records a date, not an exact answer time; the final-approval checkpoint was committed 2026-09-17T19:11:11-04:00, which is a commit time, not the user's answer time)
Question and presentation: The final-approval question was presented in round 17, after the separate payload-invariant ratification recorded as US-014. Exact rendered presentation, preserved in the legacy source:
  > **The plan is finalized and every item you were asked about is now stamped — this is the last question.**
  >
  > Where we are: this session plans the dump bulk-flush change (hold captured messages in memory and write them to disk in one bulk flush instead of one write per turn, flushing on Ctrl-C/SIGTERM), plus an independent upstream-TLS verification opt-out. You have now ratified the last open item: the three dump properties are recorded as *refinements* (nothing is deleted from the code), and the six new format properties are confirmed as written.
  >
  > What a yes does: it marks the plan implementable and runs the remaining planning gate — writing the QA plan (the milestone scenarios that will be executed at each gate), confirming every goal and validation point is covered, and the invariant-registry step.
  >
  > One thing to know before you answer, because it touches your repository: the registry step is the only part that writes outside the session directory. **This repository has no invariant registry today** — neither `invariants/` nor `.invariants/` exists — and the skill that governs this step treats a missing registry as the reason to **create** one, so a plain yes means a new `invariants/` directory appears in the repo holding this session's invariants (the three refined dump properties, the six new format properties, and the two TLS properties). If you would rather that step be recorded as **deferred** and the repo left untouched, that is option 2.
  >
  > What you are approving, in one line each: six tasks — the per-session buffer; the format selector (`documented` default, plus `change-focused` and `self-contained`) with the collapsed artifact and the single-write flush; the three flush triggers (volume threshold, interval timer, shutdown); the test set including the documented-versus-collapsed comparison; the README extension; and the TLS opt-out flag.
  >
  > Two things the plan does **not** deliver, so a yes is not a surprise later: the write reduction is **opt-in** (a run that configures nothing behaves exactly as today — this is the sentence you just stamped), and the physical flash-wear axis is **not measurable in this repository**, so "the new format is better" is proved on write counts and file counts, which is a proxy.
  >
  > The trade-off: nothing is written to the code yet, so approving is cheap to reverse *now* — but after this the plan stops being editable by conversation and changes arrive as remediation against work already done. Declining costs one more round.
  >
  > Options:
  > 1. "Approve — the plan is implementable, start implementation"
  > 2. "Approve with a change — I'll name it"
  > 3. "Not yet — I have a concern"
  >
  > Do you approve this plan as implementable, so implementation can start? (This is the last question; a yes closes planning.)
Presented at: 2026-09-16 (round 17; exact presentation time not recorded in the legacy source)
Answer source: 2026-09-17 round-18 orchestrator prompt, `Fork: final-approval`, `User response: "approve, begin"`; the prompt gloss says option 1 selected and no change or concern named. The quoted user words, not the gloss, are the authority.
Normative choice: The user approved the finalized six-task plan as implementable and authorized implementation to begin, selecting presented option 1. No change or deferral was requested; the disclosed invariant-registry elevation was included. This is distinct from US-014's earlier approval of only the payload invariant alignment. Approval does not choose a later default-format switch; US-012's comparison-and-user-judgment gate remains in force.
Factual claims: none asserted in the user's answer. The presented baseline and predictions are plan assertions, not new user-origin facts. The finalization pass's QA coverage and registry verification are recorded in the legacy round-18 gate, not inferred from the word "approve".
Verification: normative approval is evidenced by the verbatim answer and its preceding exact presentation in `iterations/020-interview.md` §§Round 17–18. Gate execution is evidenced by the same report §§Gate Steps, Registry Elevation, Validation Performed; no independent factual verdict is manufactured by this migration.
Consequence: The planning gate executed in the stated order: provisional intent, QA plan, goal/VP coverage, registry elevation, then implementable mark. The legacy report explicitly notes unresolved PENDING factual claims as owned by the subsequent assumption gate; approval did not resolve those claims.
Impact:
  Plan: landed — `plan-summary.md` records planning closed and the plan implementable; `handoff-context.md` records the final-approval gate and implementation handoff. This migration changes neither plan meaning nor implementation status.
  Registry: landed — round 18 created `invariants/index.md`, `invariants/buster_ripper/dump/invariants.md`, and `invariants/buster_ripper/server/invariants.md` and checked the index. The registry operation is a consequence of this approval, not a new invariant authored by this migration.
  Roadmap: existing default-format switch remains gated by the user's later judgment of comparison evidence; no new deferral or switch was approved here.
Legacy source: `iterations/020-interview.md` §§Round 17, User-Facing Presentation (lines 314–341), Round 18, Orchestrator Prompt / Answer Recorded / Gate Steps / Registry Elevation (lines 389–474). Checkpoint `b9c0bb0` (2026-09-17T19:11:11-04:00) corroborates landing, not the answer time.
Supersedes: —
Superseded-by: —

## US-016

State: recorded
Owner: registrar
Reserved at: 2026-09-28T13:14:43.747707+00:00
User words: "reorder and finish coding first"
Source timestamp: 2026-09-28 (orchestrator conversation; the user's answer to Forward's STOP packet, relayed verbatim in the registrar
  dispatch's Sources field)
Question and presentation: Forward stopped before the MS-002 boundary with the next chain blocked: T004 was gated behind the MS-002
  Yield Phase re-verdict ([035-yield-phase](iterations/035-yield-phase.md): "QAP re-run → MS-002 re-verdict → T004"), and the QAP-MS002
  re-run cannot execute — the required live upstream `http://hyperipper.local:30001` is unreachable (DNS `gaierror [Errno 8]`;
  Checkpoints 98/99, [048-forward-qa](iterations/048-forward-qa.md) § prerequisite failure, "Upstream-down must stop the gate as
  unavailable, not pass it"). The STOP packet offered numbered routes; route 2 was "a recorded user-approved re-order/waiver of the
  T004 gate". The user's answer selects route 2. The exact rendered packet text lives in the 2026-09-28 orchestrator conversation and
  is not restated here; the authority of this record is the verbatim user words plus the dispatch's recorded effect, quoted below
  without extension.
Normative choice: The user approves the re-order (Forward's STOP packet route 2). Effect, exactly as recorded:
  (a) **Re-sequencing — waiver of the T004 gate in execution order:** T004 and T006 are authorized to run BEFORE the MS-002 Yield
      Phase re-verdict. The 035 ordering clause ("QAP re-run → MS-002 re-verdict → T004") is waived for dispatch timing only.
  (b) **The MS-002 obligation is deferred, not discharged:** the QAP-MS002 re-run and the MS-002 re-verdict remain OWED whenever
      `hyperipper.local:30001` is reachable. They are not cancelled, not satisfied by this record, and this record grants no
      milestone acceptance — the 035 rejection stands until that re-verdict runs.
      # CORRECTION 2026-10-17 — the reachability condition is corrected by `decisions.md#us-009`'s dated correction to
      # `hyperipper.local:30000` (`:30001` is a peer Buster Ripper instance, not the model server). The deferral itself
      # stands unchanged: the QAP-MS002 re-run and the MS-002 re-verdict remain OWED whenever `:30000` is reachable, and
      # nothing in this record's authorization moves.
  (c) **No product scope, goal, or milestone criterion changes:** the six-task scope, `G-001`…`G-008`, the Milestone Matrix criteria
      (including MS-002's "Complete when"), and every VP/QAP contract stand unchanged; T004/T006 acceptance still flows through their
      own milestones (MS-003 coverage gate, MS-004 TLS gate).
  (d) **The disclosed rework risk is accepted:** a later MS-002 re-verdict finding may require adjustment of T004's tests, because
      T004 now lands before the re-verdict that was its gate. The user accepted this as route 2's disclosed consequence.
Factual claims: none — the user's words are an instruction; they assert no falsifiable proposition about the repository.
  # The blocking premise (QAP-MS002 blocked because the required live upstream is unreachable) is Forward's recorded evidence, not a
  # user claim: Checkpoints 98/99 and iterations/048-forward-qa.md (DNS `gaierror [Errno 8]`; prerequisite failure before any proxy
  # starts). Its ongoing truth is established by the gate itself when re-run, not by this record. The "remaining coding" reading of
  # "finish coding first" is grounded in plan state (T004 and T006 are the only unstarted tasks), owned by plan-summary.md, not
  # asserted here as a claim row.
Impact:
  Plan: landed (registrar batch 2026-09-28) — `plan-summary.md` §Session Metadata Last Updated, §Progress Overview Forward/Yield row,
    and §Current Phase re-ordered: next **T004 → T006** (both authorized pre-re-verdict), with the owed chain (QAP-MS002 re-run →
    MS-002 re-verdict) explicitly retained as reachability-gated. §Task List clauses, §Milestone Matrix, and §Validation Point Matrix
    are unchanged — this source changes execution order, not task content or criteria. `handoff-context.md` §Session Metadata and
    §Current Status carry the same projection with the forward-readiness statement.
  Registry: none — this statement settles execution ORDER, not behavior. It states no invariant, displaces none, and weakens none;
    `invariants/` is untouched (0 new/updated/superseded/upheld; no index rebuild).
  Roadmap: none — the source names no roadmap impact; the owed re-verdict is a session obligation carried by plan-summary.md and
    handoff-context.md, not a roadmap deferral.
Legacy source: none (the STOP packet's exact rendered text remains in the 2026-09-28 orchestrator conversation; its persisted
  evidence anchors are Checkpoints 98/99 and iterations/048-forward-qa.md)
Supersedes: — (it waives the ordering effect of the 035 routing for T004's dispatch timing; it supersedes no FK or US record, and
  the 035 report itself remains accurate history — its rejection verdict and remediation routing are untouched)
Superseded-by: —

## FK-010

State: resolved
Label: serve-script disable form
Owner: assumption-decide
Reserved at: 2026-09-28T15:46:32.900804+00:00
Type: preference
Question: The serve script's usage header documents `DUMP_DIR="" bash serve_buster_ripper.sh` as "# disable prompt diffing" (serve_buster_ripper.sh:14), but line 28's bash `:-` default replaces the exported empty value with the non-empty default, so line 36's `-n` guard is always true from the environment and argv always carries `--dump-dir ~/mllm/prompt-diffs` — the documented disable form has never disabled anything. Must the serve script honor that documented disable form, or must the documentation stop promising it?
Why-yours: "it changes the product's user-facing capability set either way (a working off-switch on the documented run path versus none)"
  # Binding, stated so the step is inspectable: the two picks add vs. retract a user-reachable behaviour on the
  # serve path. The header comment predates this session — it is repo documentation, not a ratified decision:
  # no US-*/FK-* record or goal mentions a serve-script disable, the form was never functional under the
  # recorded `:-` mechanism (verified: claims 161+162 composed 2026-09-28), and no S6 contract dimension
  # carries it (no README example uses it). No data settles whether the product should have a serve-path
  # disable switch; only the user's product judgment does.
Trigger:
  Changed: verified evidence — the 2026-09-28 S6 recomposition corrected the prior composition's factually
    wrong disable-form trace, which had asserted the empty value collapses through to flag suppression.
  Source: `assumptions/compositions/seam-S6.md` §Counterexample and §Composition ("Correction of the prior
    composition's disable-form trace"); member verdicts `assumptions/verdicts/claim-161.md` (the `:-`-default
    mechanism) and `claim-162.md` (the argv channel), both HOLDS 2026-09-28 on script
    sha256:f4076d1ceae359f201f4e68d62c41ea7af540a0620a6b65943d405351756fa08; serve_buster_ripper.sh lines
    14, 28, 36 (working tree, fingerprint re-verified at routing).
  Replaces: the T005-era composition's implicit belief that the disable form worked — never a user belief
    record, only a composition trace now corrected on the record.
  Was: the prior composition's trace "DUMP_DIR='' collapses through line 26 to empty → line 33 guard false →
    flag suppressed" — factually wrong about bash `:-` semantics.
If-wrong: Both picks are single-file, cheap and reversible (either fires the S6 composition's own
  Invalidates-On — member re-verification plus recomposition — and nothing else). What is not symmetric is
  the user-facing surface: under option 1 the serve path gains a working dump off-switch the header already
  promises, and `DUMP_DIR=""` stops writing dumps to `~/mllm/prompt-diffs`; under option 2 the repo stops
  promising a disable on that path, and disabling requires invoking `buster-ripper` directly without
  `--dump-dir` (the package's `config.DUMP_DIR = None` off-mode stays unreachable through the wrapper).
Presented:
  Wave: FK-010 (single-fork wave, order 1 of 1; dispatched by run-codify 2026-09-28 — "all other FKs
    resolved or superseded; no dependencies"). Dependencies: none (intra-wave dependency impossible —
    one fork). Predecessor/supersession: none — first presentation of this Fork (the routing pass
    2026-09-28 pre-drafted the question/options below; nothing presented before).
  Exact rendered capsule text (verbatim, matches the emitted Sprint Presentation byte-for-byte):
  ```text
  #### FK-010 — The serve script's documented "disable prompt diffing" switch has never worked — honor it or retract it?

  **Where this decision sits.** The repository ships `serve_buster_ripper.sh`, a wrapper that starts the
  proxy with environment-variable conveniences. All six implementation tasks of this session (T001–T006)
  are code-complete and accepted at task level; what remains is milestone QA (the MS-004 TLS gate author
  pass next, plus a re-run of the MS-002 gate that stays blocked until the remote upstream host is
  reachable again). This question is on none of those paths: it was surfaced by post-acceptance
  verification — a cross-check of how the script's environment variables flow into the CLI invocation,
  run when the TLS task (T006) landed and rewired the script's third variable. No task owns it, so it
  comes to you before any code can change.

  **Today's baseline.** The script's own usage header, line 14, documents:

      DUMP_DIR="" bash serve_buster_ripper.sh    # disable prompt diffing

  That form has never disabled anything. Line 28 reads `DUMP_DIR=${DUMP_DIR:-${HOME}/mllm/prompt-diffs}`;
  in bash, `:-` substitutes the default when the variable is empty **or** unset, so an explicitly exported
  empty `DUMP_DIR` is replaced by the non-empty default. Line 36's guard `[[ -n "${DUMP_DIR}" ]]` is
  therefore always true from the environment, and the script always passes
  `--dump-dir ~/mllm/prompt-diffs` to the binary. Anyone who ran the documented disable form still got
  dumps written under `~/mllm/prompt-diffs`. (Verified 2026-09-28 against the script's exact bytes —
  sha256 `f4076d1c…`, re-verified at this presentation. The README never shows this disable form; its
  examples only use non-empty `DUMP_DIR`.)

  **Decision trail.** The header comment predates this session — it is repository documentation, not a
  decision you or this session ever ratified: no goal (G-001–G-008), no recorded user statement, and no
  task (T001–T006) ever mentions a serve-script disable switch, and the form was never functional under
  the recorded mechanism. The verification pass of 2026-09-28 (recomposing the serve-script env-to-argv
  flow after T006 landed — `assumptions/compositions/seam-S6.md`, facts claim-161 and claim-162)
  corrected an earlier analysis that had wrongly believed the disable form worked, and the routing
  adjudication (`CLOSURE: question`) found no authority in the repository that can rank the two fixes
  against each other — which is why this is yours and not the agents'.

  **Why now.** The corrected trace landed as verified evidence: the script's user-documented environment
  surface contradicts its actual behavior. Either the code changes to match its documentation, or the
  documentation changes to match the code — the repo should not ship a documented off-switch that
  silently stays on. Both picks are single-file, cheap and reversible, but they **add vs. retract a
  user-reachable capability** on the serve path, and only your product judgment can choose. This is the
  only open fork; nothing else waits on it.

  **Option 1 — honor the documented disable form (change the script).** Make line 28's default apply only
  when `DUMP_DIR` is unset — `DUMP_DIR=${DUMP_DIR-${HOME}/mllm/prompt-diffs}` (drop the colon). An
  explicitly empty `DUMP_DIR` then stays empty, line 36's guard suppresses `--dump-dir`, and the binary
  runs at its own no-dump default (the package's `config.DUMP_DIR = None` off-mode, which today is
  unreachable through the wrapper). Unset `DUMP_DIR` keeps today's default; every README example is
  unchanged.
  - Observably changes: the serve path gains a working dump off-switch the header already promises;
    `DUMP_DIR="" bash serve_buster_ripper.sh` stops writing dumps to `~/mllm/prompt-diffs`.
  - Preserves: behavior with `DUMP_DIR` unset or non-empty (the default and every README example); the
    header's line-14 text as written.
  - Gain / loss / cost / reversibility: gain — the promised capability becomes real; loss — none
    identified beyond the code delta; cost — a one-token edit to one script line plus a small guard
    test; reversible — fully (revert the line).

  **Option 2 — correct the documentation (change the header).** Remove or rewrite header line 14 so the
  script no longer documents `DUMP_DIR=""` as disabling — for example, noting that disabling means
  running `buster-ripper` directly without `--dump-dir`. Script behaviour is unchanged in every case.
  - Observably changes: the repo stops promising a disable form that does not exist; the documented env
    surface matches actual behavior.
  - Preserves: every script behavior byte-for-byte; all README examples; the direct-CLI call remains the
    only way to run without dumps.
  - Gain / loss / cost / reversibility: gain — the documented surface becomes truthful; loss — the serve
    path permanently has no disable switch (turning dump off requires invoking `buster-ripper` directly
    without `--dump-dir`); cost — a comment-only edit to one file; reversible — fully (restore the
    line). If you want the switch tracked for later, an optional roadmap entry ("serve-script disable
    switch") can carry that desire.

  **Not being decided here.** Nothing about the dump formats, their defaults, the flush policy, the TLS
  posture, or any accepted T001–T006 behavior; nothing in the README (its examples never used the
  disable form); no invariant is created, displaced or weakened — the serve-script flow's verified SAFE
  verdict stands either way (this observation is outside what it checks), and this answer gates no
  milestone.

  **What your answer controls downstream.** Immediately: a small adhoc remediation task at the
  composition boundary carries the chosen fix, plus re-verification of the two facts that established
  the defect and a fresh recomposition of the serve-script flow — both options fire that flow's own
  recorded invalidation trigger, which is expected and cheap. Under Option 2 only: an optional roadmap
  entry may carry the "serve-script disable switch" remainder. Nothing else in the plan moves on this
  answer.

  **Recommendation: none.** This is a preference, not a ratification — the recorded evidence establishes
  the defect but does not rank one fix over the other, so no lean is offered.

  **Question: The serve script's documented disable form (`DUMP_DIR="" …`) does not actually disable —
  honor it in the script, or retract it from the documentation?**

  Answer format: `FK-010: Option 1` or `FK-010: Option 2` — you may modify an option explicitly or add a
  reason in your own words.
  ```
  Final question: The serve script's documented disable form (`DUMP_DIR="" …`) does not actually disable —
    honor it in the script, or retract it from the documentation?
  Options:
    1. Honor the documented disable form — make the line-28 default unset-only
       (`DUMP_DIR=${DUMP_DIR-${HOME}/mllm/prompt-diffs}`), so an explicitly empty `DUMP_DIR` stays empty,
       line 36's `-n` guard suppresses the flag, and the binary runs at its no-dump default. Unset `DUMP_DIR`
       keeps today's default; every README example is unchanged.
    2. Correct the documentation — remove or rewrite header line 14 so the script no longer documents
       `DUMP_DIR=""` as disabling (e.g. note that disabling means running `buster-ripper` directly without
       `--dump-dir`). Script behaviour is unchanged in every case.
  Recommendation: none (preference — recorded evidence establishes the defect but does not rank the options;
    no lean offered or permitted)
  Timestamp: 2026-09-28T16:19:53Z — presented by interview sprint, wave FK-010, order 1 of 1
  Report: `handoff-context.md` §Composition Observation Routing — 2026-09-28 (routing record, closure packet
    and invariant cross-reference); wave index `iterations/020-interview.md` §Round 21 (presentation record)
Resolution:
  User words: "fix the bug, make it to not lie. I do not need the dumps always."
    # Verbatim, unmodified. Unlabeled prose, mapped to this fork as the wave's only fork (wave FK-010, order 1
    # of 1) — no other mapping exists. It is a direct pick, not a delegation: "fix the bug" selects the
    # code-fix arm of the presented honor-vs-retract question, and the third clause ("I do not need the dumps
    # always") names the need only that arm serves — option 2 permanently removes the serve-path off-switch,
    # which would contradict it, so no clause of the answer is consistent with option 2. "Make it to not lie"
    # is satisfied by the code keeping the header's promise (option 1), which is how the user framed the fix.
  Source timestamp: 2026-10-03T22:18:47Z — recorded by interview-reconcile (wave FK-010, order 1 of 1; relayed
    verbatim on the reconcile dispatch path as the user's answer). Presented 2026-09-28T16:19:53Z; the capsule
    was not re-presented or altered between presentation and this answer. Script basis unchanged across the
    window: sha256 f4076d1ceae359f201f4e68d62c41ea7af540a0620a6b65943d405351756fa08 re-verified at recording
    (lines 14/28/36 as cited in the Presented evidence).
  Normative choice: **Option 1 — honor the documented disable form (change the script).** The serve script must
    actually honor `DUMP_DIR="" bash serve_buster_ripper.sh` as "# disable prompt diffing": line 28's default
    applies only when `DUMP_DIR` is unset (`DUMP_DIR=${DUMP_DIR-${HOME}/mllm/prompt-diffs}` — drop the colon),
    so an explicitly empty `DUMP_DIR` stays empty, line 36's `-n` guard suppresses `--dump-dir`, and the binary
    runs at its no-dump default (the package's `config.DUMP_DIR = None` off-mode). Unset `DUMP_DIR` keeps
    today's default; every README example is unchanged; header line 14 stands as written. The user picked the
    option unmodified — no clause added, none waived.
  Scope guard: no broader than the words and the presented question. The answer does NOT change the dump-on
    default when `DUMP_DIR` is unset, the dump formats, the flush policy, the TLS posture, or any accepted
    T001–T006 behavior — the capsule's "Not being decided here" stands. "I do not need the dumps always" is
    the ground for wanting the off-switch; it is not an instruction to flip the default.
  Factual claims: none routed.
    # "I do not need the dumps always" is a statement about the user's own usage needs — carried as the user's
    # need judgment (as FK-009 carried "both ways are minor"), not a proposition about the system. "Make it to
    # not lie" presupposes exactly the already-verified defect (claims 161/162, HOLDS 2026-09-28 on the
    # identical script bytes re-checked above), so it introduces no new falsifiable claim and needs no new
    # verification owner. Post-fix truthfulness is enforced by the remediation task's re-verification clause,
    # not by a pending claim.
  INTERPRETED completions: none — the answer names the pick and its ground directly; no user-visible clause
    was agent-completed.
Impact:
  Plan: landed (registrar batch 2026-10-04) — the adhoc remediation task at the composition boundary is recorded
    (plan-adhoc-task route) as **`T011 · Honor the serve script's documented DUMP_DIR="" disable form`**, status
    `pending`, in `plan-summary.md` §Implementation Plan Summary with its durable task context in
    `handoff-context.md` §Planned Task Context (verification profile included). It carries exactly this projection:
    `serve_buster_ripper.sh` line 28's `:-` becomes the unset-only `-` default
    (`DUMP_DIR=${DUMP_DIR-${HOME}/mllm/prompt-diffs}`) so the header's documented `DUMP_DIR=""` form actually
    disables; the three-state witness (unset → default flag; explicitly empty → no flag; non-empty → the flag);
    and the `seam-S6` member re-verification plus fresh recomposition that the fix's own `Invalidates-On` requires
    (claim-161 and claim-162 named by this record, every other script-pinned member named by the composition).
    Header line 14 and every README example stand unchanged; no `T001`–`T006` clause moves. `MS-004`'s QAP gate is
    unaffected (it tests the TLS posture, not the serve script) and stays the next milestone work — this fork's
    answer gates no task and no milestone. T011 · Honor the serve script's documented DUMP_DIR="" disable form is
    dispatchable on this registration: no further user decision is owed for it, and none is created here.
  Registry: none — no invariant is created, displaced or weakened; `invariants/` untouched (registrar batch
    2026-10-04: landed as written — 0 new / 0 updated / 0 superseded / 0 upheld, no index rebuild). The promoted
    `CO:serve-script-env-to-argv/env-var-flag-crossing` record stands; its pending update is the fresh S6
    recomposition inside the adhoc task (an assumptions/ write, not an invariants/ write).
  Roadmap: none — option 2's optional "serve-script disable switch" entry is moot: the switch becomes real, so
    there is no remainder to carry.
Supersedes: —
Superseded-by: —

## FK-011

State: resolved
Label: preserved-thinking serve default
Owner: assumption-decide
Reserved at: 2026-10-05T14:14:28.821940+00:00
Type: preference
Question: `serve_buster_ripper.sh:30` defaults `PRESERVED_THINKING` to `1`, so **every** run of the documented
  serve path injects `chat_template_kwargs={enable_thinking: true, clear_thinking: false}` into each forwarded
  `/v1/messages` request (`buster_ripper/routes/messages.py:43-50`), while `--preserved-thinking` appears nowhere
  in `README.md` (`grep -ni "preserved.thinking" README.md` → no hits). Must the serve path's outbound sampling
  semantics stay default-on and become a documented, recorded choice — or must they default off and require an
  explicit opt-in per invocation?
Why-yours: "it silently changes what every deployment asks the model to do, and both directions are legitimate
  product postures that no recorded authority picks." The code does what it says, the suite is green, and no
  invariant is crossed — this is not a defect an agent may close by choosing. The only material that could rank
  the two picks is product judgment about the default posture of a thinking-preserving proxy, and the repository
  contains none: no goal, no `US-*`, no `FK-*`, no invariant and no task clause addresses preserved thinking.
Finding and evidence (routed here, not authored here):
  Finding: **F-5** — `iterations/057-codex-review.md` §E (the reviewer's own item, cross-referenced from §B row
    "New CLI surface undocumented"); its packet reads `CLOSURE: question` / `Question required: yes`, which this
    pass re-derived independently against the cited authority rather than adopted.
  Evidence re-verified at this pass (working tree, HEAD `25c2b77f21a06826c5b560c34170052347943614`):
    `serve_buster_ripper.sh:30` `PRESERVED_THINKING=${PRESERVED_THINKING:-1}` with line 42's
    `== "1"` guard appending `--preserved-thinking`; `buster_ripper/routes/messages.py:43-50` the injection;
    `buster_ripper/main.py`'s `--preserved-thinking` option (its help text is the only place the flag is named);
    `README.md` — zero occurrences, against the documented standard `SKIP_TLS_VERIFY`/`--skip-tls-verify`
    (`README.md:90-92`) and `--dump-format` (`:169-194`) already meet.
  Not a regression: the default was added before this session by `847ebbe` ("Add preserved thinking, fix eval
    split, fix strip_code_fences regression"); this session's plan references `PRESERVED_THINKING` only as a
    **guard-shape peer** (`plan-summary.md:144`, `:173` — the T006 clause names `PRESERVED_THINKING :26→:32` as
    sharing the boolean-guard shape with a `1` default), never as an endorsed default.
Why settled authority cannot rank the options: the authority order available here is exhausted. (1) No user
  decision exists — `FK-001`…`FK-010` are resolved or superseded and none mentions thinking; `US-011`/`US-012`
  bound the serve path's *dump* posture only. (2) No active invariant exists anywhere in the registry (the
  eight `PI:*` are `Status: planned` and cover `dump_format` and `upstream_tls`), so nothing protects either
  default. (3) Approved task scope does not reach it (see Affected tasks). (4) The verified evidence is
  symmetrical — it establishes what the script does, not what it should do. (5) Standing workflow rules give
  the routing pass no ranking either. Under the Deterministic Closure Gate's step 7, two materially different
  admissible outcomes remain with no ranking authority → `question`, not `absorbed`. Note the asymmetry below is
  **cost**, not a ranking of intended behavior: it is disclosed so the choice is informed, and it does not make
  one arm the aligned remediation.
Trigger:
  Changed: the external codex review of the whole `refactor/package-structure` branch (057) reached the merge
    boundary and enumerated the branch's undocumented CLI surface; this is the one item in that set whose
    default changes outbound sampling semantics rather than documentation.
  Source: `iterations/057-codex-review.md` §E and §B (cross-reference row "New CLI surface undocumented");
    re-derived this pass from `serve_buster_ripper.sh`, `buster_ripper/routes/messages.py`, `README.md` as read
    at HEAD `25c2b77…`.
  Was: the item was invisible to 02.00's own gates — the dump/TLS plans never observed the serve script's
    thinking default, and `pyproject.toml:30-34` omits `buster_ripper/routes/*` from coverage, so no gate in
    this session watched the injection site.
If-wrong: both arms are single-file and reversible, and neither is merge-blocking for the branch: choosing
  wrongly costs either an undocumented default that keeps shipping (option 1) or a silent behavior change for
  every existing deployment plus a recomposition debt (option 2). They are **not** symmetric in cost, and the
  asymmetry belongs to the record:
  - Option 1 changes no behavior at all; the serve script's argv word set is unchanged, so the promoted
    `CO:serve-script-env-to-argv/env-var-flag-crossing` basis stays current at `sha256:cb7c0a5c…` and the four
    whole-list argv equalities in `tests/test_serve_script_argv.py` stay green untouched.
  - Option 2 moves line 30's default, which fires that record's **own** `Invalidates-On` ("line 30's
    `PRESERVED_THINKING` default moving off `1` or line 42's guard changing"), so it owes a fresh member
    re-verification plus a 25-member `seam-S6` recomposition before `seam-S6` may be cited as current again —
    the same debt T011 and T012 each paid. It also makes `tests/test_serve_script_argv.py`'s `_THINKING`
    constant conditional, which row 201's own `Invalidates-On` names as the trigger for re-verifying every
    pinned argv equality, and it silently stops preserving thinking for anyone relying on today's default.
Affected tasks: **none in this session** — and that is the packet's point. `T001`–`T006` and `T011`/`T012` carry
  no preserved-thinking clause; `T006`'s serve-script clause is the `SKIP_TLS_VERIFY` boolean guard; `T012`
  rewrote the *assembly* of line 42's append (quoting/array mechanism only, the word set byte-identical), never
  line 30's default. The new remediation waves allocated by this pass (`T013` header parity, `T014` image
  surface) do not touch the serve script either. So this Fork gates no task of 02.00 and no milestone, and
  answering it does not reopen any accepted behavior.
Options:
  1. **Keep the default on; make it a documented, recorded choice.** Add `--preserved-thinking` (and the
     `PRESERVED_THINKING` env var, with its default stated) to `README.md`'s serve-script section to the
     standard `--dump-format` and `--skip-tls-verify` already meet, and record the default-on posture as a
     decision. No code or script line moves; existing deployments are unaffected.
  2. **Default off; require an opt-in per invocation.** Change line 30 to `PRESERVED_THINKING=${PRESERVED_THINKING:-0}`
     so thinking preservation is requested explicitly, document the flag and the env var, and accept the
     recorded consequence: every current deployment stops preserving thinking on upgrade until it sets
     `PRESERVED_THINKING=1`, and the `seam-S6` recomposition plus the pinned-argv witness updates above become
     owed work.
Recommendation: none (preference — the recorded evidence establishes what the script does and what each arm
  costs, and does not rank the two postures; no lean is offered or permitted).
Presented: 2026-10-18T03:58:36Z · wave 1 · order 1 of 1 · dependencies: none · report: iterations/114-sprint.md
  Option labels: A · keep default on, document it | B · default off, require opt-in
  Recommendation: none
  Predecessor / supersession: none (original record, no successor; no prior FK asked this question)
  Advisory report: iterations/114-sprint.md
  Capsule (exact rendered text; the fence is storage only):
````
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
````
Not being decided here: the flag itself stays available either way (removing `--preserved-thinking` is a
  rejected candidate in the source packet — it leaves approved behavior unreachable, so it is not an option
  here); the injection code at `messages.py:43-50` is not in question (it does what it says); nothing about the
  dump formats, the flush policy, the TLS posture, the header-parity or image-surface remediations, or any
  accepted `T001`–`T006`/`T011`/`T012` behavior moves on this answer, and no `MS-004` verdict is disturbed.
Resolution:
  User words: "A shit why you even created this fork?"
    # Verbatim, unmodified. "A" selects Option A of the two presented; the remainder is an objection
    # to this Fork having been created at all — not a third option, not a modification of Option A, and
    # not an answer to the question's B arm. Recorded as the user's process judgment on the question's
    # admission, kept separate from the normative choice and from the routing-quality note below.
  Source timestamp: 2026-10-18T15:40:52Z — recorded by interview-reconcile (wave 1, order 1 of 1); the settled
    answer and the user's words were relayed verbatim on the reconcile dispatch path. Presented
    2026-10-18T03:58:36Z ([114-sprint](iterations/114-sprint.md)); the capsule was not re-presented or
    altered between presentation and this answer. Basis re-verified at recording on `HEAD 897bd5c` (working tree
    clean): `serve_buster_ripper.sh:30` `PRESERVED_THINKING=${PRESERVED_THINKING:-1}` and `:42`'s `== "1"`
    guard, script sha256 `c58256fbec793c520720c9e86fee8740653939275b508f50b212ab60bd0908af`;
    `git diff 4945899..897bd5c -- serve_buster_ripper.sh README.md buster_ripper/ tests/` is empty, so the
    presented basis is unchanged; `README.md` still has zero `preserved.thinking`/`PRESERVED_THINKING` hits
    (`grep` exit 1); `buster_ripper/main.py:107` is the `--preserved-thinking` option.
  Normative choice: **Option A — keep the default on; make it a documented, recorded choice.** The serve path's
    outbound sampling semantics stay default-on: line 30's `PRESERVED_THINKING=${PRESERVED_THINKING:-1}` and line
    42's `== "1"` append stand, `buster_ripper/routes/messages.py:46-47` keeps injecting
    `enable_thinking=true, clear_thinking=false`, and `README.md`'s serve-script section gains
    `--preserved-thinking` plus the `PRESERVED_THINKING` variable with its `1` default, at the standard
    `--skip-tls-verify` (`README.md:90-92`) and `--dump-format` (`:169-194`) already meet. This Resolution is
    the canonical decision record for the default-on posture; the docs task cites it and creates no second record.
    The user picked Option A unmodified — no clause added, none waived.
  Scope guard: no broader than the words and the presented question. The answer does NOT move line 30's default,
    the injection site, the flag's availability, or any dump-format, flush-policy, TLS, header-parity or
    image-surface behavior — the capsule's "Not deciding here" stands. "Why you even created this fork?" is the
    user's objection to this Fork's admission; it is not an instruction to widen Option A, and it names no other
    case or fork.
  Factual claims: none routed.
    # The words contain no proposition about the system. The two facts the routing-quality note relies on are
    # already-recorded, re-verified evidence rather than new user claims: Option A moves no script byte, so
    # `CO:serve-script-env-to-argv/env-var-flag-crossing` stays current at `sha256:cb7c0a5c…`; Option B's cost is
    # the 25-member `seam-S6` recomposition this record's own `If-wrong` already states (member count confirmed at
    # [114-sprint](iterations/114-sprint.md): 25 claim rows, `VERDICT: SAFE`).
  INTERPRETED completions: none — the answer names the pick directly; no user-visible clause was
    agent-completed.
  Routing-quality note (for the FK owner, `assumption-decide`; not a change to the answer): the objection is
    well-founded on this session's own ranking authority. Base parity preserves the shipped default-on behavior, and
    this record's own `If-wrong` already states that Option A changes no behavior at all while Option B changes every
    deployment's behavior and owes a fresh member re-verification plus a 25-member `seam-S6` recomposition. That
    asymmetry ranks Option A under the Deterministic Closure Gate's smallest-complete-remediation rule — step 4's
    scope tie-break (prefer the smaller change when correctness is equivalent or the larger change carries a concrete
    countervailing risk) and step 5 (exactly one admissible remediation → `absorbed`) — so the Fork should have been
    `absorbed → A · document the default` rather than escalated as an unranked preference. Recorded for the owner;
    Reconcile neither re-decides admission nor edits the record's own contrary paragraph ("the asymmetry below is
    cost, not a ranking of intended behavior"), which is the owner's to revisit.
  Impact:
    Plan: landed (registrar batch 2026-10-18) — one adhoc task on the `plan-adhoc-task` route, recorded as
      **`T023 · document `--preserved-thinking` and `PRESERVED_THINKING` in `README.md`'s serve-script section`**,
      status `pending`, in `plan-summary.md` §Implementation Plan Summary, with its durable task context in
      `handoff-context.md` §Planned Task Context (assumption impact `lite:small`, verification profile `parallel`
      included). The next free top-level ID was `T023` (`plan-adhoc-task` assigned it; no `T023` existed in
      `plan-summary.md`/`plan-completed.md`). The row carries exactly this projection: **document
      `--preserved-thinking` and `PRESERVED_THINKING` (with its `1` default) in `README.md`'s serve-script
      section** to the standard `--skip-tls-verify` (`README.md:90-92`) and `--dump-format` (`:169-194`) already
      meet, citing this Resolution as the recorded default-on posture. Done when: (1) the section names the flag and the
      variable, states that preserved thinking is on by default (`enable_thinking=true`, `clear_thinking=false`) and how to
      turn it off (`PRESERVED_THINKING=0 bash serve_buster_ripper.sh`); (2) it names `FK-011 · preserved-thinking
      serve default` as the decision and creates no second record; (3) `git diff --stat` shows `README.md` only —
      `serve_buster_ripper.sh` stays at sha256 `c58256fb…` and `tests/test_serve_script_argv.py` is untouched.
      Validation: read-back against `serve_buster_ripper.sh:30`/`:42` and `buster_ripper/main.py:107`, plus the
      doc-standard comparison with the two existing entries; no test run is owed for a docs-only change. Milestone:
      none — `MS-001` is resolved and stays resolved, and this task moves no milestone verdict (same shape as
      `T011` · serve-script disable form and `T022.001` · T022 drain bookkeeping). Assumption impact `lite:small`;
      Verification profile `parallel` — docs-only, single file, settled contract, easy recovery. **`T023` · document
      `--preserved-thinking` and `PRESERVED_THINKING` in `README.md`'s serve-script section is dispatchable on this
      registration: no further user decision is owed for it, and none is created here.**
    Registry: none — no invariant is created, displaced or weakened; `invariants/` untouched (registrar batch
      2026-10-18: landed as written — 0 new / 0 updated / 0 superseded / 0 upheld, no elevation and no index
      rebuild). The promoted
      `CO:serve-script-env-to-argv/env-var-flag-crossing` record stands current: Option A moves no script byte, so
      its `Invalidates-On` does not fire and no member re-verification or recomposition is owed.
    Roadmap: none — the answer keeps the shipped baseline, adds no capability and retires none, so no roadmap row
      moves (registrar batch 2026-10-18: landed as written — no roadmap row moved).
Supersedes: —
Superseded-by: —

## US-017

State: recorded
Owner: registrar
Reserved at: 2026-10-19T00:03:27Z
User words: "Keep only gate evidence"
Source timestamp: 2026-10-18 (session-close step 16 evidence rulings; the ruling was relayed verbatim in the registrar dispatch's Sources field)
Question and presentation: Completion item 9 requires every data row of `evidence/INDEX.md` to carry a `keep` or `remove`
  disposition. The index held 39 data rows: 18 already carried a verdict or a prose disposition but no keep/remove ruling,
  and 21 carried no disposition at all. The rows split between regenerable test/read-back transcripts and non-regenerable
  measurement records. The ruling was put to the user as a per-row keep/remove question; the user answered with the rule quoted
  above rather than a row-by-row list.
Normative choice: **Keep only gate evidence.** Milestone and gate transcripts, and boundary review records, are kept. Earlier
  per-task artifacts are removed, and each dropped row retains a pointer to its iteration report or the command that reproduces
  the observation, so the record of what was shown survives the artifact. Regenerable rows — the `T024` suite, the `T023` README
  read-back, and the 2026-10-17 / 2026-10-08 gate and hermetic runs — are dispositioned `remove` under completion item 9's
  agent rule, each keeping the named command and its one-line result in the row. `remove` rules on the artifact and not on the
  row: the bytes stay on disk until `session-squash`, which deletes them and keeps the row marked removed. No artifact was
  deleted at this close gate.
  # Scope guard: the words are a record-keeping rule, not a product, plan or milestone choice. They do not authorise deleting
  # any production, test, config, plan, invariant, decision or roadmap byte, and they do not move a task outcome. The explicit
  # per-row applications recorded in `evidence/INDEX.md` are: `keep` — `ms001-reboundary-091-gate-attempt`,
  # `090-01-yield-interrupted-draft`, `t019-002-fail-first-and-replays`, `cr4-probe-matrix`, `claim-425-charset-arm-reach`,
  # `ms001-gate-and-upstream-adjudication`, `ms002-comparison-record` (2026-10-06), `ms004-tls-posture-gate`,
  # `ms002-comparison-record` (2026-09-22), and the milestone boundary review packet `codex-review-post-071-packet`;
  # `remove` — `t024-wave-suite`, `t023-preserved-thinking-readme`, the five 2026-10-17 gate/hermetic rows,
  # `ms001-reboundary-091-hermetic`, `s15-reach-and-base-comparison`, `t013-header-parity-wave`,
  # `t012-yield-argv-replay`, `t002-collapsed-artifact`, and the remaining per-task Forward transcripts.
Factual claims: none — the words are a record-keeping instruction and assert no falsifiable proposition about the
  repository. The counts in `Impact` are the registrar's own reading of `evidence/INDEX.md`, not user assertions.
Verification: no factual claim is routed; the ruling's application was read back from the 39 `Disposition` cells in
  `evidence/INDEX.md` at this recording (10 `keep` / 29 `remove`, zero undecided), and no artifact byte was moved.
Consequence: none — the ruling changes what the archive keeps, not what the product does; every production, test, config,
  plan, invariant, decision and roadmap byte stands.
Impact:
  Plan: none — completion item 9 is an evidence-disposition check; the plan, task outcomes and milestone verdicts stand.
  Registry: none — no invariant is created, displaced or weakened; `invariants/` untouched.
  Roadmap: none — no roadmap row moves.
Legacy source: none
Supersedes: —
Superseded-by: —
