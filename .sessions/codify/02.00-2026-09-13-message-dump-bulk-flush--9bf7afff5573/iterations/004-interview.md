# 004 — Interview (anchor)

## Orchestrator Prompt (verbatim)

> Execute the invite-interactive skill for this session. Arguments: $ARGUMENTS
>
> [The remainder of the prompt was the `invite-interactive` skill body plus the
> `codify-agents`, `critical-thinker`, and `codify-user-prompt` skill bodies.
> `$ARGUMENTS` was empty. Session ID was supplied out-of-band as
> `02.00-2026-09-13-message-dump-bulk-flush--9bf7afff5573`; Question Mode is
> `deferred` per session metadata.]

## Session Type / Mode

- New parent session (`02.00`), no `parent_session_uid` → parent coordination
  contract does not apply.
- Invite runs once for this planning round. This report is the living anchor;
  subsequent `interview` invocations append to it.

## Context Conflict Detected

None. `decisions.md`, `plan-summary.md`, `handoff-context.md` and
`iterations/003-decide.md` agree with each other and with source.

## QA Infrastructure Check (Step 1.5)

- **Plan is QA-testable**: yes. The milestone matrix carries two QA author passes
  (`QAP-MS001`, `QAP-MS002`) and the validation points require HTTP requests to a
  running proxy (`/v1/messages`), a real subprocess signal (SIGINT/SIGTERM), and
  post-exit inspection of `DUMP_DIR/<sid>/`.
- **Project QA skill present**: **NO**. `.agents/skills/qa-project/SKILL.md` does
  not exist, and the repository has no `.agents/` or `.claude/skills/` directory
  at all.
- **Action**: the QA setup question is surfaced to the user in this round's
  `## User-Facing Presentation` (see `## QA Infrastructure Setup` below). The
  skill file is NOT written yet — the answers must be verified before writing.

### QA Infrastructure Setup

Proposed configuration derived from source, pending user confirmation:

| Item | Proposed | Source / verification |
|---|---|---|
| Start command | `.venv/bin/buster-ripper --host 127.0.0.1 --port <port> --upstream <url> --dump-dir <tmpdir>` | `.venv/bin/buster-ripper` exists (verified `ls`); CLI flags verified in `buster_ripper/main.py:12-61` |
| Base URL | `http://127.0.0.1:<port>`, endpoint `POST /v1/messages` | `main.py:126` `web.run_app(app, host, port)` |
| Test framework | pytest + pytest-asyncio + pytest-aiohttp + pytest-mock | `pyproject.toml:19-30`; `.venv` has pytest 9.0.2, aiohttp 3.13.3 (verified) |
| Test file location | `tests/qa/` (new), kept separate from the mocked `tests/` suite | `tests/` exists; `conftest.py` is empty ("no shared fixtures") so nothing conflicts |
| Teardown | Send SIGTERM/SIGINT to the subprocess and wait for exit; for this session the signal IS the feature under test | `serve_buster_ripper.sh:47` uses `exec`; `server.py:60` `on_cleanup` is the shutdown seam |
| Prerequisites | An upstream. Real vLLM (`localhost:30000`) is likely unavailable; a local aiohttp stub upstream can stand in | `main.py:13` default upstream `http://localhost:30000` |

**Open sub-question for the user**: whether QA may substitute a local stub
upstream for the real vLLM backend, or whether the QA harness must require a live
vLLM. This changes whether the milestone gate can run unattended.

**Verification performed** (per Step 1.5): `uv` present at `/opt/homebrew/bin/uv`;
`.venv/bin/buster-ripper` present; `.venv/bin/python -c "import aiohttp, pytest"`
succeeds; `tests/` exists and is writable. All pass. The skill file is written
only after the user's answers arrive and pass the same checks.

## Landscape

Full detail for later agents. User-facing version is returned to the orchestrator
under `## User-Facing Presentation`.

### The goal

In dump mode (`--dump-dir` on `/v1/messages`), buster-ripper currently writes to
disk on every intercepted request. The user wants captured messages held in memory
and written out in bulk — fewer write operations — and wants Ctrl-C (SIGINT) /
SIGTERM to flush the buffer before the process exits. Durability of in-flight
messages is explicitly NOT a requirement (`US-001`).

### The two policy dials

Everything in this change is code-determined except two independent policy
choices:

| Dial | Positions | Carried by |
|---|---|---|
| **D1 — When does the buffer hit disk?** | (a) on a pending-turn threshold **and** at shutdown; (b) at shutdown only | `FK-001` (preference, no recommendation) |
| **D2 — What does a flush write?** | (a) the documented per-turn layout; (b) one collapsed per-session file | `FK-002` (ratification, recommend preserve) |

### The three options (combinations of the dials)

- **Option 1 — threshold + documented layout**: buffer per session; flush at a
  turn threshold and at shutdown; output format unchanged. ~50% fewer writes
  (2N → N + ⌈N/K⌉). Bounded loss window; `index.txt` readable during the
  session. More code than Option 2.
- **Option 2 — shutdown only + documented layout**: same buffering, written only
  at graceful shutdown. Fewest writes (N + 1); zero disk I/O on the request path.
  But a SIGKILL/crash/OOM loses the whole session, and `index.txt` does not exist
  for the session's whole life.
- **Option 3 — threshold + collapsed single file**: append pending turn records to
  one per-session file. ~100% reduction. Breaks the documented output contract
  (`README.md:146-162`), removes the per-turn diff artifacts, requires a README
  rewrite and an explicit override of `EI:.../output_layout` and
  `EI:.../sequential_diff_chain`; a fresh `RB-*` Reuse Evidence row is needed.

### Framing surprises the user has not seen

1. The write-reduction ceiling under the documented layout is **~50%**, not ~100%.
   Only the per-turn `index.txt` append can collapse; `turn_NNN.diff` is one file
   per turn. N turns: 2N writes → N + 1.
2. Shutdown-only flushing loses the **whole session** on a hard kill, not an
   in-flight window, and leaves `index.txt` nonexistent for the session's life —
   the documented reading interface (`README.md:146-162`) does not work while the
   proxy runs.
3. Buffering also removes **blocking disk I/O from the request path** —
   `dump_turn` is synchronous and called inline from `handle_messages`
   (`routes/messages.py:55-59`). Any option fixes this; the write count is the
   smaller win.

### Approach status

- `FK-001` (D1) is a **preference**: the user's "we don't care about making this
  super robust" and the documented `index.txt` reading workflow point opposite
  ways. No source or execution settles it. **No recommendation.**
- `FK-002` (D2) is a **ratification**: the code-and-docs-determined answer is to
  preserve the documented layout, but accepting a ~50% ceiling against a stated
  "in bulk" goal is the user's act to own. **Recommend preserve.**

### Resulting plan overview (organize's candidate list; final task-out is PLAN-SELECTED work)

1. `T001` — per-session pending-write buffer in `dump.py`; `turn`/`prev_text`
   advance at append time.
2. `T002` — bulk flush routine: write pending turn files in order, then
   `index.txt` once; no-op when `DUMP_DIR is None`; must not raise.
3. `T003` — wire the flush into `server.on_cleanup`, wrapped; add the threshold
   trigger only if `FK-001` resolves that way.
4. `T004` — tests (`tests/test_dump.py` + an `on_cleanup` flush case); total
   coverage must stay ≥ 80% (headroom: ≤ 4 uncovered new statements).

### Validation overview

- `VP-T001-writes-deferred`, `VP-T001-session-isolation`,
  `VP-T001-dump-failure-isolated`, `VP-T002-layout-preserved`,
  `VP-T002-diff-chain`, `VP-T002-write-count-reduced` → validated at `MS-001`
  (constituent tasks `T001`, `T002`), authored by `QAP-MS001`.
- `VP-T003-shutdown-flush` → validated at `MS-002` (task `T003`), authored by
  `QAP-MS002`; requires subprocess + real signal, no in-process mock can prove it.
- `VP-T003-passthrough-regression` → `MS-003` (task `T004`); the existing suite
  is the gate.

### Invariant impacts

- No invariant registry exists in this repository; collect's six `EI:` entries are
  session-local drafts, all re-assessed as `intended`.
- Options 1 and 2 preserve all of them. **Option 3 would override
  `EI:.../output_layout` and `EI:.../sequential_diff_chain`** and would need an
  explicit override register plus a fresh `RB-*` row.
- New planned invariants (themes): buffered append preserves turn order; flush
  preserves the output layout; flush cannot raise out of `on_cleanup`; flush is
  idempotent per pending batch.

## Topic 1: Flush trigger policy (`FK-001`)

Presented first. Preference, no recommendation. See `## Question Asked (verbatim)`
below.

## Question Refinement: Flush trigger policy

Draft → challenge → verify → refined.

- **Assumption challenged**: that the trigger choice is independent of the payload
  choice. It is not fully — Option 3 pairs the threshold trigger with a collapsed
  file, but the two dials are logically orthogonal (a shutdown-only collapsed file
  is coherent). Verified against `003-decide.md` §Options: the option list is
  explicitly "three combinations of two independent dials". Keeping the trigger
  question separate is correct.
- **Edge case surfaced**: a shutdown-only flush leaves `index.txt` **nonexistent**
  for the session's entire life, not merely stale. `README.md:146-162` documents
  reading it live. This is the concrete cost that must appear in the question text.
- **Trade-off completeness**: the "super robust" counter-argument (the user said
  they do not care about durability) is real and is why this is a preference with
  no recommendation rather than a ratification. Stated in the question.
- **Downstream dependency made visible**: the trigger choice decides whether `T003`
  gains a threshold branch and whether the `VP-T002-write-count-reduced` target is
  `N + ⌈N/K⌉` or `N + 1`.
- **Verification against session data**: `US-001` scopes the accepted loss to "in
  flight messages"; `FK-001`'s `If-wrong` records that shutdown-only turns that
  into whole-session loss. No session data contradicts the draft.

## Topic Queue

One FK per turn. Ratifications are not batchable.

### Topic 2 — `FK-002` (ratification, recommend preserve)

- **FK**: `FK-002`
- **Question**: "Preserving the documented per-turn dump layout caps the write
  reduction at about 50% (2N writes become N+1 for N turns). Do you accept that
  ceiling and keep the documented layout, or collapse the session dump into a single
  append-only file to reach roughly 100%?"
- **Options**:
  - "Keep documented layout (recommended)" — `turn_000.json` / `turn_NNN.diff` /
    `index.txt` stay exactly as documented. N turns go from 2N writes to N+1 —
    roughly a 50% reduction. No README change, no reading-workflow change, the
    per-turn diff artifacts remain directly observable on disk.
  - "Collapse to one file" — append every pending turn record to a single
    per-session file. Roughly 100% reduction (one write per flush). But the
    per-turn diff files and `index.txt` stop existing on disk, `README.md:146-162`
    must be rewritten, diffs become a read-time concern, and two session invariants
    must be explicitly overridden.
- **Recommendation**: keep the documented layout and accept the ~50% ceiling.
- **Depends on**: `FK-001` only for option *framing* (the collapsed-file variant
  is naturally paired with the threshold trigger). The question itself is orthogonal;
  it may be asked as drafted after `FK-001` resolves. **Stub status: complete.**
- **Invalidates if answered "collapse"**: `VP-T002-layout-preserved` and
  `VP-T002-diff-chain` are replaced by a single-file-format validation point;
  `RB-001` must be re-run; an override register entry is required.

### Topic 3 — QA infrastructure setup (Step 1.5; not an FK)

- **Question**: the six QA setup items in `## QA Infrastructure Setup`, plus the
  open sub-question of stub-vs-live upstream.
- **Status**: surfaced in this round's `## User-Facing Presentation`. **Must be
  answered before Forward reaches its QA infrastructure gate.** The skill file is
  written only after the answers are verified.
- **If unanswered**: `forward-qa` will refuse at `QAP-MS001` / `QAP-MS002`.

### Topics resolved by decide — NOT queued

- Turn-0 buffering → Forward discretion footnote (buffer uniformly). Not a user
  question.
- `dump.py:63` bare `assert` → fold a real guard into the same increment. Not a
  user question.
- Scope (`--dump-dir` on `/v1/messages` only) and "no background flush timer" →
  resolved from code.

## Question Asked (verbatim)

See the returned `## User-Facing Presentation` — the canonical copy is the
`Presented` block written into `decisions.md` for `FK-001`.

## Topic 1 — Answer Recorded (round 1)

**User words (verbatim)**:

> Okay, good points. We need to have both time-based and data volume-based flash
> frequencies. Do you think it's feasible to have both?

The user did **not** pick either presented position. They answered the causal
question — *when should the buffered dump be written to disk?* — with a third
position that the presented option set did not contain: **both** a time-based
trigger and a data-volume trigger, in addition to the shutdown flush. Recorded
as `FK-001`'s `Resolution` with normative choice "flush on both a time interval
and a data-volume threshold, plus the graceful-shutdown flush".

The answer also carries a direct question to the agent — *"Do you think it's
feasible to have both?"* — answered in `## Feasibility Answer` below.

### Planning gap this answer exposes

`handoff-context.md` records, from decide: *"Resolved from code: no background
flush timer is needed."* That call is now falsified by user authority. The
reasoning behind it is visible in the landscape: `US-001`'s "we don't care about
making this super robust" was read as "the loss window does not matter", so the
only trigger considered was *when to write*, never *how stale a buffered turn may
get*. The user's requirement is about **staleness**, not durability — they want
`index.txt` to stay current enough to read while the proxy runs. A durability
framing cannot express that, which is why the option set was incomplete.

This is a **scope expansion**, not a correction of a wrong binding: nothing in
the Anchor Map is invalidated. But the trigger cluster was never planned to
agent-ready depth under this requirement.

## Feasibility Answer

**Yes — both triggers are feasible, and the repository already contains the
exact pattern needed for the time-based half.** Grounded in source:

| Requirement | Mechanism | Evidence |
|---|---|---|
| Data-volume trigger | a counter/byte check inside `dump_turn` before it appends; no background task needed | `buster_ripper/dump.py:61-109` — `dump_turn` is the single append point and already holds `SessionState` |
| Time-based trigger | one `asyncio.create_task` running a `while True: await asyncio.sleep(interval)` loop, stored on `app` and cancelled in `on_cleanup` | `buster_ripper/server.py:25-34` (`spawn_kv_cache_monitor`) and `:60-72` (cancellation), driving `buster_ripper/compaction.py:28-...` (`poll_kv_cache`). This is a working, in-repo instance of the exact shape |
| Shutdown flush | already planned (`server.on_cleanup`) | `buster_ripper/server.py:60` |

Two properties make the combination safe rather than merely possible:

1. **Both triggers converge on one flush routine.** The volume trigger calls it
   from `handle_messages`; the timer calls it from the event loop. `dump_turn`
   and every write it performs are **synchronous**, so on asyncio's single thread
   a flush runs to completion without yielding — no lock, no interleaving, no
   partial flush is reachable. This is a property of the code as written, not an
   assumption, but it is a property the increment must *preserve*: introducing an
   `await` inside the flush would silently create the interleaving window.
2. **No new dependency.** The timer reuses the existing task/app-state/cancel
   lifecycle; the only new artifact is one coroutine and one `app` key.

**The caveat the user should hear now.** The time-based trigger is not free — it
trades directly against the write-reduction goal (`G-001`), which is the stated
purpose of the whole change. Every flush writes the pending turn files plus one
`index.txt` append. For N turns:

- documented layout: `2N` writes today → `N + F`, where `F` = number of flushes
- collapsed layout: `2N` → `F`

A short interval makes `F` approach `N`, which drives the documented-layout
reduction toward **zero** — i.e. an aggressive interval can erase the benefit the
change exists to deliver. The interval is therefore not a free parameter; it is
the dial that sets both the staleness bound *and* how much of the win survives.
That interaction is why the interval and threshold values need real option
analysis rather than a guessed default, and it is the reason for the
`REPLAN-INCREMENT` below.

**QA consequence.** A fixed wall-clock interval is not deterministically testable
through the project harness: a milestone gate cannot wait out a real timer and
still be reliable. `VP-*` coverage of the time trigger requires the interval to
be **injectable** (config/CLI/env), so a gate can set it to a sub-second value,
or a clock seam the harness can advance. This is a QA-capability requirement that
must land in the plan before the first milestone that depends on it — see
`## QA Capability Implication`.

## Delta-Check (queue re-evaluation after this answer)

Per the fast-cycle contract: does the answer invalidate or reorder any queued
question?

| Queued item | Verdict | Why |
|---|---|---|
| `FK-002` (payload: documented layout vs collapsed file) | **not invalidated, but its premise is refreshed** | The dial is orthogonal — the choice is unchanged. But the write-count arithmetic it cites changes from `N+1` to `N+F`, and `F` is now a free variable. The *comparison* is interval-robust: collapsed always saves exactly `N` writes versus documented at the same `F`. Presentation refreshed to state the comparison instead of a fixed number; the choice itself is untouched |
| QA infrastructure setup (Topic 3) | **not invalidated; one item added** | The harness now additionally needs the flush interval injectable. Added to `## QA Capability Implication` |
| `VP-T002-write-count-reduced` target | **reordered / needs re-derivation** | Its stated target (`≤ N + 1`) is now only the shutdown-only case. The target must be expressed in terms of the actual flush count `F` |
| `T003` scope | **materially enlarged** | Gains a timer coroutine, an `app` state key, a cancellation branch in `on_cleanup`, and tests. The `≤ 4 uncovered new statements` coverage headroom (`plan-summary.md` §Goals and Constraints) gets materially tighter — this is a real constraint, not a footnote |
| `G-001` success criterion | **needs re-derivation** | "file-write operations < 2N" is still satisfiable but no longer captures the trade the interval controls |

**Full queue refinement is required, so this is a `REPLAN-INCREMENT`, not an
inline adjustment** (see the returned summary). It is scoped to the flush-policy
cluster only: the trigger's new parameters, the timer mechanism, the refreshed
`VP-T002-write-count-reduced` target, the coverage headroom, and the timer's
`RB-*` Reuse Evidence row. Settled topics (`T001`, `T002`, the Anchor Map
bindings, the milestone skeleton) stay untouched.

`FK-002` is asked in the same turn because the exhaust-before-rerouting rule
requires the sweep to complete before rerouting, and `FK-002`'s choice is
independent of the trigger parameters. Its refreshed presentation is written into
`decisions.md` before this turn returns.

## Question Refinement: Payload layout (`FK-002`, refreshed presentation)

`FK-002` has never been presented — its `Presented` block was empty, so this is a
first presentation, not a change to recorded text.

- **Draft → challenge**: the drafted text said "2N writes become N+1 for N turns".
  That number was the *shutdown-only* figure and was already slightly wrong for
  the threshold option (`N + ⌈N/K⌉`). After this round it is wrong for both.
  Presenting it would put a stale quantity in front of the user.
- **Verified against session data**: the arithmetic above (`N + F` vs `F`) is
  derived from `dump.py:82`, `:95` (one `write_text` per turn file) and `:104-106`
  (one index open+write+close per turn) — the three write sites `RB-001` already
  identified.
- **Trade-off completeness**: the collapsed option is not merely "fewer writes" —
  it removes the per-turn diff artifacts the tool exists for and breaks the
  documented reading workflow (`README.md:146-162`, verified this round). Both
  costs stay in the presentation.
- **Downstream made visible**: choosing "collapse" replaces
  `VP-T002-layout-preserved` and `VP-T002-diff-chain`, requires an override
  register for `EI:.../output_layout` and `EI:.../sequential_diff_chain`, and
  requires a fresh `RB-*` row. Stated in the option text.
- **Recommendation preserved**: decide's recorded recommendation (preserve the
  documented layout) is kept unchanged. The refreshed framing does not alter the
  basis for it; if the trigger resolution weakens it, that is decide's call to
  revise, not the presenter's.

## QA Capability Implication

The QA capability gate (`interview` → QA Capability Gate) now has a **third**
required capability, alongside the ones already surfaced by invite:

- The harness must be able to make the flush interval **deterministic** — either
  by configuring it to a sub-second value at process start, or by a clock seam
  the test can advance. Without this, no `TC-*` can prove the time trigger
  without a sleep-and-hope race, which is a flaky gate.

Because `.agents/skills/qa-project/SKILL.md` does not exist yet, this becomes an
additional item in the QA setup answers rather than a migration of an existing
declaration. It is added to Topic 3's queued question.

## Next Step

1. The orchestrator dispatches the `REPLAN-INCREMENT` for the flush-policy
   cluster (trigger parameters: interval value/basis, volume-threshold basis,
   timer mechanism + its `RB-*` row, refreshed `VP-T002-write-count-reduced`,
   coverage headroom re-check). Decide authors the new successor FKs.
2. In the same turn, `FK-002` (payload layout, refreshed presentation) is
   presented — it is independent of the trigger parameters.
3. Topic 3 (QA infrastructure setup, now including interval injectability) is
   asked after `FK-002`.
4. After the new trigger-parameter FKs resolve and `FK-002` resolves, decide runs
   PLAN-SELECTED on the chosen option.

---

# Round 2 — `FK-002` answered

## Topic 2 — Answer Recorded (round 2)

**User words (verbatim)**:

> Another good point, we need to change the flash format. So our bulks would be
> single write, not a bunch of writes which would kill the purpose and any
> benefit of saving SSD.

The user answered the causal question — *what does a flush write?* — by taking the
second presented option, **collapse to one file**, and rejecting the recommended
first option. Recorded as `FK-002`'s `Resolution`: the dump payload format changes
so that one bulk flush is one write; the documented per-turn layout
(`turn_000.json` / `turn_NNN.diff` / `index.txt` as separate per-turn files) is
not preserved.

Two readings of the answer were considered and the second rejected:

- **"Keep per-turn files, batch only the index append"** — rejected. A bulk of K
  turn files cannot be a single write while per-turn files exist. The user's
  words are explicitly about the *bulk* being one write, and the reason given is
  write count, so the per-turn file split is exactly what has to go.
- **"Change the flush format so a flush is one write"** — accepted. This is the
  presented "collapse to one file" position, and it is the only reading that
  satisfies "our bulks would be single write".

**This is a rejection of a `ratification`, and it is recorded as a choice, not a
falsification.** The factual basis recorded for the recommendation — a preserved
layout caps the reduction at ~50% — still holds. The user declined the ceiling and
chose the ~100% alternative with its costs visible in the presented option text
(per-turn diff artifacts stop existing on disk, `README.md:146-162` must be
rewritten, two session invariants must be overridden, a fresh `RB-*` row is
required). No factual claim is `DOES-NOT-HOLD` here, so no scoped-consequence
acceptance is needed.

The user's message also carries three falsifiable propositions, recorded as
`Factual claims` on the FK with `assumptions PENDING` (per the User Authority
Boundary — a user-originated factual claim gets the same verification treatment
as an agent's, and may not enter the plan as established fact):

1. A preserved per-turn layout cannot deliver a single write per bulk.
2. Fewer file-write operations reduce SSD wear.
3. "flash format" means "flush format".

Claim 2 is the one the whole decision rests on and is **not** source-checkable in
this repository. It is recorded, not dismissed and not assumed: the normative
choice (one write per flush) stands on its own even if the SSD mechanism is
narrower than the phrasing implies, but no downstream agent may cite the SSD
rationale as established fact until a verdict exists. Forward Collect's
assumptions pass owns it.

### Delta-check (queue re-evaluation after this answer)

| Queued item | Verdict | Why |
|---|---|---|
| `FK-A` (flush interval **value**) | **premise invalidated — must be re-authored before asking** | The question itself survives (how long may a buffered turn stay unflushed?). Its recorded `If-wrong` framing does not: `½·(1 − F/N)` reaching zero at `F = N` is the *documented-layout* figure. Under the collapsed layout the cost is `F` writes for N turns, so a short interval no longer erases the benefit — it only adds a linear write cost. Decide must revise the framing before `FK-A` exists as a canonical FK |
| `FK-B` (volume-threshold basis) | **not invalidated** | Basis and value of the volume trigger are independent of what a flush writes. Still genuine, still to be authored by decide |
| Topic 3 (QA infrastructure setup) | **not invalidated** | The harness config items (start command, base URL, framework, location, teardown, prerequisites, stub-vs-live upstream) and the interval-injectability requirement are all format-independent. Remains the next askable topic after the replan |
| `G-001` criterion / `VP-T002-write-count-reduced` | **re-derived again** | `N + F` was the documented-layout figure. The collapsed layout is `F`, so the reduction versus today's `2N` is `1 − F/(2N)` |
| `VP-T002-layout-preserved`, `VP-T002-diff-chain` | **invalidated** | Both assert the documented per-turn layout. They have no replacement yet |
| `RB-001` | **must be re-run** | It concluded "a new flush symbol in `dump.py`" against the documented layout. The collapsed format introduces a new read path as well as a new write path; the row's role/peer search must be repeated for it |
| Invariant dispositions | **override register now required** | `EI:.../output_layout` and `EI:.../sequential_diff_chain` are displaced, not upheld |
| `README.md:146-162` | **becomes a task** | The documented output contract and reading workflow no longer describe what the tool does |

## Answer-Impact Replanning Check (MANDATORY)

Run on every task the chosen scope touches, before any move toward final approval.

| Check | Result |
|---|---|
| Done-when criteria exist for the affected tasks | **FAIL** — `T002`'s contract ("write pending turn files in order, then `index.txt` once") describes the displaced layout; `T003` inherits it; there is no task for the new format or the README rewrite |
| Anchor Map rows bound with stance tokens for the chosen scope's referents | **FAIL** — the `bulk flush` row's payload contract and the `dump index` row are now wrong, and no row exists for the collapsed artifact or its read path |
| Every `Mode: NEW` row cites a complete `RB-*` row | **FAIL** — the collapsed format's writer/reader is a new referent with no `RB-*` row; `RB-001` was scoped to the documented layout |
| Validation point + milestone mapping exist | **FAIL** — `VP-T002-layout-preserved` and `VP-T002-diff-chain` are invalidated with no replacements; `MS-001`'s "complete when" still names them |
| Task fits the session sizing constraint | **UNCERTAIN** — format definition + read path + README rewrite + tests is additive scope never sized |
| Signal contract where behavior is signal-only | **N/A** — every affected behavior is observable in `DUMP_DIR/<sid>/` contents and the existing suite |

**Result: `REPLAN-INCREMENT: payload/format cluster`.** The chosen scope was a
placeholder alternative in the Options table and was never planned to agent-ready
depth. Final approval is BLOCKED while these checks fail.

This is the *middle path*, not a full reroute: the trigger half (`FK-001`), `T001`,
the milestone skeleton, and every pre-existing Anchor Map binding stay untouched.
The replan is scoped to the payload format, its Anchor Map rows and `RB-*` row, its
validation points, and the README/override-register consequences.

**Ordering constraint for the dispatcher**: this replan must land **before**
decide authors `FK-A`. `FK-A`'s framing is the documented-layout figure and is
wrong until the replan supplies the collapsed-layout arithmetic.

## Remaining queued topics (not asked this turn)

- **`FK-A` / `FK-B`** — not askable. Neither exists as a canonical FK; decide must
  reserve and fill them, and `FK-A`'s premise must be repaired first (above).
- **Topic 3 — QA infrastructure setup** — the only remaining askable topic, and it
  is explicitly not an FK (`## Topic Queue` → "Topic 3 … not an FK"). The one-FK-
  per-turn contract therefore has no FK to carry this turn, and the exhaust-before-
  rerouting sweep is complete with respect to FKs: there are no unasked canonical
  FKs. Topic 3 is carried to the next turn.
- **The collapsed format's definition** — a materially new, user-visible question
  created by this answer (does `index.txt` survive; are per-turn diffs still
  readable and who reconstructs them; append-only or rewritten per flush; what
  replaces the documented reading workflow). Interview does not allocate an ID for
  it; decide owns the option analysis and the successor FK. Recorded in
  `decisions.md` §FK-002 → "Successor question this answer creates".

## Gates Found Unmet (not cured by this agent)

1. `.agents/skills/qa-project/SKILL.md` does not exist, and this plan declares two
   QA author passes (`QAP-MS001`, `QAP-MS002`). The QA Capability Gate and QA Plan
   Creation both block on it. Reported, not written — the skill file requires
   user-confirmed content.
2. Final approval is blocked pending `FK-A`/`FK-B` authoring + resolution,
   PLAN-SELECTED, and the payload/format replan above. Reported, not cured.
