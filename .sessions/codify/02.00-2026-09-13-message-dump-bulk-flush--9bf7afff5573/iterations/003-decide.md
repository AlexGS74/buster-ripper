# 003 — Decide (OPTIONS mode)

## Orchestrator Prompt (verbatim)

> Execute the decide-interactive skill for this session. Arguments: $ARGUMENTS
>
> [The remainder of the prompt was the `decide-interactive` skill body plus the
> `codify-agents`, `critical-thinker`, `codify-user-prompt`, and
> `codify-deterministic-closure` skill bodies. `$ARGUMENTS` was empty. No `Mode:`
> line was present, so this is OPTIONS mode. Session ID was supplied out-of-band
> as `02.00-2026-09-13-message-dump-bulk-flush--9bf7afff5573`; Question Mode is
> `deferred` per session metadata.]

## Session Type / Mode

- New parent session (`02.00`), no `parent_session_uid` → the parent
  coordination contract does not apply. No parent path to read or reconcile.
- **Mode: OPTIONS.** No `Mode:` line in the prompt. Output is the option set,
  estimate-level costs/risks, one-line invariant themes, and skeletons only.
  Full done-when task-out, VP/milestone matrices, and Anchor Map finalization are
  deferred to PLAN-SELECTED.

## Context Conflict Detected

None. Orchestrator prompt carried no technical content to conflict with session
files. `decisions.md`, `plan-summary.md` (organize revision) and
`handoff-context.md` agree with each other and with source.

## Changes Since Iteration 002

- READ: `dump.py`, `server.py`, `routes/messages.py`, `config.py`,
  `pyproject.toml`, `README.md:135-175` — all four Anchor Map claims this phase
  depends on re-verified at their bound symbols (see "Validation Performed").
- NEW: option space authored (3 options) with skeletons; no full task-out.
- NEW: `FK-001` (flush trigger policy) and `FK-002` ("in bulk" semantics /
  write-reduction ceiling) reserved, filled, and set `open`.
- RESOLVED-FROM-CODE (not a user question): turn-0 buffering → Forward discretion
  footnote (below).
- MERGED: organize's deferred questions 1 and 4 are one causal record, not two —
  "in flight messages magnitude" *is* the flush-trigger trade-off. One FK
  (`FK-001`), not two.

## What the User Actually Faces

Two policy dials, not one. Everything else in this change is code-determined.

| Dial | Positions | Carried by |
|---|---|---|
| **D1 — When does the buffer hit disk?** | (a) on a pending-turn threshold **and** at shutdown; (b) at shutdown only | `FK-001` |
| **D2 — What does a flush write?** | (a) the documented per-turn layout; (b) one collapsed per-session file | `FK-002` |

The two dials are independent, which is why the option list below is three
combinations rather than a single either/or.

### Framing surprises organize surfaced that the user has not yet seen

1. **The write-reduction ceiling under the documented layout is ~50%, not
   ~100%.** Per intercepted request today: `turn_NNN` file write
   (`dump.py:82`, `:95`) + `index.txt` append (`dump.py:104-106`) = 2N for N
   turns. Buffering can only collapse the index appends (one open+write per
   flush instead of per turn). Turn files are one-file-per-turn and cannot be
   merged without changing the layout. So N turns → **N + 1** writes. Only the
   layout-breaking collapse reaches ~100%. `FK-002`.
2. **Shutdown-only flushing loses the *whole session* on a hard kill, not an
   in-flight window**, and leaves `index.txt` nonexistent for the session's
   entire life — the documented reading interface (`README.md:146-162`) does not
   work while the proxy runs. `FK-001`.
3. **Buffering is not only a write-count change — it removes blocking disk IO
   from the event loop.** `dump_turn` is synchronous and is called inline from
   `handle_messages` (`routes/messages.py:55-59`), i.e. two blocking writes per
   intercepted request on the request path. Any of the three options fixes this;
   the write *count* is the smaller of the two wins.

## Options

### Option 1: Buffer per session, flush on a pending-turn threshold and at shutdown — documented layout preserved

- **What it is**: Intercepted requests append their rendered turn to an
  in-memory per-session buffer instead of touching disk. When the buffer reaches
  a small threshold (order of tens of turns) it is written out in one batch; a
  final batch runs inside the shutdown path that Ctrl-C / SIGTERM already reach.
  File names, numbering and the index format are exactly what they are today.
- **What it changes**: `SessionState` gains a pending-writes field; `dump_turn`
  appends rather than writes; a new flush routine in `dump.py` writes the
  pending turn files in order and then all index lines in a single open;
  `server.on_cleanup` calls the flush wrapped so it cannot raise.
- **What stays the same**: the dump directory layout and index line format, the
  diff chain semantics, `/v1/messages` responses and latency class, session
  keying, `--dump-dir` as the mode switch, the absence of any custom signal
  handler, the absence of new endpoints or config.
- **Pros**: bounded loss window (at most the unflushed tail); `index.txt` is
  readable during the session, so the documented reading workflow keeps working;
  ~50% fewer write operations (2N → N + ⌈N/K⌉); blocking IO leaves the request
  path for all but one request in K.
- **Cons**: more code than Option 2; strictly more writes than Option 2 (the
  extra index opens); introduces a threshold constant that needs a defensible
  default.
- **Complexity**: medium. **Risk**: low-medium — the two real hazards are a
  flush that raises inside `on_cleanup` (it would replace `GracefulExit` and the
  process would exit with a traceback) and a flush that reorders turns (it would
  corrupt every later diff).
- **Rejected because**: —

### Option 2: Buffer per session, flush at shutdown only — documented layout preserved

- **What it is**: Same buffering, but the buffer is written only when the process
  shuts down gracefully. Nothing is written while the proxy runs.
- **What it changes**: as Option 1, minus the threshold branch.
- **What stays the same**: everything Option 1 keeps.
- **Pros**: the smallest change that satisfies the literal request; the fewest
  total writes (N + 1 for N turns); zero disk IO on the request path for the
  whole session — the maximum relief for a latency-sensitive proxy.
- **Cons**: on SIGKILL / crash / OOM the entire session's dump is lost, which is
  a larger loss than the "in-flight messages" the user explicitly waived;
  `index.txt` does not exist until exit, so the documented `README.md:146-162`
  reading workflow does not work while the proxy is up; the dump is all-or-
  nothing, so a long session has no partial record.
- **Complexity**: low. **Risk**: medium — the whole-session loss window and the
  mid-session absence of the documented index are the two consequences a user
  most often discovers after the fact.
- **Rejected because**: —

### Option 3: Collapse the session dump into one append-only file, flush on threshold and at shutdown

- **What it is**: Instead of one file per turn, each flush appends the pending
  turn records to a single per-session file (e.g. one JSON object per turn).
  This is the only design that reaches a near-total write reduction.
- **What it changes**: the dump output format itself; `README.md:146-162` must be
  rewritten; the per-turn `turn_NNN.diff` artifacts stop existing on disk and
  diffs become a read-time concern; `index.txt` as a separate artifact
  disappears; `EI:.../output_layout` and `EI:.../sequential_diff_chain` are
  deliberately overridden and the plan must carry an override register for them.
- **What stays the same**: request path behavior, signal handling, session
  keying, the buffering/flush mechanics.
- **Pros**: ~100% fewer write operations (one write per flush, ⌈N/K⌉ total);
  one artifact per session, which is easier to grep and copy than a directory of
  diff files.
- **Cons**: breaks a documented output contract and the diff-per-turn reading
  workflow the tool exists for; the diff chain is no longer directly observable
  on disk; requires a README rewrite and an explicit invariant override; a
  second `RB-*` Reuse Evidence row would be needed because the flush artifact is
  no longer the one RB-001 assessed.
- **Complexity**: medium-high. **Risk**: high — this is the one option that
  changes an interface users already depend on, and the repo has no invariant
  registry to make the override checkable.
- **Rejected because**: —

### Which option the agent leans toward (analysis, not decision)

Option 1, and the lean is sourced: the user's own acceptance sentence scopes the
loss to "in flight messages" (`US-001`), and Option 2 turns that into
whole-session loss; and `README.md:146-162` documents `index.txt` as the reading
interface, which Option 2 makes nonexistent for the session's whole life. The
counter-argument is real and must be stated: the user also said "we don't care
about making this super robust", and a threshold flush is durability machinery
they did not ask for. That is why `FK-001` is a preference with no
recommendation attached rather than a ratification.

Option 3 is presented because the ~50% ceiling is the surprise in this change and
the user must see the only alternative that beats it — not because it is
recommended.

## Skeletons Per Option (no full task-out in this mode)

Sizes are estimates; anchors are stance *deltas* only; the canonical map is
finalized in PLAN-SELECTED.

| Option | Tasks (titles + rough size) | Anchor stance deltas | Would-be `Mode: NEW` artifacts |
|---|---|---|---|
| 1 | T1 pending buffer on `SessionState` + append path (S) · T2 flush routine + threshold trigger (M) · T3 wire into `on_cleanup` wrapped (S) · T4 tests: buffer, flush, layout, diff chain, isolation, failure, signal (M-L) | `dump entry point` extend; `per-session dump state` extend; `dump index` extend; `shutdown flush seam` extend; all others unchanged | `bulk flush` (RB-001) |
| 2 | T1 pending buffer (S) · T2 flush routine, no threshold (S) · T3 wire into `on_cleanup` wrapped (S) · T4 tests minus the threshold cases (M) | same as Option 1 minus the threshold branch inside `dump_turn` | `bulk flush` (RB-001) |
| 3 | T1 pending record buffer (M) · T2 single-file append flush (M) · T3 wire into `on_cleanup` wrapped (S) · T4 tests for the new format (M-L) · T5 README dump-section rewrite (S) | `dump index` **replace/remove** (index.txt ceases to exist); `dump entry point` extend (larger); `per-session dump state` extend; `shutdown flush seam` extend | `bulk flush` (RB-001, **evidence must be re-run** — RB-001 assessed a per-turn-file writer, not a single-file appender) |

**Validation themes only** (VP/milestone matrices are PLAN-SELECTED work):
writes deferred until a flush; flush preserves layout and arrival order; diff
chain intact across a batched flush; write count ≤ N+1 for N turns; two sessions
flush independently; a raising flush neither fails a request nor changes the
process exit; a real SIGINT/SIGTERM puts buffered turns on disk and exits
cleanly. Option 3 replaces the first two themes with a single-file-format theme.

**Invariant themes only** (full conflict matrix is PLAN-SELECTED work): buffered
append preserves turn order (sequencing stays at append time, not flush time);
flush preserves the output layout; flush cannot raise out of `on_cleanup`; flush
is idempotent per pending batch. Option 3 adds an override register for
`EI:.../output_layout` and `EI:.../sequential_diff_chain`.

## Resolved Without a User Question

- **Turn-0 buffering (organize deferred Q3)** — *not a user decision.* No
  escalation-floor clause fits: neither branch is irreversible, neither is
  expensive, and the observable difference (whether `turn_000.json` survives a
  hard kill) lies entirely inside the durability envelope `US-001` already
  waived. **Forward discretion footnote — recommended reading: buffer turn 0
  exactly like every other turn.** Rationale: the buffer is per-turn pending
  state, so turn 0 needs no branch; special-casing it would break the uniform
  "writes deferred" property for one write per session of gain, and under
  Option 1 it is flushed with the first batch anyway.
- **Scope = the existing `--dump-dir` path on `/v1/messages` only** — verified
  from code by collect and re-verified by organize; no dump call exists in
  `routes/chat.py`, `count_tokens.py`, or `passthrough.py`.
- **No background flush timer** — a threshold check inside the append path is
  synchronous; reusing the `compaction.poll_kv_cache` task pattern would add a
  task to cancel in `on_cleanup` for no benefit.
- **Coverage gate (organize deferred Q5)** — a plan constraint, not a question.
  Headroom is ≤ 4 uncovered new statements; the new `dump.py` lines must ship
  with tests.
- **Confusability triage** — the one backlog finding (duplicate `main` entry
  points, `server.py:99` vs `main.py:62`) does **not** endanger this session's
  implementation: both entry points reach `create_app`/`on_cleanup`, and the
  flush must tolerate `DUMP_DIR is None` regardless. Stays in the quality
  backlog; no pre-implementation cleanup task.
- **`dump.py:63` bare `assert config.DUMP_DIR is not None`** — organize noted it
  as an adjacent latent defect (stripped under `python -O`) sitting inside the
  file this change rewrites. Not a user question and not a new requirement; the
  flush path must tolerate `DUMP_DIR is None` anyway. Recommend folding a real
  guard into the same increment rather than adding a task for it. PLAN-SELECTED
  will place it.

## Fork Authoring

Two Forks authored, both reserved via the allocator and filled in place:

| FK | Type | Question | Recommendation |
|---|---|---|---|
| `FK-001` | preference | Flush on a pending-turn threshold + shutdown, or shutdown only? | none |
| `FK-002` | ratification | Preserve the documented per-turn layout (accepting the ~50% ceiling) or collapse to one file? | preserve the documented layout |

`FK-001` is a preference: the user's own words ("we don't care about making this
super robust") and the documented reading workflow point in opposite directions,
and no source or execution settles which wins. `FK-002` is a ratification: the
code-and-docs-determined answer is to preserve the layout, but accepting a ~50%
ceiling against a stated goal of "in bulk" is the user's act to own.

Organize's deferred questions 1 and 4 were **merged into `FK-001`** — they are
one causal record ("in flight messages magnitude" is the reason the trigger
choice matters), and two FKs would ask the same question twice.

## Assumptions Carried

- The three `US-001` factual claims remain `assumptions PENDING`; organize's
  source reads corroborate all three but the verdicts were not flipped.
  `assumptions: claim-{n}` settlement is the assumptions verifier's, not this
  phase's.
- The ~50% ceiling assumes "write operation" counts an open+write+close sequence
  per artifact, which is how organize counted it and how the instrumented
  validation point will count it. The ratio (2N → N+1) is invariant to that
  choice; only the absolute numbers move.
- The aiohttp SIGINT/SIGTERM → `on_cleanup` chain is confirmed by source
  (`web_runner.py:283-284` → `web.py:511` → `web.py:425` → `server.py:60`), not
  assumed. No custom signal handler is needed and none should be added.

## Compaction Decision

- Prior same-phase files reviewed: none — there is no prior `*-decide.md` in this
  planning cycle. `001-collect.md` (collect) and `002-organize.md` (organize) are
  different phases/slices.
- Superseded: none
- Not superseded: `001-collect.md`, `002-organize.md`
- Compaction performed: no
- If no: there is no prior same-phase file to compact, and no prior FK to
  self-contain. Nothing was deleted.

## Dropped-Clause Sweep

Not triggered — this pass supersedes no prior definition. It is the first Decide
of the cycle and authors new option analysis rather than replacing an existing
one.

## Validation Performed

- Read `buster_ripper/dump.py:1-109` — confirmed the three write sites
  (`:82` `write_text` turn 0, `:95` `write_text` turn N, `:104-106`
  `index.open("a")`), the module-global `_sessions` (`:24`), `SessionState`
  (`:16-20`), `prev_text` advancing at `:108` and `turn` at `:109`, and the bare
  `assert` at `:63`.
- Read `buster_ripper/server.py:1-113` — `on_cleanup` (`:60-72`) cancels the KV
  task and closes the shared session; registered at `:79`; `server.py:main`
  (`:99-102`) does not configure `DUMP_DIR`.
- Read `buster_ripper/routes/messages.py:55-59` — the only `dump_turn` call
  site, synchronous, inline on the request path, wrapped in
  `try/except Exception`.
- Read `buster_ripper/config.py` — `DUMP_DIR: Optional[Path] = None`.
- Read `pyproject.toml` — `--cov-fail-under=80`; `routes/*`, `main.py`,
  `server.py` omitted from coverage, `dump.py` not omitted.
- Read `README.md:135-175` — the documented output layout and the "Reading the
  index" workflow that Option 2 would leave absent for the session's whole life.
- `git status` / `git log` on branch `refactor/package-structure` — working tree
  has only unrelated `.sessions/` changes from session `01.02`; nothing of this
  session's is at risk.

## Questions Asked

No questions - deferred mode (`Question Mode: deferred` in session metadata; no
active-mode phrase in the orchestrator prompt). Decide does not ask questions in
any case; the two decision points are authored as `FK-001` and `FK-002` for
invite to present.

## Next Step

Ready for **invite**: present the three options and the two dials, then ask
`FK-001` first. `FK-002` is a ratification and follows. No plan text was
finalized in this mode; PLAN-SELECTED runs after the user picks.

`DECIDE-OPTIONS: 3 options ready`
