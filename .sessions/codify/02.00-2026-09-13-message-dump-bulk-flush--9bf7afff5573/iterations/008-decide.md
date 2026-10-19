# 008 — Decide (OPTIONS mode)

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

- Continuing session `02.00-2026-09-13-message-dump-bulk-flush--9bf7afff5573`.
  Identity verified against `session-identity.json` (UID
  `9bf7afff-5573-402a-b102-827423a1dded`; handle suffix `9bf7afff5573` = first 12
  hex of the UID; display id `02.00-2026-09-13-message-dump-bulk-flush` matches
  the handle prefix). **No `parent_session_uid`** → the parent coordination
  contract does not apply; there is no parent path to resolve or reconcile.
- **Mode: OPTIONS.** No `Mode:` line in the prompt, same as iteration 003. Output
  is the option set, estimate-level costs/risks, one-line invariant dispositions,
  and skeletons only. Full done-when task-out, VP/milestone matrices, and Anchor
  Map stance finalization are deferred to PLAN-SELECTED.
- **Question Mode: deferred.** Decide does not ask. Options are authored for
  invite/interview to present.

## Context Conflict Detected

None between sources. One **factual error in the plan surface** was found and is
recorded under §Findings — it is a correction, not a conflict with a higher-priority
source.

## Changes Since Iteration 007

- NEW: `FK-003` reserved via the allocator, filled, set `open` — the successor FK
  for the collapsed artifact's record content (`G-006`). Three options authored.
- NEW: `FK-A` (flush interval value) and `FK-B` (volume-threshold basis) re-tested
  against the escalation floor. **Neither is askable.** Both become Forward
  discretion footnotes with a recommended reading.
- NEW: the three `EI:` entries in the contradiction surface given one-line
  dispositions (the `OI:` register itself lands in PLAN-SELECTED).
- NEW: Finding — `plan-summary.md` §Anchor Map's "flush trigger parameters" row
  cites a justification that is false against source.
- NEW: Finding — the same row is named `DUMP_FLUSH_TURNS`, inconsistent with the
  basis `FK-001` already recorded.
- UPDATED: `plan-summary.md` §Current Phase, §Open Questions 2b, §Approved Override
  Register (disposition column), Anchor Map Notes (factual correction).
- READ (this pass): `buster_ripper/dump.py` (whole file), `buster_ripper/config.py`
  (globals), `buster_ripper/main.py` (CLI flag census), `README.md:120-175`,
  `tests/conftest.py`, `tests/test_routes_integration.py:1-60`.
- UNCHANGED: the settled `FK-001` / `FK-002` decisions, `T001`, `T003`'s trigger
  half, every pre-existing Anchor Map binding's referent, the milestone skeleton,
  and every element of the flush-policy cluster (iteration 005).

## What the user actually faces

The plan is at agent-ready depth. Both policy dials are settled (`FK-001` triggers,
`FK-002` collapsed payload). **Exactly one genuinely open user question remains**:
what a flush's record *contains* — the half of the documented output contract that
`FK-002` displaced without replacing. Everything else that looked like a decision is
code-grounded and is dispositioned below as a Forward discretion footnote.

| Item | Status after this pass |
|---|---|
| **The artifact's record content** | **`FK-003` — open, authored, awaiting the user** |
| Flush interval value (`FK-A`) | **not askable** → Forward discretion footnote |
| Volume-threshold basis (`FK-B`) | **not askable** → Forward discretion footnote |
| Compression | **not askable** → not adopted (Forward discretion footnote) |
| `index.txt` survival, append-only, no-tool-required | settled from the record (iteration 006) — not re-asked |
| Turn-0 full body in the artifact | settled — required by `VP-T002-turn0-available`; the `EI:` disposition follows |
| QA infrastructure setup (Topic 3) | carried — a presentation **topic**, not an FK; code-natural answers recorded below |

## Options Analysis

`FK-002` fixed the requirement (one bulk flush is ONE write) and displaced the
documented per-turn layout, but left undetermined what a record contains. `G-006`
("the dump stays readable — function, not layout") is OPEN with four alternate
readings and **no repository evidence excludes any of them**. Three of the four
candidate questions the scope-first sweep produced were settled from the record
(`index.txt` cannot survive as a separate file; the artifact is append-only; no new
tool is required *if* the artifact embeds the change information). One remains.

### Option 1 — Per-turn diffs inside the collapsed artifact (change-focused; closest to today)

- **What it is**: the session's artifact is one append-only file. Each flush appends,
  for every pending turn, that turn's change information in the same form the tool
  produces today — a one-line summary, turn 0's full JSON request body, and for every
  later turn the unified diff of the rendered request against the previous turn.
- **What it changes**: three files (`turn_000.json`, `turn_NNN.diff`, `index.txt`)
  become one; the *information content is identical to today*.
- **What stays the same**: everything a reader can see today — the summary line per
  turn, the full baseline, and the per-turn diffs; the `grep`/`less` reading habit;
  the README's cache-buster workflow, with its commands retargeted at one file.
- **Pros**: fewest bytes written of the three (it is the payload the tool already
  writes, minus the per-file metadata); preserves today's change-focused reading
  experience exactly; needs no viewer; the `+N/-M` summary counts stay computable at
  append time for free (`dump.py:96-97` already does it); directly serves the
  documented purpose (`README.md:134-135`).
- **Cons**: a turn's full text is not directly present — recovering it means
  replaying the diff chain from turn 0; the artifact is a text log, not a structured
  record, so a scripted consumer must parse diffs.
- **Complexity**: low.
- **Risk**: low — the only cost is that "show me turn K in full" becomes a replay
  rather than a file open, and turn 0's full body is still present.
- **Anchor stance delta**: none. No new `Mode: NEW` row. `T002`'s record builder
  produces summary + diff; `T005` documents one file. `EI:.../turn_zero_is_full_body`
  refines cleanly.

### Option 2 — Per-turn full rendered bodies inside the collapsed artifact (self-contained)

- **What it is**: the same single append-only artifact, but every turn's record
  carries the *complete* rendered request (`_render_body` output, `dump.py:77`) rather
  than a diff. Turn 0 remains the full JSON body.
- **What it changes**: no diffs are stored at all. Every record stands alone.
- **What stays the same**: one write per flush, one file per session, append-only,
  arrival order, one summary line per turn.
- **Pros**: any turn's full content is directly readable — no replay, no diff chain to
  trust; each record is robust to damage in an earlier one; a `grep` for a cache-buster
  string hits every turn it appears in.
- **Cons**: bytes written per session grow with (turns × rendered size) instead of with
  the payload — the direct cost the user's stated SSD rationale targets; the *change*
  between turns is no longer visible directly (the reader must compare two records); the
  `+N/-M` summary counts become a computed comparison rather than a by-product.
- **Complexity**: low.
- **Risk**: medium — the risk is the one the user's own rationale cares about. The
  research (iteration 007, F1/F2) is explicit that **bytes written, not write-call
  count, is the wear quantity**, and this is the only option that raises them
  materially. It is also the option that best serves `G-006` reading (b).
- **Anchor stance delta**: none. No new `Mode: NEW` row.

### Option 3 — Machine record (JSON Lines) plus a viewer/reconstruction tool

- **What it is**: one JSON object per turn, appended one per line
  (`{"turn":N,"ts":…,"msgs":…,"tools":…,"bytes":…,"payload":…}`), read with `jq`.
- **What it changes**: the artifact stops being a human-readable log; the documented
  `cat`/`grep`/`less` workflow no longer works, so a small viewer/reconstruction
  command must be built and `README.md:146-162` documents a tool instead of a format.
- **What stays the same**: one write per flush, one file per session, append-only,
  arrival order.
- **Pros**: the most compact and the most scriptable of the three; trivial to aggregate
  across sessions; the record shape can change later without breaking a text format.
- **Cons**: requires a viewer the plan has never scoped and no repository evidence
  authorizes (`RB-003`'s read-path search found no code reader at all); the tool's
  documented purpose moves behind a tool; smallest records are the least FTL-friendly
  shape per the research (iteration 007, F2); it is the only option that adds a new
  component to the codebase.
- **Complexity**: medium.
- **Risk**: medium-high — the risk is the tool's stated purpose. `README.md:134-135`
  says the dump exists to identify remaining cache busters; this option moves that from
  "read the artifact" to "run a tool", which is a `Mode: NEW` addition the plan has
  neither scoped nor budgeted, and which `RB-003` explicitly declined to authorize.
- **Anchor stance delta**: **adds** a `Mode: NEW` row (viewer/reconstruction
  command) that requires its own complete `RB-*` row, and turns `T005` into tool
  documentation.

### Rejected — "write a blob" (no readability commitment)

Considered and rejected, recorded so no reader re-derives it. The cheapest way to reach
one write per flush is to write the least: append turn 0's body and nothing else, or a
single opaque payload. That satisfies `G-001` completely and destroys the tool's
documented purpose while every implementation task completes — it is precisely `G-006`'s
failure witness ("the artifact is written once per flush and is well-formed, but a reader
cannot find the changed turn or see what changed"). It is not a product choice; it is the
failure mode. Not presented.

### Analytical note (not a recommendation, and deliberately not in `FK-003`)

The trade-off is a genuine three-way product choice and `FK-003` carries **no
recommendation** — for a `preference` a presented recommendation would be
pre-selection. For the user's own benefit when they weigh it:

- The documented purpose (`README.md:134-135`, "identifying remaining cache busters")
  and the user's own stated rationale for `FK-002` ("saving SSD") both point the same
  way — Option 1 is the change-focus the reading workflow already uses and the
  lowest-byte of the three.
- Option 2 buys self-containment (no replay) at a byte cost the research measures on
  the wear axis the user cares about.
- Option 3 is the only one that spends new scope (a viewer) the plan has not
  authorized, and it moves the documented purpose behind that tool.

That is analysis of the trade-off, not a decision, and the user may weigh the axes
differently — self-containment is a real property Option 1 lacks.

## Framing surprises the user has not yet seen

Surfaced for alignment alongside the pick, per OPTIONS-mode duty.

1. **A database is not a wear mitigation** — the user's own candidate in `US-002`
   ("maybe … using a database") is **refuted for this workload** by external sources
   (iteration 007, F5): SQLite's defaults cost 3–4 syncs per commit, WAL writes each
   changed byte ~twice, and documented real-world incidents reached ~37 TB in 21 days
   (Codex) and a projected 600 TBW/year (NordVPN), with cumulative write volume
   growing quadratically with DB size. No plan element changes.
2. **"This kind of payload can be really harmful" is not supported at estimated
   volumes** (~6.4 MB/session; ~23 GB/year at 10 sessions/day against 300–600 TBW
   ratings). The *design pattern*, not the payload size, is what determines harm —
   and the settled design is already the lowest-write pattern of the candidates.
3. **The interval no longer trades against the write-reduction goal.** The old framing
   ("a short interval drives `F → N` and the reduction toward zero") is **false** under
   the collapsed payload: at `F = N` the reduction is still **50%**, and zero is
   unreachable. The interval now sets only the staleness bound and the append frequency.
4. **The way to lose `G-001` moved.** It is no longer "too many flushes"; it is a
   **rewrite-per-flush** artifact — write count `F`, bytes written `O(N²)`. That is why
   append-only is load-bearing rather than stylistic, and why the byte axis must be kept
   as a failure witness.
5. **`index.txt` does not survive.** A per-flush `index.txt` would be a second write per
   flush (contradicting `FK-002`); a shutdown-only one would not exist during the
   session (contradicting `FK-001`'s recorded basis for the time trigger). Settled from
   the record, not re-asked. The summary *line* survives inside the artifact.
6. **Turn 0's full body must be in the artifact.** A missing full baseline makes the
   whole session unreadable (`EI:.../turn_zero_is_full_body`). This constrains all three
   options and is proven by `VP-T002-turn0-available`.
7. **The documented reading workflow changes** (`README.md:146-162`, task `T005`). How it
   changes depends on the option: a one-file `grep`/`less` habit (1), a compare-two-records
   habit (2), or a tool invocation (3).
8. **`buster_ripper_legacy.py:41-45` documents the same displaced layout** in its module
   docstring. It is the tracked pre-refactor monolith referenced by nothing under
   `buster_ripper/` and is **excluded from scope** — recorded so no reader binds the old
   layout to live code.

## `FK-A` — escalation-floor re-test (decide's call: DO NOT ASK)

`FK-A` is the flush **interval value**. Re-tested independently against the three
escalation-floor clauses (`interview-interactive` → "Escalation floor"):

| Clause | Verdict |
|---|---|
| contradicts a recorded user decision or a protected invariant | **no** — any interval is consistent with `FK-001`'s "both triggers + shutdown" |
| irreversible or expensive to reverse (data loss, public API surface, security) | **no** — one `config` global plus a CLI flag; a default, not a surface |
| plan surface gives no basis at all **and** a wrong guess is expensive | **fails the second conjunct** — the plan surface supplies a direct in-repo peer (`config.py:44 KV_POLL_INTERVAL: int = 10  # seconds between /metrics polls`), and the repaired arithmetic bounds the cost of any wrong guess: every achievable `F` leaves ≥ 50% of the reduction, and a short interval costs only extra small appends |

**Decision: do not author `FK-A` as a user question.** Recorded as a **Forward
discretion footnote** (below). `FK-A` is **not** reserved and **not** filled — an
abandoned reservation would be blocking.

## `FK-B` — escalation-floor re-test (decide's call: DO NOT ASK)

`FK-001`'s `Resolution` already records the normative choice as a "**data-volume**
threshold", and the user's own words are "data volume-based". Turns-vs-bytes is
therefore settled against `FK-001`'s presented option text. The residual was "bytes of
what".

| Clause | Verdict |
|---|---|
| contradicts a recorded user decision or a protected invariant | **no** |
| irreversible or expensive to reverse (data loss, public API surface, security) | **no** — one comparison in one branch, plus a config default |
| plan surface gives no basis at all **and** a wrong guess is expensive | **fails the first conjunct** — `dump.py:77` already computes `current_text = _render_body(body)` and `dump.py:83` already measures `len(current_text) / 1024` for the summary line; the rendered text is the code-natural measure and a wrong guess is one cheap edit |

**Decision: do not author `FK-B`.** Forward discretion footnote (below).

## Forward Discretion Footnotes (decide-authored, binding on Forward)

| Handle | Recommended reading | Basis |
|---|---|---|
| `FK-A` — flush interval value | Default the interval to the in-repo peer's value — **10 s** — as `config.DUMP_FLUSH_INTERVAL: int = 10`, and expose it as a CLI flag so the QA harness can inject it at process start. The flag's *existence* is a settled QA requirement (`QAP-MS001`/`QAP-MS002` need a deterministic `F`); only its *value* is discretion | `config.py:44 KV_POLL_INTERVAL: int = 10`; iteration 006 §`FK-A` escalation-floor re-test |
| `FK-B` — volume-threshold basis | Measure pending volume as **bytes of rendered text** — `len(_render_body(body))`, the same quantity `dump.py:83` already computes for the summary line and the same text the diff is computed from (`dump.py:87-89`). Raw JSON body bytes are **not** the basis (they are only materialized for turn 0, `dump.py:82`) | `dump.py:77,83,87-89`; `FK-001` `Resolution` ("data-volume threshold") |
| `FK-B` — volume-threshold default | Default the threshold to a **single-digit MiB of rendered-text bytes** (recommended reading: **1 MiB**, ≈ 20 turns at the documented example sizes: turn 0 ≈ 412 KB + ≈ 30 KB/turn) so the buffer bounds both memory and the unflushed tail; expose as a CLI flag for QA injection. PLAN-SELECTED records the concrete value | `README.md:146-162` (documented example sizes); `G-005` boundary ("no memory cap is promised unless the user picks a byte basis" — a byte basis is now the recorded one) |
| Compression | **Not adopted.** No goal requires it; `RB-003` records "no compression facility is warranted"; it is the one remaining byte-reduction lever but it changes the read path/format, which is a user-visible contract change no goal asks for. It stays a roadmap candidate, not a plan element | iteration 007 §Optional mitigations; `RB-003` |
| Turn-0 buffering | Buffer turn 0 uniformly with every other turn (unchanged from iteration 003/004). Its *presence in the artifact* is the separate, load-bearing clause handled by `EI:.../turn_zero_is_full_body` | `EI:.../turn_zero_is_full_body` |
| `dump.py:63` bare `assert` | Fold a real guard into the same increment (`config.DUMP_DIR is None` → no-op return), matching the flush's "no-op when `DUMP_DIR is None`" contract | `dump.py:63`; plan Anchor Map `bulk flush` row |
| Artifact name / extension | Forward discretion, presented in `FK-003`'s option text rather than spent as a fork: user-visible (the README names it) but cheap, reversible, and mechanical given the content choice | iteration 006 §Successor Question Scope |

## Planned Invariants In Scope (one-line dispositions only — the `OI:` register lands in PLAN-SELECTED)

Three `EI:` entries are in the contradiction surface. All three are **session-local
drafts** — no invariant registry exists in this repository (`.invariants/` absent,
re-confirmed).

### `EI:buster_ripper.dump/dump_turn/output_layout`

- **Disposition: `override`.**
- **Why**: direct displacement. The statement ("a session directory contains
  `turn_000.json`, `turn_{N:03d}.diff` for N ≥ 1, and `index.txt` with one summary
  line per turn") names three files that `FK-002` removes. The replacement statement
  is **not** written here — it needs the `FK-003` answer (whether the artifact
  carries diffs, bodies, or a machine record), and PLAN-SELECTED writes it into the
  `OI:` register.
- **Note**: `FK-002`'s `Resolution` already supplies the *direction*
  ("the documented per-turn layout is NOT preserved"); the direction is a recorded user
  decision. Only the replacement statement is pending.

### `EI:buster_ripper.dump/dump_turn/sequential_diff_chain`

- **Disposition: `refine`** (recommended over override).
- **Why**: the property survives. `T001` keeps `turn`/`prev_text` advancing at
  **append** time, so "turn N diffs turn N−1's rendered text, in arrival order"
  still holds. Only the clause "`prev_text` advances **only when a turn is written**"
  changes — the flush must never advance it, or the diff chain resets whenever the
  interval fires. Restated: *"`prev_text` advances at append time; each turn's record
  is derived from the previous turn's rendered text, in arrival order."*
- **Caveat for PLAN-SELECTED**: under `FK-003` option 2 (full bodies) there is no
  diff chain in the artifact at all; the invariant then governs only the
  `turn`/`prev_text` sequencing that the record builder consumes, and the
  disposition is still `refine` (the sequencing clause is what is load-bearing).

### `EI:buster_ripper.dump/dump_turn/turn_zero_is_full_body`

- **Disposition: `refine`** — and this is **not contingent on `FK-003`.**
- **Why**: the load-bearing clause is "a missing full baseline makes the whole
  session directory unreadable" (iteration 001). All three options preserve turn 0's
  full body — Option 1 embeds it as the baseline of the diff chain, Option 2 embeds
  it as an ordinary full-body record, Option 3 embeds it in the record's payload. The
  *mechanism* ("written as a separate `turn_000.json` file") is displaced by
  buffering regardless. Restated: *"turn 0's full body is present in the session
  artifact once flushed, never only a diff."* `VP-T002-turn0-available` proves it.
- **This resolves the iteration-006 Finding.** The plan named two overrides; three
  invariants are in the surface. `turn_zero_is_full_body` now has an explicit
  disposition, and the reason it does not depend on the successor FK is that the plan
  already requires the full body (`VP-T002-turn0-available`) and no option removes it.

### Unchanged dispositions (verified this pass)

- `EI:buster_ripper.routes.messages/handle_messages/dump_failure_isolated` —
  `preserve`. The record-building path moves more work inside the swallow-all
  contract at `routes/messages.py:55-59`, but the contract itself is unchanged and
  must be preserved. Proved by `VP-T001-dump-failure-isolated`.
- `EI:buster_ripper.session/session_id/derived_after_normalize` — `preserve`.
  Untouched by the payload change; buffering keys on it.
- `EI:buster_ripper.server/on_cleanup/graceful_shutdown_hook` — `preserve`
  (extended, not replaced). `on_cleanup` gains an ordering clause (cancel timer →
  final flush → close session), which is an extension of the same invariant, not a
  new one. Proved by `VP-T003-shutdown-flush`.

### Suspected contradiction pairs (re-verified)

- `EI:.../output_layout` ↔ *one-flush-one-write theme* — direct displacement.
  **Override.**
- `EI:.../turn_zero_is_full_body` ↔ *collapsed artifact theme* — **no longer a
  contradiction**: the artifact must embed turn 0's body. **Refine.**
- `EI:.../sequential_diff_chain` ↔ *buffered append theme* — a flush must not
  advance `turn`/`prev_text`. **Refine.**
- `G-006` ↔ `G-001` — a goal-level tension, not an invariant one: the cheapest
  way to one write per flush is to write the least; the cheapest way to stay readable
  is to write the most. `FK-003` arbitrates.

## Findings (repairs for PLAN-SELECTED; not user questions)

### F-1 — The plan's justification for the flush-parameter globals is false against source

`plan-summary.md` §Anchor Map, row *flush trigger parameters*, Notes reads: *"`config.py`
is the established home for every tunable (`KV_POLL_INTERVAL` at `:44` is the direct
peer) and **every global there has a CLI flag in `main.py`**"*.

The second clause is **false**. `config.KV_POLL_INTERVAL` (`config.py:44`) has **no CLI
flag** — `grep -n "poll" buster_ripper/main.py` returns nothing, and
`KV_POLL_INTERVAL` is referenced only at `config.py:44`, `compaction.py:29,57`,
`server.py:19,29`. The peer supplies the *module-global shape* and the *value*, not the
flag precedent.

This does not weaken the row's conclusion — the flag is separately required, because the
QA gate needs the interval injectable at process start (iteration 004 §QA Capability
Implication, third required capability). It removes a supporting claim that would not
survive a reader's check. **PLAN-SELECTED must drop the clause**; the correct
justification is "`config.py` is the established home for tunables; the peer
`KV_POLL_INTERVAL` supplies the value and the module-global shape; the CLI flag is
required by the QA injectability capability, and is a new precedent rather than an
existing convention."

### F-2 — The trigger-parameter global is named for a basis the record no longer uses

`plan-summary.md` §Anchor Map row *flush trigger parameters* names the globals
`config.DUMP_FLUSH_INTERVAL, DUMP_FLUSH_TURNS`. `FK-001`'s `Resolution` records the
normative choice as a "**data-volume** threshold" and the user's words are "data
volume-based"; the volume basis is bytes (see §`FK-B` above). `DUMP_FLUSH_TURNS` is a
leftover from the *superseded* option text ("pending-turn threshold") that the user's
answer replaced. **PLAN-SELECTED must rename it** (recommended `DUMP_FLUSH_BYTES`) and
re-derive its default, or record explicitly that a turns basis was chosen against the
recorded words.

### F-3 — Declared VP redundancy needs a resolution in PLAN-SELECTED

`plan-summary.md` declares `VP-T002-single-write-per-flush` redundant with
`VP-T002-write-count-reduced` + `VP-T001-writes-deferred`, and asks decide to keep
whichever is tighter rather than both by default. **Recommended for PLAN-SELECTED**: keep
`VP-T002-write-count-reduced` as the contract (it is `G-001`'s observable and it carries
the injected `F`), and fold the sharp per-flush form ("a flush carrying K ≥ 2 pending turns
performs exactly one write-bearing file operation") into it as an assertion. One VP, two
assertions.

### F-4 — The byte axis must survive as a failure witness

Iteration 007's deferred question ("should the write-count VP be paired with a byte-volume
check?") is **decide's call, not a user question**. Recommended for PLAN-SELECTED: yes —
keep `G-001`'s witness B (rewrite-per-flush → bytes `O(N²)`) as a live failure witness,
because the count alone cannot catch the regression the research identifies as the real one
(iteration 007, F1/F2). The count is the proxy; bytes are the physical quantity. A
count-only criterion would report green on a rewrite-per-flush artifact.

### F-5 — Two coverage-visible clauses to carry unchanged

`dump.py:63`'s bare `assert config.DUMP_DIR is not None` sits inside the rewritten file
(see §Forward Discretion Footnotes). The coverage headroom constraint
(`U ≤ 3.2 + 0.2·K`, `pyproject.toml:19,26-30`) is unchanged and loosens under the
payload rewrite. Neither is a user question.

## QA Infrastructure Topic (carried, not an FK)

Iteration 004 recorded Topic 3 (QA infrastructure setup) as the only remaining askable
topic and explicitly **not an FK**; iteration 006 confirms it is unchanged by the payload
change. It remains a presentation topic for invite/interview, and the six proposed items are
derived from source. Decide's analysis of the one genuinely open sub-question:

- **Stub vs live upstream.** The code-natural answer is **stub**: the existing suite mocks
  the upstream (`tests/test_routes_integration.py:32-60`, `_create_mock_response` +
  `mocker.patch.object(backend, …)`), `main.py:13` defaults `--upstream` to
  `http://localhost:30000`, and the QA gate must be able to run unattended for
  `QAP-MS001`/`QAP-MS002`. A wrong guess is cheap (point the harness at the other URL).
  It therefore **fails the escalation floor** and is not authored as a fork; it is recorded
  here so invite/interview can present it as a topic with a recommended reading.
- **`tests/qa/` location** is safe: `tests/conftest.py` is empty ("no shared fixtures"),
  so a separate QA tree conflicts with nothing.
- **Interval injectability** (the third required QA capability) is carried by the
  `FK-A` discretion footnote's CLI flag.

## Fork Authoring Record

- `FK-003` — reserved via `codify codify reserve-decision-record --kind fork --owner decide`,
  filled in place, `State: open`. `Type: preference`; `Why-yours` quotes the escalation-floor
  clause *"it is irreversible or expensive to reverse (data loss, public API surface,
  security)"* verbatim; `Presented.Recommendation: none` (a preference must not carry a
  recommendation). One-line index link added to `handoff-context.md` §Open Forks.
- **Not authored**: `FK-A`, `FK-B`, and the QA-topic sub-question — each fails the
  escalation floor and is dispositioned above as a Forward discretion footnote or a
  presentation topic. **No reservation was left abandoned**: the only allocator call this pass
  returned `FK-003`, which was filled.
- **Legacy migration**: not applicable — `decisions.md` exists and is canonical; there are no
  legacy Fork/User Choice records in `handoff-context.md` (its Fork sections are indexes only).

## Dropped-Clause Sweep

**Not triggered.** This pass does not supersede a prior definition of the same scope: it
authors a new FK, dispositions two FKs it declines to author, and dispositions three `EI:`
entries. It does not rebuild a task's done-when, rewrite a contract, or replace a prior
report (no compaction — see below). Iteration 006 already ran the sweep for the payload
cluster, and its dispositions are carried unchanged.

## Compaction Decision

- Prior same-phase files reviewed: `iterations/003-decide.md`.
- **Superseded**: none.
- **Not superseded**: `iterations/003-decide.md`.
- **Compaction performed: no.**
- **If no**: different slice. Iteration 003 authored the `FK-001` (trigger) and `FK-002`
  (payload) option space; both are now resolved and canonical in `decisions.md`, and this
  pass changes neither. This pass authors the *successor* FK those resolutions created
  (`FK-003`) plus the `FK-A`/`FK-B` dispositions, which 003 explicitly deferred to the
  replan. 003 also remains the provenance for the two-dial landscape and is review
  evidence; nothing it holds is unresolved in a way that would make it unsafe to keep.

## Deferred Questions

- **Which device/filesystem hosts `DUMP_DIR` in production** (Linux ext4/XFS beside vLLM,
  or macOS APFS on the dev box)?
  - Status: DEFERRED for /codify-review
  - Impact: the exact WAF differs (ext4 measured 4× on small overwrite; APFS is
    copy-on-write); the directional verdict does not. The dev machine's volume is 88% full,
    which elevates WAF at machine level regardless of the design.
  - Agent analysis: `none` — a factual premise for any numeric claim, not a user
    preference; source-checkable by inspecting the deployment host.
- **The concrete default value of the volume threshold.**
  - Status: DEFERRED to PLAN-SELECTED (decide owns the VP/default set)
  - Impact: sets how much may buffer before a flush.
  - Agent analysis: recommended reading recorded in §Forward Discretion Footnotes
    (single-digit MiB of rendered-text bytes).
- **Compression** — see §Forward Discretion Footnotes; not adopted, roadmap candidate.

## Questions Asked

No questions asked — deferred mode, and no question was required to evaluate the
options. `FK-003` is authored for invite/interview to present; `FK-A`/`FK-B` were
dispositioned without asking, per the escalation floor.

## Validation Performed

| Claim | How verified | Result |
|---|---|---|
| Session identity / handle suffix | `session-identity.json` vs handle | matches (`9bf7afff5573` = first 12 hex of UID) |
| No parent coordination contract | `session-identity.json` has no `parent_session_uid` | confirmed — contract does not apply |
| `_render_body` is what per-turn diffs and the size summary are computed from | read `dump.py:77-102` | confirmed (`current_text` at `:77`; `size_kb` at `:83`; diff at `:87-93`) |
| Today's per-turn write count is 2 (`turn_NNN` write + `index.txt` append) | read `dump.py:81-106` | confirmed (3 write-bearing sites) |
| Turn 0 is the full JSON body, later turns are unified diffs | read `dump.py:80-102`; `README.md:146-162` | confirmed |
| `KV_POLL_INTERVAL` exists and is the direct peer | read `config.py:44` | confirmed (`int = 10  # seconds between /metrics polls`) |
| `KV_POLL_INTERVAL` has a CLI flag | `grep -n "poll" buster_ripper/main.py` | **no match — the plan's claim is false (F-1)** |
| No `qa-project` skill exists | `ls` of `.agents/skills/`, `.claude/skills/`, and the global skill dirs | confirmed absent in all three |
| The existing suite mocks the upstream | read `tests/test_routes_integration.py:1-60` | confirmed (`TestServer` + `mocker.patch.object(backend, …)`) |
| `tests/conftest.py` is empty | read the file | confirmed (1 comment line) |
| No `Mode:` line in the dispatch | prompt inspection | OPTIONS mode |
| No orphaned reservation | `decisions.md` §`FK-003` | `State: open`; no `reserved` block remains |

## Next Step

1. **invite** presents the landscape: the three options for `FK-003`, the framing
   surprises above, and the carried QA infrastructure topic. Invite asks the first
   alignment question.
2. **interview** resolves `FK-003`; the one-FK-per-turn contract carries it.
3. **decide** then runs **PLAN-SELECTED** on the chosen option, which writes the
   `OI:` register (three entries), the Anchor Map finalization (including the
   `FK-003`-dependent record serializer row, the `F-1`/`F-2` repairs, and a
   `Mode: NEW` viewer row **only** if Option 3 is chosen), the VP/milestone matrices
   (`F-3`/`F-4`), and the full task-out.
4. **Final approval stays blocked** until `FK-003` resolves, PLAN-SELECTED lands
   with `User Approved: yes` on each `OI:` entry, and the `qa-project` skill exists
   for `QAP-MS001`/`QAP-MS002`.
