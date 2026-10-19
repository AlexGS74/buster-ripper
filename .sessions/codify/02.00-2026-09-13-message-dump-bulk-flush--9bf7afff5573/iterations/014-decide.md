# 014 — Decide (REPLAN-INCREMENT: repair `FK-006` — the encoding/format cluster)

## Orchestrator Prompt (verbatim)

> Execute the decide-interactive skill for this session. Arguments: $ARGUMENTS
>
> Session ID: 02.00-2026-09-13-message-dump-bulk-flush--9bf7afff5573
> Mode: REPLAN-INCREMENT
> Cluster: encoding/format cluster (FK-006, G-006)
> Reason (from Interview, verbatim): the user's answer selects none of the three presented options, and its repair changes the Anchor Map (T002 record builder + a possible Mode: NEW viewer row), T005, the VP-T002-* set, and the OI: replacement statement. Scoped to decide: repair the option set, reserve and fill a successor FK, then invite re-presents.
>
> [The remainder of the dispatch was the `decide-interactive` skill body plus the
> `codify-agents`, `critical-thinker`, `codify-user-prompt`, and
> `codify-deterministic-closure` skill bodies. `$ARGUMENTS` was empty. Question Mode is
> `deferred` per session metadata; no `Mode: plan-selected` line was present.]

## Session Type / Mode

- Continuing session `02.00-2026-09-13-message-dump-bulk-flush--9bf7afff5573`. Identity
  verified against `session-identity.json`: UID `9bf7afff-5573-402a-b102-827423a1dded`;
  handle suffix `9bf7afff5573` = first 12 hex of the UID; display id
  `02.00-2026-09-13-message-dump-bulk-flush` matches the handle prefix. **No
  `parent_session_uid`** → the parent coordination contract does not apply; there is no
  parent path to resolve, no sibling to inspect, and no `coordination status` to copy.
- **Mode: OPTIONS (the prompt carries no `Mode: plan-selected` line), scoped by the
  dispatch to the `FK-006` repair.** This is the fourth `decide` pass of the session
  (`003` initial, `008` flush-policy, `010`/`011`/`012` payload-format repairs). The
  scope is exactly: repair the option set → reserve and fill a successor FK → invite
  re-presents. **No full task-out, no VP/milestone finalization, no Anchor Map stance
  finalization, no `OI:` register approval** — those are PLAN-SELECTED, which the
  dispatch places *after* the pick (`invite` re-presents, `interview` resolves).
- **Question Mode: deferred.** No `AskUserQuestion` on this path. `decide` does not ask
  the user anything; it authors the successor FK with a complete `Presented` block and
  returns it for `invite` to present. Exactly one `FK` was reserved (the allocator's)
  and filled.
- **Input**: `iterations/013-organize.md` (the reroute input), `iterations/009-interview.md`
  §Round 3 (the answer), `decisions.md` §`FK-006`/§`US-008`/§`US-007`,
  `iterations/012-decide.md`, `plan-summary.md`, `handoff-context.md`, and the source.

## Context Conflict Detected

None. `decisions.md` §`FK-006` (`State: open`, `Owner: decide`, `Type: preference`,
`Recommendation: none`, `Presented` complete and unaltered), `plan-summary.md` §Current
Phase / §Open Questions 2b, and `handoff-context.md` §Current Status / §Open Forks all
name `FK-006` as the single open FK. The prompt's `Mode`, `Cluster` and `Reason` fields
match `009-interview.md` §Round 3 and `013-organize.md` §Next Step in substance; every
factual clause in them was re-derived from session files and source before use, and no
orchestrator-prompt technical detail conflicts with the session files. The prompt names
no `Mode: plan-selected`, so PLAN-SELECTED is **not** this pass — consistent with the
dispatch's own "then invite re-presents".

## Changes Since Iteration 012 (same phase, earlier cycle)

- NEW: **the fusion diagnosis** — `FK-006`'s failure mode is a *set shape*, not an option
  defect: it fused the encoding axis with the human-facing axis into three packages.
- NEW: **`FK-007`** reserved and filled — the repaired set, asking only the human-facing
  axis, with the encodings carried as fixed context.
- NEW: **`FK-006` marked `superseded`** (`Superseded-by: FK-007`) with a supersession note.
- NEW: **Deterministic Closure Gate** run on organize's `OPEN` row (ii) — the default
  encoding — with a `CLOSURE: absorbed` packet (below).
- NEW: **the Anchor Map construction boundary is settled** (organize 013 left it open):
  the encoding branch lands on `dump entry point`; the artifact's name lands on
  `collapsed dump artifact`. Recorded as a per-option stance delta, not finalized.
- UPDATED: `plan-summary.md` §Current Phase (compressed), §Options, §Open Questions 2b/2c.
- UPDATED: `handoff-context.md` §Current Status, §Open Forks, §Resolved Forks, §Notes.
- UNCHANGED: `FK-001`, `FK-002`, every other `Presented` text, the three `EI:`
  dispositions, `FK-A`/`FK-B`, the flush-policy cluster, `T001`/`T003`'s trigger half, the
  milestone skeleton, `RB-001`/`RB-002`/`RB-003`, `T002`/`T004`/`T005`'s existing text.
- NOT DONE, deliberately: no PLAN-SELECTED work (no done-when, no VP/milestone
  finalization, no `OI:` approval, no Anchor Map stance finalization) — the pick is unmade.

## Deterministic Closure Gate — organize's `OPEN` row (ii), the default encoding

Applied per `codify-deterministic-closure` before treating "does `US-008`(a) 'as one
option' demote the current encoding from the default?" as a user question.

```text
CLOSURE: absorbed
Finding: 013-organize.md §Goal Shape Analysis row 3 (`US-008`(a) default-encoding reading) / US-008 §Normative choice
Required outcome: which encoding a no-configuration process produces
Authority: US-007 → Normative choice ("the current change-focused format is retained rather than replaced"); FK-006 → Presented.Preamble item (2) ("'Keep the current' is honoured as the default … If you would rather a new encoding be the default, say so — it is a one-line change to this set") — presented, and not taken; US-008(a) ("stays available as one option among the selectable dump encodings — kept, not replaced" — a retention clause, silent on the default)
Admissible candidates: (a) the change-focused encoding stays the default; (b) a new encoding becomes the default
Rejected candidates: (b) — contradicts US-007's retention clause, and the user was shown (b) as an explicit one-clause correction in the presentation they answered and did not take it; US-008(a) does not ask for it (it asks that the current encoding remain *selectable*, which (a) preserves)
Selected closure: (a) — the change-focused encoding remains the default
Risk shape: unchanged — (a) is the state-preserving option; no user-visible behaviour moves, and the new encodings are additive
Deferred additive work: none
Owning route: decide (this pass) → plan-summary.md §Goal Definition Matrix `G-006` / §Validation Point Matrix `VP-T002-default-encoding`, finalized in PLAN-SELECTED
Question required: no
```

**Not papered over.** The row is not left implicit: `FK-007`'s preamble item (2) states the
default **explicitly, with its ground, and with a one-clause correction path**, so the user
sees the state they are getting and can overturn it in one clause if the closure is wrong.
That is the same correction mechanism `FK-006`'s preamble used and which the user did not
take. The closure is recorded here rather than silently assumed, and it makes
`VP-T002-default-encoding` a live validation point rather than a contingent candidate.

**Row (iii) — interactive viewer vs one-shot — is NOT closed**, and it is not asked as its
own question either: it is presented as two distinct options inside `FK-007` (the viewer and
the strictly cheaper one-shot form), so the user settles it by the same pick. **Row (i) — the
element's identity — is the `FK-007` question itself.**

## Goal Shape Consumption (organize 013)

Organize's Goal Shape Analysis is consumed, not re-derived. Disposition of its three `OPEN`
rows:

| Organize row | Disposition this pass |
|---|---|
| (i) the identity of the "more human-visible or human-friendly" element | **`FK-007`'s question.** The three readings are carried as the option set: the full-text encoding (the "human-friendly" reading, already in every `FK-006` option → a clarification), the small terminal viewer (the "human-visible" reading), and — for the literal "**add**" reading, an element the set lacks — the preamble's explicit invitation to name it, because the plan cannot build an unevidenced element and `US-003`'s "small tui" bounds a web UI out |
| (ii) whether `US-008`(a) demotes the current encoding from the default | **closed deterministically** (`CLOSURE: absorbed`, above); stated explicitly in `FK-007`'s preamble with a one-clause correction path |
| (iii) whether the viewer is interactive or the one-shot "print turn N" subset | **not closed** — presented as two options inside `FK-007` (options 2 and 3); it is a genuine preference with a stated cheaper subset |

`G-006`'s **goal tension** (`G-006`'s "without additional tooling" clause vs `US-008`'s
human-facing instruction) is **not** resolved here — it is arbitrated by the pick, and
PLAN-SELECTED writes `Met when` per encoding (F-11). `FK-007`'s options state the tension
inside the cost clause of each tool option ("the readability goal's 'without additional
tooling' clause has to be restated to admit it") so the user sees it while choosing.
`G-001` (write-count) is untouched by every option — stated as a fixed item in the preamble.

## Options Analysis

### The diagnosis (why a repair and not a re-presentation)

`FK-006` presented three packages, each a *shipped set of encodings* plus the viewer
question. Iteration 012 §"Why three options and not four" had already observed that "the
viewer is orthogonal to the encoding set" — and then packaged the viewer inside options 2
and 3. Organize 013 §"Structural observation" named the consequence: a reader who did not
experience the viewer as a thing being offered. The user's sentence then answered the two
axes **separately** ("keep the existing one as one option" / "add something more
human-visible or human-friendly"), which a fused set cannot absorb — every option already
keeps the existing encoding, so the first clause excludes nothing, and the second clause
names an element the set claims to have but the user did not experience as offered.

**Therefore the repair is a shape change, not a new option.** `FK-007` asks the
human-facing axis alone, names the encoding set as fixed context, and gives the
"add" reading its own explicit escape hatch.

### Option 1 — The full-text encoding is the addition; no new tool

- **What it is**: the offered set is two encodings and nothing else — the current
  change-focused record (kept, and still the default) plus a self-contained record carrying
  every turn's complete rendered text. Any turn opens and reads on its own, with no replay
  from turn 0; what changed between two turns is found by comparing them.
- **What it changes**: `dump_turn` gains an encoding branch; `config` gains an encoding
  global; `main.py` gains its flag; the README documents two encodings; the artifact's name
  states which one it holds (F-12).
- **What stays the same**: one artifact per session, one write per flush, the write count
  `F` versus `2N`, arrival order, the summary line, turn 0's full body, the default reading
  workflow, and `cat`/`grep`/`less` on both encodings.
- **Pros**: the smallest realization of the instruction; both encodings stay human-readable,
  so `US-005`'s interest is met without a new component; `G-006`'s "without additional
  tooling" clause survives **as written**; no `Mode: NEW` anchor beyond the encoding
  selector.
- **Cons**: the self-contained encoding costs ≈2.3× the bytes when selected; the change
  between turns is no longer shown in place; two serializers, two README sections, two test
  surfaces; the artifact no longer announces its encoding by its layout (F-12).
- **Complexity**: low-medium. **Risk**: low — the only irreversible part is shipping a
  documented encoding, which is what the user asked for.
- **Anchor stance delta**: `dump entry point` `extend` (owns the **encoding branch**);
  `collapsed dump artifact` `NEW` (owns the **artifact filename constant**, F-12, and the
  append target); `dump summary line` `extend` (encoding-independent, unchanged);
  **adds** a `Mode: NEW` config row for the encoding selector + its `--dump-format` flag. No
  tool row.
- **VP delta**: `VP-T002-change-readable` qualified per encoding (F-11);
  `VP-T002-collapsed-artifact` gains F-12's name clause; `VP-T002-default-encoding` live
  (closure above). No `VP-T002-human-facing` — nothing new is added for a reader beyond the
  encoding itself.

### Option 2 — Add a small terminal viewer

- **What it is**: option 1's two encodings and default, plus a command that lists a
  session's turns, shows a turn's reconstructed full text, and highlights what changed
  against the previous turn.
- **What it changes**: the plan gains a read-side command; `README.md:146-162` is rewritten
  to document a tool as well as the encodings; a third entry point exists.
- **What stays the same**: everything option 1 keeps, including `grep`/`less` working on
  the artifact; the artifact remains the primary record.
- **Pros**: navigation neither encoding gives on its own — the change-focused record makes
  you replay the chain, the self-contained one makes you scroll to find a turn; **zero
  dependency cost** (stdlib `curses` is available in this interpreter; `difflib` is already
  imported); it is the direction `US-003` raised and the "human-**visible**" half of
  `US-008`.
- **Cons**: a component this plan has never scoped, needing authorization and its own
  complete `RB-*` row; a third entry point in a repository that already has two `main`s
  (the `server.py:99` duplicate is an open `quality-backlog.md` confusability finding);
  `G-006`'s "without additional tooling" clause must be restated; more test surface; the
  README documents a tool.
- **Complexity**: medium. **Risk**: medium — scope, not correctness: a read-side component
  arrives in a session whose purpose is a write-side change, and `RB-003`'s read-path search
  found no code reader to build on, so it is a component from zero.
- **Anchor stance delta**: option 1's rows, **plus** a `Mode: NEW` tool row requiring its
  own complete `RB-*` row (organize 013 gathered the evidence: name search zero matches;
  `curses`/`difflib` as the nearest reusable primitives; neither supplies turn semantics,
  a listing, or an ordering rule); `T005` becomes tool documentation; **adds**
  `VP-T002-human-facing`.

### Option 3 — Add a one-shot "print turn N" command

- **What it is**: the two encodings, plus a non-interactive command that prints one turn's
  reconstructed text and its change against the previous turn, then exits.
- **What it changes**: a new command and a new README section; no terminal-state handling.
- **What stays the same**: everything option 2 keeps, minus the interactive surface.
- **Pros**: the same information on demand with far less surface — no key handling, no
  terminal-state management, much smaller test surface; `FK-006`'s own option text named it
  "a strictly cheaper subset of the same option".
- **Cons**: it cannot browse a session, so the "human-**visible**" half is delivered more
  weakly; it is still a new command, a new entry point, and a README section.
- **Complexity**: low-medium. **Risk**: low-medium — it under-delivers browsing if browsing
  was wanted.
- **Anchor stance delta**: option 1's rows, **plus** a `Mode: NEW` tool row (one-shot form)
  with its own complete `RB-*` row; **adds** `VP-T002-human-facing` in its one-shot form.

### Not an option — the machine record

Deliberately **not** an option here, and named as a clause in the preamble instead: it is
the one candidate that is *less* human-facing, so it cannot answer this question. The user
can name it and it rides the encoding set. Its measured cost (1.047× the plain-text log,
not smaller — F-10) and its consequence (for it, `cat`/`grep`/`less` stop working, so
`G-006`'s clause becomes encoding-dependent — F-11) are stated in the preamble so the user
chooses with the number in front of them.

### Not an option — a web/HTML view

`US-003`'s "small tui" bounds it out, and no evidence exists for it. Named in the preamble
as part of the "if none of these is the element you had in mind, name it" clause.

### Leaning

`decide` does **not** lean and records **`Recommendation: none`**. This is a `preference`:
no repository evidence ranks the three outcomes, and the instruction's own words ("more
human-visible **or** human-friendly") accept either property. Presenting a lean would be
pre-selection.

## Fork Authoring Record

- **Allocator call**: `codify codify reserve-decision-record --session-id
  02.00-2026-09-13-message-dump-bulk-flush--9bf7afff5573 --kind fork --owner decide` →
  **`FK-007`**, `State: reserved`, `Reserved at: 2026-09-14T14:07:45.444993+00:00`.
- **Filled in place** and set `State: open`. `Type: preference`; `Why-yours` quotes the
  escalation-floor clause verbatim ("it is irreversible or expensive to reverse (data loss,
  public API surface, security)"); `Trigger.Changed`/`Source`/`Replaces`/`Was` name the
  direct source lines; `Presented` carries the final question, preamble, three options, and
  `Recommendation: none`; `Supersedes: FK-006`.
- **`FK-006` marked `superseded`** with `Superseded-by: FK-007` and a supersession note. Its
  `Presented` block is **unaltered** — nothing the user saw changes.
- **One-line index moved** in `handoff-context.md`: `FK-006` → `## Resolved Forks`;
  `FK-007` → `## Open Forks`. No causal detail copied into the baton.
- **`US-008` is unchanged** — the statement is the trigger, and it is already recorded
  losslessly by `interview`; it is not re-authored, re-scoped, or resolved here.

## Anchor Map — construction boundary settled (organize 013 left it open)

Organize 013 flagged that three rows describe record *construction* with no stated
boundary. `decide` settles it (a design statement, not a stance change; the row text is
finalized in PLAN-SELECTED):

| Row | Owns | Mode |
|---|---|---|
| `dump entry point` (`dump.py:dump_turn`) | the **encoding branch** — what a record contains is chosen by the configured encoding, inside the function that already builds the whole record (`dump.py:61-109`: one `turn == 0` branch, one `turn ≥ 1` branch) | `extend` |
| `collapsed dump artifact` (`dump.py`, new symbol) | the **artifact filename constant** (the name that states the encoding — F-12) and the append target the flush writes to | `NEW` |
| `dump summary line` (`dump.py:84`, `:98-101`) | the per-turn **summary-string construction**, which is **encoding-independent** (it is the record's human-readable index line in every encoding) | `extend` |

**Why not a separate serializer symbol.** `FK-006`'s `Impact` said the artifact row's
"serializer becomes encoding-parameterized". That is corrected here: `dump_turn` already
builds the entire record, so a second serializer symbol on the artifact row would be the
wrapper/relay case the skill forbids. The artifact row owns the *name* and the *one-write
append*; the *record content* branch stays where the code is.

**Forward discretion (recorded, not asked).** If a tool ships, its entry-point form is a
mechanical, cheap-to-reverse choice: recommended reading is a **separate console script
declared in `pyproject.toml:[project.scripts]`**, not a third module-level `main` (which
would sharpen the existing `server.py:99` vs `main.py:62` confusability finding) and not a
new option group on the proxy's own CLI (which starts the server, a different job). The
artifact's name/extension carrying the encoding stays Forward discretion per F-12.

## Planned Invariant Themes (one line each — drafting is PLAN-SELECTED)

- `PI:T002/artifact_encoding/encoding_is_per_artifact` (hard) — unchanged in substance; its
  replacement statement's *membership* is now fixed by `US-007` + `US-008`(a) rather than by
  `FK-006`.
- `PI:T002/artifact_encoding/encoding_does_not_change_write_count` (hard) — unchanged;
  `T004` must run the write-count test **per shipped encoding**.
- **Candidate (tool options only)**: the tool lists a session's turns in **arrival order**
  (not as threads — F-8) and shows each turn's reconstructed text with its change against
  the previous turn. Becomes a planned invariant only if a tool ships.
- **Candidate (now live, not contingent)**: with no encoding configured, the change-focused
  encoding is produced — makes the default an observable contract rather than a config
  detail (grounded in the closure above).

## Validation Point Deltas (estimate-level — finalization is PLAN-SELECTED)

| Option | `VP-T002-change-readable` | `VP-T002-collapsed-artifact` | `VP-T002-human-facing` | `VP-T002-default-encoding` |
|---|---|---|---|---|
| 1 | qualified per encoding (F-11) | + F-12 name clause | not added | live |
| 2 | qualified per encoding | + F-12 name clause | added (viewer form) | live |
| 3 | qualified per encoding | + F-12 name clause | added (one-shot form) | live |

`VP-T003-*` are unchanged by every option. `T004` covers each shipped encoding in all
three. No `VP-*` handle is renamed by this repair.

## Dropped-Clause Sweep (`FK-006` → `FK-007`)

Triggered: this pass replaces a prior definition of the same scope (the `FK-006` presented
set). Direct predecessor only, one hop.

| `FK-006` clause | Disposition | Why / where it now lives |
|---|---|---|
| Preamble (1) "Your answer settles **which encodings exist**, not how one is chosen" | **replaced** | False for the successor, which asks the human-facing element and states the encoding set as fixed context. The `FK-006` sub-clause "one encoding per artifact, chosen once when the process starts" is **carried** (`FK-007` preamble 3) |
| Preamble (2) "'Keep the current' is honoured as the default … say so — it is a one-line change" | **carried** | `FK-007` preamble 2, and now **settled** by the closure gate |
| Preamble (3) write-count goal unchanged; byte cost depends on the encoding selected | **carried** | `FK-007` preamble 1 + 5 |
| Preamble (4) the machine-record figure correction (1.047×, not "most compact") | **carried** | `FK-007` preamble 3 + 5 |
| Option 1 "Two text formats, no viewer" (two encodings; change-focused default; self-contained opens any turn with no replay; ≈2.3×; no new component) | **carried** | `FK-007` option 1, restated so the human-facing property is explicit |
| Option 2 "Two text formats + small terminal viewer" (lists turns; shows reconstructed text; highlights the change; no new dependency — `curses` available, `difflib` imported; a new command not a new stack; cost = unscoped component + README documents a tool + `G-006`'s clause restated; the one-shot form is a strictly cheaper subset) | **carried** | `FK-007` option 2 (+ option 3 for the subset). The `difflib` line citation is `dump.py:4`, not `:5` (organize 013 Correction 3) — the claim holds |
| Option 3 "Two text formats + machine record + viewer": the **package** | **dropped — deliberate** | The two halves are separated, which is the repair: the viewer half is carried (options 2/3); the machine-record half is carried as a preamble clause the user can name |
| Option 3's machine-record clauses (one JSON object per turn; for `jq` and cross-session aggregation; measured 1.047×; value is script-readability without parsing diffs; `cat`/`grep`/`less` stop working so `G-006`'s clause becomes encoding-dependent) | **carried** | `FK-007` preamble 3 + 5 |
| Option 3's "a third serializer, a third README section and a third test surface" | **dropped — deliberate** | The machine record is a *nameable clause*, not an option, so its cost clause is not carried in `Presented`; it returns with the encoding if the user names it, and it is recorded here and in `FK-007` `Impact` (`T005` documents each shipped encoding) |
| `Recommendation: none`, `Type: preference`, `Why-yours` | **carried** | `FK-007` |
| `If-wrong` clauses (2.3× byte cost; machine record costs human readability; a tool spends unscoped scope) | **carried** | `FK-007` `If-wrong`, re-based on `US-008` |
| `Timestamp` / `Report` of the presentation | **superseded** | New `Presented` written; `FK-006`'s own block is preserved unaltered |
| The conversation-scoped encoding variant (lived in `FK-005`'s preamble, withdrawn from `FK-006`'s set) | **carried** | Not in `FK-006`'s `Presented`; carried outside the FK as a PLAN-SELECTED candidate (`plan-summary.md` §Open Questions 2e) |

All other clauses carried verbatim.

## Compaction Decision

- Prior same-phase files reviewed: `iterations/003-decide.md`, `iterations/008-decide.md`,
  `iterations/010-decide.md`, `iterations/011-decide.md`, `iterations/012-decide.md`.
- **Superseded**: none.
- **Not superseded**: all five.
- **Compaction performed: no.**
- **If no**: the slice differs and the prior files still carry live reasoning. `003` is the
  initial options pass (`FK-001`/`FK-002`); `008` is the flush-policy options pass plus the
  `FK-A`/`FK-B` escalation-floor re-tests; `010`/`011`/`012` are the three prior payload
  repairs. This pass **supersedes `FK-006`** (a record, in `decisions.md`) — not the reports
  that authored it: `012` holds the measured byte figures, the F-10/F-11/F-12 derivations,
  and the "why three options and not four" reasoning that `FK-007`'s diagnosis rests on, and
  `011`/`010` hold the F-6/F-8/F-9 derivations. `decisions.md` §`FK-006` and §`FK-007` are
  self-contained; the reports remain the provenance. Deleting any would remove reasoning a
  live option still depends on.

## Deferred Questions

- **Whether the tool is interactive or the one-shot form.**
  - Status: **NOT deferred** — presented as two options inside `FK-007` (options 2 and 3).
  - Impact: the tool task's size, its entry point, and `VP-T002-human-facing`'s form.
  - Agent analysis: `none` — a preference with a stated cheaper subset.
- **Which device/filesystem hosts `DUMP_DIR` in production** (Linux ext4/XFS beside vLLM,
  or macOS APFS on the dev box)?
  - Status: DEFERRED for /codify-review (unchanged).
  - Impact: the exact write-amplification factor differs; the directional verdict does not.
  - Agent analysis: `none` — a factual premise, not a user preference.
- **QA infrastructure setup** (`.agents/skills/qa-project/SKILL.md` absent) — unchanged; a
  presentation **topic**, not an FK. If a tool ships, its invocation joins the harness
  (already flagged in `interview`'s Topic 2 text).
- **The tool's entry-point form** — Forward discretion with a recommended reading
  (separate console script), recorded above; not a user question (mechanical, cheap,
  reversible), the same treatment `FK-006` gave the artifact's name/extension.

## Questions Asked

**No questions** — deferred mode. The one live user decision is `FK-007`, authored and
returned complete for `invite` to present. No `AskUserQuestion` call was made; `decide` does
not ask.

## Validation Performed

| Claim | How verified | Result |
|---|---|---|
| Session identity / handle suffix | `session-identity.json` vs handle | matches (`9bf7afff5573` = first 12 hex of UID) |
| No parent coordination contract | `session-identity.json` has no `parent_session_uid` | confirmed — nothing to resolve or reconcile |
| Exactly one open FK before this pass | `grep` of `State:` in `decisions.md` | `FK-006` open; `FK-001`/`FK-002` resolved; `FK-003`/`FK-004`/`FK-005` superseded |
| `FK-007` reservation is allocator-written and unique | `codify codify reserve-decision-record --kind fork --owner decide` | returned `FK-007` with `State: reserved`; filled in place, set `open` |
| No FK/US id derived by hand | none written except the allocator's | confirmed |
| `dump_turn` is the whole record path | `dump.py:60-109` | confirmed — one function, two branches, one summary append |
| `difflib` import line | `dump.py` | **line 4** (`:5` is `import json`) — organize 013 Correction 3 stands |
| The summary construction exists inline | `dump.py:84`, `:98-101` | confirmed — no symbol owns it |
| `config.DUMP_DIR` / `KV_POLL_INTERVAL` | `config.py:16`, `:44` | confirmed |
| `--dump-dir` is the only dump CLI option | `main.py:23-24` | confirmed — `--dump-format` is new surface |
| Two entry points today | `main.py:62`, `server.py:99`; `pyproject.toml:10-11` | confirmed |
| Dependencies unchanged | `pyproject.toml:5-8` | `aiohttp`, `click` |
| The documented read surface | `README.md:130-162` | confirmed — layout + "Reading the index" |
| The real dump is unreachable | `ls /Volumes` | `data` **not** mounted → every byte figure is carried, not re-measured |
| `FK-006` `Presented` text unaltered | `decisions.md` §`FK-006` | confirmed; only `State`/`Superseded-by` and the supersession note were added by this pass |
| No invariant registry | `ls .invariants` | absent |

## Gates Found Unmet (reported, not cured)

1. **`FK-007` is open** — the successor is filled and ready; it must be presented
   (`invite`) and resolved (`interview`) before PLAN-SELECTED can land.
2. **`.agents/skills/qa-project/SKILL.md` does not exist** while the plan declares
   `QAP-MS001`/`QAP-MS002`. Blocked on Topic 2's answers; not written here.
3. **Final approval remains blocked** on (1), on PLAN-SELECTED landing with
   `User Approved: yes` on each `OI:` entry, and on (2).
4. **`FK-006`'s `Presented.Report`** named `iterations/012-decide.md`; the *presentation* is
   recorded in `009-interview.md` §Round 3. Advisory field only; unchanged.
5. **`FK-006`'s `Impact` names the artifact row's "serializer"** as encoding-parameterized;
   this pass corrects the boundary (the encoding branch lands on `dump entry point`). The
   stale wording is preserved in the superseded record as review evidence and corrected in
   this report and in `plan-summary.md`.

## Next Step

1. **`invite`** re-presents `FK-007` — the question, preamble, and three options exactly as
   recorded in `decisions.md` §`FK-007` → `Presented`. One FK per turn.
2. **`interview`** records the answer into `FK-007` → `Resolution` and flips `State: open` →
   `resolved`. If the answer again selects no option, it returns to `decide` for a further
   successor rather than being forced.
3. **`decide` PLAN-SELECTED** on the pick — the `OI:` register (`output_layout`'s
   replacement statement per encoding, carrying F-6's definition of "full"), `G-006`
   re-derived (`F-11` + the `US-008` tension), the Anchor Map finalization (`F-1`/`F-2`
   repairs, the encoding selector as a `Mode: NEW` config row, the boundary settled above,
   plus a `Mode: NEW` tool row with its own complete `RB-*` row if a tool ships), the
   VP/milestone matrices (`F-3`/`F-4` + the two candidate VPs), and the full task-out.
4. **Final approval stays blocked** until `FK-007` resolves, PLAN-SELECTED lands with
   `User Approved: yes` on each `OI:` entry, and the `qa-project` skill exists.
