# 019 — Decide (REROUTE: payload/dump-layout cluster — the R1/R2 reading of `US-012`)

## Orchestrator Prompt (verbatim)

> Execute the decide-interactive skill for this session. Arguments: $ARGUMENTS
>
> Session ID: 02.00-2026-09-13-message-dump-bulk-flush--9bf7afff5573
> Mode: REROUTE
> Cluster: dump payload/layout cluster
> Reason (user words, verbatim): "Old behavior should stay unaffected until we prove a new one is better and it's worth switching to."
> Reading to verify against the words rather than assume: "old behavior stays unaffected" places the documented per-turn layout as the default with the collapsed format opt-in (organize's S-1 / R1), not merely available.

(`$ARGUMENTS` was empty. The dispatch carried the `decide-interactive`, `codify-agents`, `critical-thinker`, `codify-user-prompt`, `codify-deterministic-closure` skill bodies.)

## Session Type / Mode

- **Continuing session**; **REROUTE** (organize 018 → decide → invite → interview), scoped to the **payload/dump-layout cluster only**.
- Session identity re-verified against `session-identity.json`: display id `02.00-2026-09-13-message-dump-bulk-flush`, handle suffix `9bf7afff5573` = first 12 hex of UID `9bf7afff-5573-402a-b102-827423a1dded`. **No `parent_session_uid`** → the parent coordination contract does not apply; no parent contract was resolvable or owed.
- **Mode reading, stated so it is inspectable.** The dispatch names `Mode: REROUTE`, not `OPTIONS` and not `plan-selected`. Read against the phase's own rule ("prefer the narrow path: a reroute is for a broken model, not for ordinary sizing work plan-selected mode can do itself"), the *model* is not broken — organize 018 re-derived the contradiction surface and every binding still resolves — so this pass carries **PLAN-SELECTED duties for the reopened payload slice only** (full task-out, finalized Anchor Map stances, planned invariants, conflict matrix, VP/milestone matrices) and **OPTIONS duties for nothing**: no option set is manufactured, because the closure gate below admits exactly one shape. The TLS cluster is untouched — no `T006`, `MS-004`, `G-007`, `OI:` entry 4, `PI:T006/*`, anchor row, VP or task belonging to it is edited by this pass.
- **The dispatch's reading is a hypothesis, not authority.** Priority 5. It is verified below against `US-012`'s own words and the session's settled records, and it survives the closure gate on its own evidence — the verification, not the dispatch, is what settles it. Had the words excluded it, the dispatch's hint would have been recorded as a conflict and overridden.

## Context Conflict Detected

| Source | Says | Resolution |
|---|---|---|
| Priority 1 — `decisions.md#us-012` (user, 2026-09-16, round 14) | the documented per-turn layout **must not be deleted by the change that introduces the collapsed artifact**; old behavior stays unaffected **until** the new one is proven better **and judged worth switching to** | **Current authority.** Followed; it is the only rank-1 source on the ordering |
| Priority 1 — `decisions.md#fk-002` `Resolution.Normative choice` (user, 2026-09-13, round 2) | "The documented per-turn layout is **NOT** preserved"; `Options (settled)` still read the collapsed format as the default | **Superseded by the user's own later words.** Record is intact; `State` set to `superseded`, `Superseded-by: US-012` (see §Decision Recorded) |
| Priority 3 — `handoff-context.md` §Current Status | the reading is `OPEN` and "`decide` owns whether that is `CLOSURE: absorbed` (no fork) or a fork on the reading" | **Consistent.** The gate is run below; it returns `absorbed`, so the no-fork branch is taken and the reason is recorded rather than asserted |
| Priority 5 — the dispatch's `Reason:` field | the user's second message, verbatim | **Consistent with `US-012` and adds no gloss.** Its scope clause ("this cluster only") matches `US-012`'s own non-extension clause |
| Priority 5 — the dispatch's "Reading to verify against" line | the reading is S-1/R1 | **Verified independently, not adopted.** See §Deterministic Closure Gate; the gate's rejection of S-2 does not depend on the dispatch's framing |

## Input Absorption

| Input | Origin | Disposition |
|---|---|---|
| `US-012` — the documented layout must not be deleted by the increment that adds the new format | interview / user | Governing constraint; drives the closure and every finalization below |
| `US-012` — "unaffected" is not defined as *default* vs *merely available*; "prove it is better" is unmeasured | interview / user | **The reading is closed below by the gate; the measure is settled as an agent-decidable item** |
| `US-013` — the agents decide the remaining agent-decidable items; the escalation floor is not repealed | interview / user | Applied: F-24's design, the measure set, the switch's destination and `FK-002`'s `State` are settled here; nothing that contradicts `US-012` is settled by delegation |
| `F-13` — the two writers must be mutually exclusive per turn (`2N + F > 2N`) | organize 018 (derived) | **Carried as the constraint that forces the selector to choose *which* writer runs**; now an invariant (`PI:T002/dump_format/one_write_path_per_turn`) |
| `F-14` — `output_layout` is a `refine`, not an `override` | organize 018 | Carried; the register's entry 1 `Decision` cell is `refine` |
| `F-15` — goals/VPs are collapsed-path-only | organize 018 | Carried into the finalized rows |
| `F-16` — `VP-T002-default-encoding` is HELD | organize 018 | **Un-held and finalized** to the R1 clause (§Validation Point Matrix) |
| `F-17` — "old behavior unaffected" has no witness | organize 018 | `VP-T002-documented-format-preserved` retained and finalized |
| `F-18` — "prove it is better" has no measure and no evaluation point | organize 018 | Settled: M1 + M2 are the measures; `MS-001`'s paired runs are the evaluation point; M4 is explicitly not the proof; M3 is named as the physical axis with its trigger |
| `F-19` — the default's identity is `OPEN` | organize 018 | **Closed by the gate to S-1** |
| `F-20` — under S-1 the headline benefit is opt-in | organize 018 | **Recorded as a plan-visible disclosure** (`INTERPRETED`, not a fork) and fenced in the plan text |
| `F-21` — `T001`'s append is conditioned on the format | organize 018 | Carried verbatim into the task list |
| `F-22` — the capacity extraction grows | organize 018 | Carried; `T002`'s extraction spans both writers |
| `F-23` — `T005` is addition-only | organize 018 | Carried |
| `F-24` — the selector spans two axes; the `artifact_encoding` family name would be a misnomer | organize 018 | **Settled to design (i)** — one closed set of three values; the anchor term and the `PI:` family are renamed (§Anchor Map, §Planned Invariants) |
| The 12 findings `F-1`…`F-12`, the research increment, the capacity findings | prior | Carried by citation; not re-derived |

## User Statement Classification

**No new `US-` was reserved, and none should be.** The dispatch's `Reason` field quotes the user verbatim and those exact words are **already** canonically recorded as `US-012` (`User words`, both messages, in order). Re-reserving would duplicate an existing canonical US. The dispatch carries **no `Fork:` field**, so nothing here is an FK answer.

`US-012`'s clauses are **normative** throughout ("is premature", "should stay unaffected", "worth switching to" are sequencing and value judgments), which is why its `Factual claims: none` is correct and no claim row is manufactured for it. Its one **falsifiable premise** — that the plan deletes the old write path in the same increment that adds the new format — was verified by interview in round 14, re-verified by organize 018, and re-verified again this pass: `grep -n "write_text\|open(" buster_ripper/dump.py` → exactly `:82`, `:95`, `:105`, all three live, and `plan-summary.md` §Task List `T002` carried `REMOVED` clauses naming all three. The premise holds.

## Deterministic Closure Gate

`codify-deterministic-closure` loaded and applied in full before any question was considered. The gate is run on the **reading** (`F-19`), because that is the only thing that could have become a question this round.

### Required outcome (stated independently of any fix)

A payload shape that simultaneously satisfies: **(i)** the documented per-turn layout's behavior is unaffected until the collapsed format has been demonstrated better **and** judged worth switching to (`US-012`); **(ii)** one bulk flush is ONE write (`FK-002`); **(iii)** the session performs no more write-bearing operations than today.

### Authority (settled material only)

1. `decisions.md#us-012` — explicit user decision, 2026-09-16 (rank 1).
2. `decisions.md#fk-002` `Resolution.Normative choice` — explicit user decision, 2026-09-13, revised *in ordering only* by `US-012` (rank 1).
3. `decisions.md#us-007` — *"can we implment additional formats and keep teh current?"* / *"I'm for switchable option"* (rank 1).
4. `decisions.md#us-008` — *"Keep the existing one as one option and add something more human-visible or human-friendly."* (rank 1).
5. `decisions.md#us-013` — the delegation covers **agent-decidable** items and does not repeal the escalation floor (rank 1).
6. The derived arithmetic: `2N + F > 2N` for all `F ≥ 1` (rank 4 — verified counterexample, not preference).

### Candidates

| Shape | Default (no config) | Documented selected | Collapsed selected | Writes, N turns |
|---|---|---|---|---|
| **S-1** | documented | `2N` | `F` | `2N` or `F` |
| S-2 | collapsed | `2N` | `F` | `F` |
| S-3 | documented | `2N` | unreachable | `2N` |
| S-X | — | `2N + F` | `2N + F` | `2N + F` |

### Rejections

- **S-2** → violates `US-012`'s **second clause**. The clause gates the switch on a demonstration and a worth judgment ("*until we prove a new one is better and it's worth switching to*"). S-2 performs the switch at implementation time: the default changes on the day the code lands, before any comparison evidence exists and before the user has judged it. It also violates clause 1 under the plain reading, where "old behavior" is what a run with no configuration produces. S-2 survives only under the conjunction of *two* strained readings at once (clause 1 = "merely available" **and** clause 2's "switch" = removal-only). Neither is required by any settled source, and under the plain reading both fail. It is also the shape the user's *first* message refuses ("*deleting something before we implemented something better is premature*") applied to the product's default rather than to a file.
- **S-3** → fails `FK-002`'s normative requirement and `G-001` in **every** configuration: the reduction is unreachable in production, so the session ships a writer nothing can select. A strictly larger no-benefit change (dead branch, its tests, its documentation).
- **S-X** → `2N + F > 2N`: strictly worse than today while every test passes. Excluded by arithmetic and by `G-001` witness C.
- **Not an option:** "do nothing" / "defer the whole payload cluster" — `FK-002` settled that one bulk flush is one write, `FK-001` settled the triggers, and `US-012` refused only the *removal*; the collapsed format is still required. Manufacturing a do-nothing option would pad the set with an outcome the user has not asked for.

### Why S-1 is unique, not merely preferred

S-1 is compatible with **both** readings of clause 1 (under "merely available", S-1 gives strictly more: the documented format is certainly available *and* is the default). S-2 is compatible with only the strained reading of clause 1 and fails clause 2 regardless. There is therefore **no reading of the user's words under which S-2 is required and S-1 is forbidden**. S-1 is the unique admissible shape.

### Packet

```text
CLOSURE: absorbed
Finding: F-19 (organize 018 — the R1/R2 reading of US-012; iterations/018-organize.md §Ordering Shapes,
  §Goal Shape Analysis G-001/G-008)
Required outcome: the documented per-turn layout's behavior is unaffected until the collapsed format is
  demonstrated better and judged worth switching to (US-012); one bulk flush is one write (FK-002); no
  configuration performs more write-bearing operations than today
Authority: decisions.md#us-012 (rank 1); decisions.md#fk-002 Resolution.Normative choice (rank 1, revised in
  ordering by us-012); decisions.md#us-007, #us-008 (rank 1); decisions.md#us-013 (delegates agent-decidable
  items only); the derived inequality 2N + F > 2N
Admissible candidates: S-1 — the documented per-turn layout stays the DEFAULT and the collapsed formats ship as
  opt-in values of the dump format selector
Rejected candidates:
  S-2 (collapsed default, documented selectable) -> performs the switch US-012's second clause gates, before any
    demonstration exists and before the user judged it worth switching; also violates clause 1 under the plain reading
  S-3 (built, no switch) -> the reduction is unreachable in production, so FK-002's normative requirement and
    G-001 fail in every configuration; strictly larger no-benefit change
  S-X (both writers live) -> 2N + F > 2N, worse than today; excluded by arithmetic and G-001 witness C
Selected closure: S-1. config.DUMP_FORMATS = ("documented", "change-focused", "self-contained");
  config.DUMP_FORMAT defaults to "documented"; the format dispatch chooses which writer runs; the documented writer's
  three write sites are retained as the documented branch; the collapsed formats are opt-in
Risk shape: unchanged and reduced. The default path is byte-identical to today, so no existing reader, no documented
  contract and no existing test changes behavior. The only risk carried is that the session's headline reduction is not
  delivered by default — which is the direct consequence of US-012, disclosed as INTERPRETED below
Deferred additive work: the switch (flipping the default and/or deleting the documented writer) — owner: a later
  session; trigger: the user's judgment of the MS-001 comparison evidence. Recorded in roadmap.md
Owning route: decide (this pass) -> invite (landscape + one pass) -> interview (payload half of the invariant
  alignment pass)
Question required: no
```

**The escalation-floor test, run before concluding "no fork".** The floor's clauses are: (a) the choice would contradict a recorded user decision or a protected invariant; (b) it is irreversible or expensive to reverse (data loss, public API surface, security); (c) the plan surface gives no basis at all AND a wrong guess is expensive. None fits the **admissible** residue:

- (a) S-1 contradicts no live user decision — `US-012` is the later statement on the same question and it is the one that requires S-1. The only decision S-1 could be said to contradict is `FK-002`'s displacing ordering, which `US-012` itself revised; S-2 is the shape that contradicts a live decision, and S-2 is rejected, not asked.
- (b) S-1 is the **cheap-to-reverse** direction: a later default flip is one value in `config.py` plus a branch deletion, with no data migration and no contract break while both formats are documented and shipped. The expensive direction is shipping S-2 and having to withdraw a default users have already depended on.
- (c) The plan surface **does** give a basis — `US-012`'s own gating clause — so this clause cannot apply.

**Therefore no fork is raised.** Per the decide rule, a `ratification` with `Changed:` empty is a contradiction, and a `preference` whose answer is settled by rank-1 authority is pre-selection. The question does not survive the gate; it becomes a **disclosure** (§Disclosure).

## The Disclosed Consequence (INTERPRETED — not a fork, not silent)

**The consequence of S-1, in user terms:** the session's headline benefit is **opt-in**. A user who configures nothing gets today's `turn_000.json` / `turn_NNN.diff` / `index.txt` at today's `2N` writes; the buffering, the flush timer, the volume trigger and the shutdown flush are all inactive until `--dump-format change-focused` (or `self-contained`) is passed. The write reduction the session exists for is **not delivered by default**.

**Why this is a disclosure and not a Deferred Ruling about unspecified behavior.** The behavior is *specified* — by `US-012`'s second clause, which holds the switch back until a demonstration exists and the user has judged it. The Fix Decision Principle's INTERPRETED path covers behavior *the plan never approved*; here the plan's approval (`FK-002`) is exactly what the user revised. What the plan never approved is the *consequence*, and a consequence is disclosed, not decided.

**How it reaches the user, by two independent routes so it cannot be missed:**

1. **`plan-summary.md` §Goals and Constraints** carries it as a Topic-level caveat, and `G-001`/`G-008`'s `Met when` clauses state it.
2. **`OI:` entry 1's replacement statement** carries the default clause, and that entry is a **user-only stamp** that has not been written — so the user reads and approves exactly the sentence that produces this consequence, in the payload half of the invariant alignment pass.

**Fenced against the tempting wrong reading.** The plan text names the misreading and forbids it, per the plan-format rule: *"the documented layout is retained"* does **not** mean *"the collapsed format may be the default"*. Retention is about the writer surviving in the code as a selectable branch; the default is a separate clause (`PI:T002/dump_format/default_format_is_documented`) and it is `documented`.

## Decision Recorded — `FK-002` supersession

- **`FK-002` `State`: `superseded`; `Superseded-by: US-012`.** The successor is a **user-statement record, not an FK**, and this is deliberate: the revision came from the user directly in round 14 and the derived reading was absorbed by the closure gate, so **no question was escalated and no successor FK exists to name**. The `Superseded-by:` field's documented form is `FK-NNN`; `US-012` is written instead because it is the record that actually supersedes the clause, and a note in the block says so. Leaving `State: resolved` was rejected: the field is what a skimming reader sees first, and the revised clause is exactly the one the `### Revision under review` pointer was added to stop a reader applying.
- **`US-012`'s block is not edited.** It is owned by `interview`; its prose already names `FK-002` and its `Impact.Plan` already names every destination. No field of it is rewritten by this pass.
- **No `FK-` and no `US-` ID was reserved, filled, superseded or reused by this pass.** `FK-002`'s `State`/`Superseded-by` are writes to a record this phase owns (`Owner: decide`).

## Options Analysis

**No option set is presented, and that is the gate's output, not an omission.** The closure returned `absorbed`: exactly one shape is admissible under settled authority, so a menu would be padding (`S-2`/`S-3`/`S-X` are not alternatives merely because they can be described — two violate `US-012`, one is worse than today). The single path, stated directly:

### The path: S-1 — the documented layout stays the default; the collapsed formats ship as opt-in values

- **What it is**: the dump format becomes a single configuration choice with three values. `documented` is the default and produces today's `turn_000.json`, `turn_{N:03d}.diff` and `index.txt` exactly as they are now. `change-focused` and `self-contained` are opt-in collapsed formats that buffer turns in memory and write one artifact per flush. The dispatch chooses **which** writer runs, so no turn is ever written by both.
- **What it changes**: `config.py` gains `DUMP_FORMATS` and `DUMP_FORMAT`; `main.py` gains `--dump-format`; `dump.py` gains the dispatch, the two collapsed record builders and the artifact path builder, and `dump_turn`'s three write sites become the documented branch instead of being deleted; `server.py`'s `on_cleanup` gains a wrapped flush and cancels a second task; `README.md` gains the collapsed formats beside the unchanged documented-layout block.
- **What stays the same**: a default run's output — same files, same names, same `index.txt` line format, same `2N` writes, same reading workflow at `README.md:154-159`. Also unchanged: `/v1/messages` behavior, the flush-trigger policy (`FK-001`), the artifact's append-only rule, the session keying, and the whole TLS cluster.
- **Pros**: the only shape compatible with `US-012` under every reading; the default path is byte-identical, so no existing reader, test or documented contract moves; the switch stays available and is cheap (one default value, one branch deletion); the comparison evidence the user asked for is produced at `MS-001` before anything is displaced.
- **Cons**: the session's headline reduction is **not** delivered by default (disclosed above); the buffering/flush machinery is inactive on the default path, so the default path exercises less of the new code than a collapsed run does; `dump.py` grows rather than shrinks (the documented writer is retained beside the new ones) — Green by file line count, but the earlier ~150–170 projection is void.
- **Complexity**: medium — the same as the plan's shape, plus one branch and one retained writer.
- **Risk**: low. The risky direction is the one not taken.
- **Rejected because**: n/a — this is the selected path.

## Finalized Agent-Decidable Items (`US-013`'s delegation)

**1. `F-24` — the selector's design: ONE closed set of three values (design (i)).**

```python
config.DUMP_FORMATS: tuple[str, ...] = ("documented", "change-focused", "self-contained")
config.DUMP_FORMAT: str = "documented"
```
`main.py`: `--dump-format`, `type=click.Choice(list(config.DUMP_FORMATS))`, `default="documented"`, `show_default=True`, assigning `config.DUMP_FORMAT` **through the option's own `callback`** (so `main()` gains no parameter — the capacity closure of 017 still holds).

- Why not design (ii) (a layout switch plus a two-value encoding set, two globals): it adds a second global and a second flag for an axis whose only values are sub-choices of a layout that has already been selected; it splits `PI:.../format_set_is_closed` into two closed sets; and it forces the artifact's filename to encode two independent variables where `F-12` settled that it carries **the configured value**. Design (i) keeps one injectability seam, which is also what `US-010`'s harness contract names (`--dump-format <format>`, singular).
- Why the flag keeps its name: `--dump-format` is already the token the plan and the harness contract use, and the *term* "format" spans both axes honestly (a value names a complete output shape). What changes is the **Anchor Map term** and the **`PI:` family name**, which said "encoding" and would mislead the implementer into building an encoding-only set.

**2. The measure "better" is proved on.** **M1** (write-bearing file operations per session) and **M2** (files created in `DUMP_DIR/<sid>/`) are the decisive, deterministic measures: `2N` → `F` and `N + 1` → `1`. **M4** (logical bytes) shows **no** improvement (≈ +1.5%, `F-6`) and must **not** be presented as the proof. **M3** (physical bytes × WAF) is the real axis (`007` F7) and is **not measurable in this repository** (`/Volumes/data` is unmounted, re-verified this pass) — so a proof on the write *count* is a **proxy** proof, and the plan says so in `G-001`. The evaluation point is `MS-001`: the **paired runs** (documented vs collapsed, same N) plus the two directory listings, recorded side by side.

**3. The switch's destination.** A **`roadmap.md` entry** — owner: a later session; trigger: **the user's judgment of the `MS-001` comparison evidence**. This is the closure algorithm's step-6 case (complete-now vs additive-work-later), so it is absorbed and deferred with a named trigger and owner rather than asked. Written into `roadmap.md` this pass.

**4. `FK-002`'s `State`/`Superseded-by`.** Settled above.

## Anchor Map Stance Finalization

**One stance delta, no `Mode` change, one term rename.** The `Mode` tokens were re-verified this pass: every `Mode: NEW` row in the payload cluster still cites a complete `RB-*` row (`RB-001`, `RB-002`, `RB-003`, `RB-004`), and the TLS cluster's `RB-005` is untouched.

| Term | Binding | Mode | Stance decision |
|---|---|---|---|
| dump format selector **(renamed from "dump encoding selector")** | `config.py:DUMP_FORMAT, DUMP_FORMATS` + `main.py:--dump-format` | **NEW** (unchanged) | Renamed because the value set now spans the **layout** axis as well as the record-encoding axis (`F-24`), settled to design (i). Same binding, same `Mode`, same `RB-004` — a term rename, not a new referent, so no new `RB-*` row is owed and no re-search is required (018 re-ran `RB-004` under exactly this widened role and returned zero selector matches) |
| dump entry point | `dump.py:dump_turn` | **extend** (unchanged) | Its role is finalized as the **format dispatch**; the documented writer is a retained branch, not a removal target. The three write sites stay in the code |
| collapsed dump artifact | `dump.py` (filename constant + path builder + append target) | **NEW** (unchanged) | **Added**, never in place of the documented layout. The artifact's filename carries the configured value for the two collapsed values (`<value>.log`); the documented value produces the three documented files and no `.log` |
| all other rows | — | unchanged | No row added, removed, or re-stanced by this pass; no `candidates:` token survives anywhere |

**Private helpers are still not rows.** `T002`'s capacity extraction creates module-private functions inside `dump.py` — the same concepts decomposed for LNCP, authorizing no new artifact.

## Planned Invariants In Scope

**The `PI:T002/artifact_encoding/*` family is renamed to `PI:T002/dump_format/*`.** The old family name is a misnomer for a set that now contains `documented`, a layout rather than an encoding (`F-24`), and the misnomer is exactly the kind that misleads an implementer into building an encoding-only set. None of the four entries was ever confirmed or approved, so nothing approved is being renamed; every reference is updated in `plan-summary.md` this pass.

### `PI:T002/dump_format/format_is_per_artifact`
- **Statement**: one artifact carries exactly one configured format; the format is chosen once per process (`config.DUMP_FORMAT`, set by `main()`, never mutated); a collapsed artifact's filename states the configured value.
- **Reason**: the artifact's identity is `(sid, format)`; without this, a second format silently appends to the first format's file.
- **Strength**: hard_requirement.

### `PI:T002/dump_format/format_does_not_change_write_count`
- **Statement**: the configured format changes what a record *contains*, never how many write-bearing operations a session performs.
- **Reason**: protects `G-001`; the encoding axis was already claimed not to move the count, and the layout axis moves it only by *selecting a writer*.
- **Strength**: hard_requirement.

### `PI:T002/dump_format/default_format_is_documented`
- **Statement**: with no format configured, the output is the documented per-turn layout (`turn_000.json`, `turn_{N:03d}.diff`, `index.txt`) at `2N` write-bearing operations; the collapsed formats are opt-in.
- **Reason**: `US-012` clause 1 read with clause 2. **This replaces the contested `PI:T002/artifact_encoding/default_encoding_is_change_focused`**, which would have been false on the day it was written under S-1.
- **Strength**: hard_requirement. Witness: `VP-T002-default-encoding` (finalized below).

### `PI:T002/dump_format/format_set_is_closed`
- **Statement**: the accepted set is exactly the three values in `config.DUMP_FORMATS`; an unknown configured value is a hard failure, never a silent fallback.
- **Reason**: each shipped value becomes a documented format a user may depend on, so the set's size is the irreversible part.
- **Strength**: preferred.

### `PI:T002/dump_format/one_write_path_per_turn`
- **Statement**: for any configuration, a session of N turns performs **either** `2N` **or** `F` write-bearing operations — never `2N + F`.
- **Reason**: `F-13`'s derived constraint. The record path is split into two writers; this is the invariant that keeps the split from producing a per-turn double write.
- **Strength**: hard_requirement. Witness: the paired runs of `VP-T002-write-count-reduced` and the two listings of `VP-T002-file-count-reduced`.

### `PI:T002/dump_format/documented_format_is_retained`
- **Statement**: the documented writer remains in the code as a selectable value and produces today's three files with today's content and line format when selected; no increment removes it before a demonstration the user judged worth switching to authorizes it.
- **Reason**: `US-012`'s requirement, and the theme no existing invariant covers. It is what a future increment would violate by deleting the branch.
- **Strength**: hard_requirement. Witness: `VP-T002-documented-format-preserved`.

### Unchanged
- `PI:T006/upstream_tls/verification_on_unless_explicitly_disabled` and `PI:T006/upstream_tls/optout_is_process_wide` — **confirmed in round 15 by delegation (`US-013`)**; byte-unchanged by this pass.
- **Not authored**: the third `PI:T006` ("the skip is announced") — `FK-008` resolved to no runtime record, and an invariant naming an unshipped behavior is a prescription the code would violate on the day it is read.

## Invariant Conflict Matrix

| Planned invariant | Existing invariant | Relationship | Decision | Why |
|---|---|---|---|---|
| `PI:T002/dump_format/format_is_per_artifact` | `EI:buster_ripper.dump/dump_turn/output_layout` | **refines** | **refine** (018 — was `override`) | The documented property is **true whenever the documented format is selected**, so it survives and is only scoped. An override requires displacement, which `US-012` forbids (`F-14`) |
| `PI:T002/dump_format/format_is_per_artifact` | `EI:.../turn_zero_is_full_body` | refines | **preserve** | Every shipped format keeps turn 0's raw body **and** its rendering (`F-6`) |
| `PI:T002/dump_format/format_does_not_change_write_count` | `EI:.../sequential_diff_chain` | refines | **preserve** | The append-time `turn`/`prev_text` sequencing is format-independent |
| `PI:T002/dump_format/default_format_is_documented` | `EI:.../output_layout` | refines | **refine** | Replaces the HELD row. Under S-1 the default is the documented layout, so the statement is true on the day it is written — the direction is now settable |
| `PI:T002/dump_format/one_write_path_per_turn` | *(none)* | unknown | **preserve** | No existing invariant constrains the number of write paths per turn; new surface derived from `F-13` |
| `PI:T002/dump_format/documented_format_is_retained` | `EI:.../output_layout` | refines | **refine** | The retained-and-selectable clause is the scope qualifier `output_layout` needs; nothing is displaced |
| `PI:T002/dump_format/format_set_is_closed` | *(none)* | unknown | **preserve** | No existing invariant constrains the size of the format set; new surface from `US-007`/`FK-007` |
| `PI:T006/upstream_tls/verification_on_unless_explicitly_disabled` | `EI:.../on_startup/shared_client_verifies_by_default` | refines | **preserve** | Unchanged; confirmed round 15 |
| `PI:T006/upstream_tls/optout_is_process_wide` | *(none)* | unknown | **preserve** | Unchanged; confirmed round 15 |

**None of the three dump `EI:` entries is `known_bug` or `likely_accidental`** — all three are `intended`, which is why the bias toward `override`/`remove` does not fire. **The payload cluster has no override target at all**: all three are refinements, and the register keeps all three payload entries as `refine` (see below — this corrects 018's "entry count drops to two", which would have removed the user's own ratification point for the reversal).

## Approved Override Register (finalized — `User approved` remains user-only)

| OI | Old invariant | Decision | New invariant | User approved |
|---|---|---|---|---|
| `OI:buster_ripper.dump/dump_turn/output_layout` | session dir holds `turn_000.json` / `turn_{N:03d}.diff` / `index.txt` | **refine** (018 — was `override`) | **Scope qualifier + default clause.** The property is **unconditional on the default path**: with the documented format selected (or nothing configured) the session dir holds `turn_000.json`, `turn_{N:03d}.diff` and `index.txt` **unchanged**, at `2N` write-bearing operations. **When a collapsed format is selected** the session dir instead holds one artifact for that format, appended once per non-empty flush, named `DUMP_DIR/<sid>/<format>.log` where `<format>` is exactly the configured value. One format per artifact, chosen once per process; **the default is `documented`** — *"the documented layout is retained"* does **not** mean *"a collapsed format may be the default"*. **change-focused** records: one summary line per turn; turn 0 carries the full normalized JSON body **and** its `_render_body` rendering; turn ≥ 1 carries the unified diff of `_render_body` output against the previous turn. **self-contained** records: one summary line per turn; every turn carries its complete `_render_body` rendering; turn 0 additionally carries the full normalized JSON body. Turn 0's "full body" therefore means **both** the raw normalized JSON body and its rendering (`F-6`), in every shipped collapsed format | **pending — user only** (entry 1 stays REFUSED until the user stamps it; the refusal was round 14 and nothing this pass cures it) |
| `OI:buster_ripper.dump/dump_turn/sequential_diff_chain` | …`prev_text` advances only when a turn is written | **refine** | `prev_text` and `turn` advance at **append** time, never at flush time, so a flush never resets the chain. For a collapsed format each turn's record derives from the previous turn's rendered text, in arrival order; the append-time sequencing is format-independent (`F-11`). The clause reads **per selected format**: the documented format's chain still advances when its turn is written and its diff files are unchanged | **pending — user only** (no longer contingent: the shape is settled, so it is presentable) |
| `OI:buster_ripper.dump/dump_turn/turn_zero_is_full_body` | turn 0 is written as a full JSON body | **refine** | Turn 0's full normalized JSON body **and** its rendered rendering are present in the session artifact once flushed, never only a diff — in **every** shipped collapsed format (`F-6`). The documented format's `turn_000.json` already satisfies this verbatim and is **preserved**, not refined | **pending — user only** (no longer contingent) |
| `OI:buster_ripper.server/on_startup/shared_client_verifies_by_default` *(TLS — untouched this pass)* | the shared `ClientSession` (`server.py:47`) is built with no `ssl`/`connector=` argument, so aiohttp's `ssl=True` default applies | **refine** | unchanged from 017 | **yes — by explicit delegation** ([`US-013`](decisions.md#us-013)) |

## Goal Definition and Coverage Finalization

**`G-001` `Met when` — finalized (was `OPEN`).** For N turns flushed `F` times **with a collapsed format selected**, total write-bearing file operations `= F` versus `2N` for the same N with the documented format selected; reduction `= 1 − F/(2N)`, i.e. **50% ≤ reduction < 100%** for every achievable `F`, for every shipped format. **With no format configured the documented layout is produced at `2N` writes, so a default run does not meet this clause — the reduction is opt-in (`US-012`).** Failure witnesses A, B and C unchanged.

**`G-002`/`G-004`/`G-005`** keep 018's collapsed-format scope qualifier verbatim. **`G-003`, `G-006`, `G-007`** unchanged.

**`G-008` `Met when` — finalized (was `OPEN`).**
- **(a)** With **no format configured**, N turns produce exactly `turn_000.json`, `turn_{N:03d}.diff` and `index.txt` with today's content and line format, at `2N` write-bearing operations. *(Settled to the default reading: the documented layout stays the default.)*
- **(b)** A comparison of the collapsed path against the documented path exists at `MS-001`, naming its measure (**M1** and **M2**).
- **Alternate readings — SETTLED** (`CLOSURE: absorbed`, §Deterministic Closure Gate): reading (a) is taken. Reading (b) ("merely available") is excluded because it performs the switch `US-012`'s second clause gates. **Failure witnesses W1, W2, W3 remain live** — W1 is now the shape the plan must not drift into.
- **Boundary** unchanged: only the dump layout; not the TLS cluster, not `--upstream`'s default, not the flush-trigger values.

**Goal Coverage** — `G-008`'s disposition changes from **reading-dependent** to **mapped**: T001, T002, T004, T005 → `VP-T002-documented-format-preserved`, `VP-T002-write-count-reduced`, `VP-T002-file-count-reduced`. Every other row unchanged; no orphan task.

## Validation Points — Finalized

| Validation point | Change this pass |
|---|---|
| **`VP-T002-default-encoding`** | **UN-HELD.** The clause is finalized to the R1 branch: *with no `--dump-format` passed and no config override, the output is the **documented layout** (`turn_000.json`, `turn_{N:03d}.diff`, `index.txt`) at `2N` write-bearing operations.* The R2 candidate clause is recorded in the row's history and must not be authored. The handle is stable; `Observable after: T002`, `Validate at: MS-001`, `QAP-MS001` unchanged. This is the witness for `PI:T002/dump_format/default_format_is_documented` and for `G-001`'s default clause |
| `VP-T002-documented-format-preserved` | Unchanged clause; `F-17`'s witness for `G-008`(a) and for `PI:T002/dump_format/documented_format_is_retained` |
| `VP-T002-file-count-reduced` | Unchanged clause; measure **M2** |
| `VP-T002-write-count-reduced` | Unchanged clause, plus its **paired-runs** role: the documented run and the collapsed run at the same N **are** the comparison `G-008`(b) requires, on measures **M1** and **M2** |
| `VP-T001-writes-deferred`, `VP-T002-volume-flush`, `VP-T003-interval-flush`, `VP-T003-shutdown-flush`, `VP-T002-collapsed-artifact`, `VP-T002-change-readable`, `VP-T002-turn0-available` | Unchanged (018's collapsed-format scope qualifiers carry) |
| `VP-T002-human-facing`, `VP-T002-single-write-per-flush`, `VP-T006-optout-is-explicit`, `VP-T006-dump-cluster-unaffected` | Still NOT CREATED / RETIRED; handles must not be reused |
| every `VP-T006-*` | Untouched |

**`T004` un-held clause.** The test for `VP-T002-default-encoding` is no longer held: `T004` writes it against the finalized R1 clause. Nothing else in `T004` moves.

## Task List — Finalization of 018's Deltas

018's five payload deltas are **adopted as written**, with the following items now settled rather than `OPEN`. No task is added or removed; `T006` is untouched.

- **`T001`** — append to `state.pending`/`state.pending_bytes` **only when a collapsed format is selected** (`F-21`); `turn`/`prev_text` advance unconditionally at append time. Unchanged otherwise.
- **`T002`** — the two superseded `REMOVED` clauses stay dropped. Settled here: `config.DUMP_FORMATS = ("documented", "change-focused", "self-contained")`; `config.DUMP_FORMAT = "documented"`; `--dump-format` with `click.Choice`, `default="documented"`, `show_default=True`, assigning through its own `callback`; the documented writer's three sites (`dump.py:82`, `:95`, `:105`) **retained as the documented branch**; the collapsed artifact named `DUMP_DIR/<sid>/<format>.log`; `flush_all()` with no `await`, never raising, no-op when `DUMP_DIR is None` or nothing is pending; `dump.py`'s module docstring updated; `dump.py:63`'s bare `assert` replaced by a real `ValueError` guard. The capacity extraction spans both writers (`F-22`).
- **`T003`** — trigger machinery active only on the collapsed path; `flush_all` finds nothing pending on the documented path and the timer no-ops. Unchanged otherwise.
- **`T004`** — adds the documented-format preservation test, the file-count test, the per-format write-count pair (the comparison), and the now-un-held default-format test.
- **`T005`** — addition-only (`F-23`): nothing is removed; the documented layout block and "Reading the index" stay current and correct; the collapsed formats, the selector, each format's reading workflow, `F-8`'s limitation and the pre-existing-files note are added.
- **`T006`** — untouched.

## Compaction Decision

- Prior same-phase files reviewed: `003-decide.md`, `008-decide.md`, `010-decide.md`, `011-decide.md`, `012-decide.md`, `014-decide.md`, `015-decide.md`, `017-decide.md`.
- **Superseded: none.**
- **Not superseded: all eight.** Slice reasoning: `003` is the original three-option payload/trigger decision; `008` the flush-policy parameters and the `FK-A`/`FK-B` non-askability; `010`/`011`/`012` the encoding-dial answers; `014` the artifact boundary and the human-facing pick; `015` the `FK-007` plan-selected finalization; `017` the **TLS** cluster's finalization. This pass settles the **ordering/layout-retention** slice (`US-012`), which none of them covered — `015` in particular remains the valid record of the encoding axis and of the artifact's name, and its `RB-004` evidence is an *input* here, not a superseded conclusion.
- **Compaction performed: no.** No prior decide report's FKs are affected (no FK was authored or amended by this pass), and each older report remains the process record of a distinct slice. `017-decide.md` still describes the TLS cluster as it was decided; it is not edited.
- If no: no blocker; the reason is slice-based, per the compaction rule.

## Dropped-Clause Sweep

**Triggered** — this pass wholesale replaces three definitions: `FK-002`'s `Resolution.Normative choice` (its ordering clause), `OI:` entry 1's replacement statement (its default clause), and `G-001`'s / `G-008`'s `Met when` clauses. One hop only; predecessors enumerated from the artifacts being replaced and from `FK-002`'s lineage.

| Clause | Disposition | Why / where it now lives |
|---|---|---|
| `FK-002` — "The documented per-turn layout is NOT preserved" | **dropped — deliberate** | Reversed by `US-012` (the user's own later words); `FK-002` is now `superseded`. The record's `User words`, `Presented`, `Factual claims` and `Impact` are **carried verbatim** |
| `FK-002` — "one bulk flush is ONE write" | **carried** | `US-012` does not retract it; it is the premise `F-13` is derived from. Still in `FK-002` and in `G-001` |
| `FK-002` — the three `Factual claims` | **carried verbatim** | `decisions.md#fk-002`, untouched; the revision is normative only |
| `FK-002` — `Presented` block (both options, the recommendation) | **carried verbatim** | Unaltered; the user's round-2 answer is a true record of what was asked and answered |
| `FK-002` — "`FK-A`'s `½·(1 − F/N)` framing is displaced arithmetic" | **carried** | Still true and still cited in `plan-summary.md` §Current Phase |
| `OI:` entry 1 — "the default is `change-focused`" | **dropped — deliberate** | Replaced by "the default is `documented`" (`US-012` clause 2 read with clause 1; `F-19` closed by the gate) |
| `OI:` entry 1 — "**override**" direction | **dropped — deliberate** (018) | `F-14`: an override requires displacement, which `US-012` forbids. Direction is `refine` |
| `OI:` entry 1 — "the session dir holds one collapsed per-session artifact" | **restored — scoped** | True when a collapsed format is selected; the documented selection holds the legacy three files |
| `OI:` entry 1 — the artifact's name `DUMP_DIR/<sid>/<encoding>.log` | **carried, re-tokened** | Unchanged for the collapsed values; `<encoding>` becomes `<format>`, exactly the configured value (`F-12`) |
| `OI:` entry 1 — the change-focused and self-contained record contents; turn 0's "full body" = body **and** rendering | **carried** | `plan-summary.md` §Approved Override Register entry 1; `T002`'s `ADDED` list |
| `OI:` entries 2–3 — their replacement statements | **carried, re-scoped** | Their "contingent" status is discharged (the shape is settled); each statement now reads **per selected format** |
| `OI:` entry 4 and the two `PI:T006/*` | **carried verbatim** | Round 15's delegation; untouched by this pass |
| `G-001`'s "reduction `= 1 − F/(2N)` … **for every achievable F**" | **carried, and scoped** | Now states the default path's clause explicitly (the reduction is opt-in) |
| `G-001`'s failure witnesses A, B, C | **carried** | Unchanged |
| `G-008`'s "(a) … **under R1** / Under R2 …" | **dropped — replaced** | The reading is settled; (a) is now unconditional |
| `G-008`'s "They classify the same implementation oppositely → `OPEN`" | **dropped — replaced** | `CLOSURE: absorbed`; reading (b) is inadmissible, not merely unchosen |
| `G-008`'s failure witnesses W1, W2, W3 | **carried** | Unchanged; W1 is now the plan's fence |
| `G-002`/`G-004`/`G-005`'s collapsed-format scope qualifiers | **carried** | `F-15` |
| `PI:T002/artifact_encoding/*` — the family name | **dropped — deliberate** | Renamed `PI:T002/dump_format/*`; the old name misdescribes a set containing `documented` (`F-24`) |
| `PI:.../default_encoding_is_change_focused` — the statement | **dropped — replaced, never approved** | Replaced by `default_format_is_documented`, which is true on the day it is written under S-1. The old entry was `CONTESTED` and never confirmed, so no override register entry is owed |
| `PI:.../encoding_is_per_artifact`, `.../encoding_does_not_change_write_count`, `.../encoding_set_is_closed` | **carried, renamed** | Same statements and strengths under the `dump_format` family |
| `VP-T002-default-encoding`'s two candidate clauses | **dropped — replaced** | The R1 clause is authored; the R2 clause is recorded in the row's history and must not be authored |
| `T002`'s `REMOVED` clauses for the three write sites; `T005`'s two `REMOVED` clauses | **carried as dropped** (018) | Replaced by 018's replacement clauses, adopted verbatim; nothing is removed from `dump.py` or `README.md` |
| The session Topic ("Buffer captured messages … flush on Ctrl-C/SIGTERM") | **carried, with a caveat** | The machinery is inactive on the default path; the caveat is now written into §Goals and Constraints so the plan does not read the Topic as delivered by default |
| All other clauses of the plan, the FK records and the US records | **carried verbatim** | Untouched by this pass |

## Deferred Questions

- **None.** `US-013` delegates the remaining agent-decidable items and this pass settles every one of them; the reading is closed by the gate, and a question the gate absorbs is not deferred — it is answered.
- **The consequence is not deferred either**: it is disclosed on the plan surface and lands in `OI:` entry 1's statement, which the user must stamp. That is the route by which it reaches them.

## Questions Asked

**No questions were asked**, and the reason is the closure gate's own output (`Question required: no`), not a judgement that the session had asked enough. The escalation-floor test above is the citation: no clause fits, so the reading is not a user decision, and the consequence is a disclosure rather than a question.

**No `FK-` and no `US-` ID was reserved, filled, superseded or reused by this pass.**

## Gates Found Unmet (reported, not cured)

**Count reported to the orchestrator: `gates found unmet 3`.**

1. **`OI:` entries 1–3 still read `pending — user only`.** Only the user may stamp them. They are now **presentable**: the shape is settled, so entries 2–3 are no longer contingent. **Not cured** — this is the payload half of the invariant alignment pass, and it is what `invite` presents.
2. **`plan-summary.md` §Approved Decisions still says the payload cluster "is not approved" and awaits `decide`.** This pass updates it to "awaits the user's `OI:` stamps only".
3. **Final approval is still absent** and the implementability mark stays off until the user stamps entries 1–3. What changed this pass is that **nothing is `OPEN` for `decide` any more**: the reading is closed, the shape is settled, and the only thing between the plan and `forward` is the user's own stamp.

## Validation Performed

| Claim | How verified | Result |
|---|---|---|
| Session identity / handle suffix | `session-identity.json` vs handle | matches (`9bf7afff5573` = first 12 hex) |
| No parent coordination contract | `session-identity.json` has no `parent_session_uid` | confirmed |
| The three write sites are the whole set and are live | `grep -n "write_text\|open(" buster_ripper/dump.py` | confirmed — exactly `:82`, `:95`, `:105` |
| A turn today costs exactly two write-bearing operations | read of `dump.py:80-106` | confirmed — one `write_text` (`:82` or `:95`) plus one `index.open("a")` + `write` (`:105-106`) |
| The documented layout and its reading workflow are where the plan says | read of `README.md` with line numbers | confirmed — layout block `:146-152`, "Reading the index" `:154-159`, trailing note `:161-162` |
| `README.md:153` is blank, so `:153-162` and `:146-162` resolve to the same section | read of `README.md:146-162` | confirmed — 018's citation nit is real and harmless; recorded, not filed |
| No `--dump-format` flag exists; `--dump-dir` is the only dump CLI option | `grep -n "dump" buster_ripper/main.py` | confirmed — `:25` only |
| No dump format/encoding global exists | `grep -n "DUMP\|KV_POLL" buster_ripper/config.py` | confirmed — `DUMP_DIR:16`, `KV_POLL_INTERVAL:44` only |
| `/Volumes/data` is unmounted, so M3 is unmeasurable here and the byte figures are carried, not re-derived | `ls -d /Volumes/data` | confirmed — absent |
| No invariant registry exists, so every `Registry: none` destination is accurate, not skipped | `ls -d invariants .invariants` | confirmed — both absent |
| No new `US-` reserved | the dispatch's `Reason` is verbatim `US-012`'s own `User words` | confirmed — reserving would duplicate a canonical record |
| No `FK-` ID reserved or authored; `FK-002`'s block edited only in `State`/`Superseded-by` and its note | read-back of `decisions.md#fk-002` | confirmed — `User words`, `Presented`, `Resolution`, `Factual claims`, `Impact` unchanged |
| Both context files within caps | `wc -l` after this pass | **`plan-summary.md` = 400/400, `handoff-context.md` = 188/200** — both compliant. `plan-summary.md` started at 385 and the finalization added net 15 lines; the history 019 makes obsolete (`T002`'s and `T005`'s now-redundant `SUPERSEDED`/`RESCOPED` warning bullets, folded into their dropped-clause lines; item 8's stale `--dump-format` `OPEN` note) was rewritten in place per supersession-by-rewrite, with the old wording recorded in this report's §Dropped-Clause Sweep |

## Next Step

1. **`invite`** — present the replan's landscape **and** its first question in **ONE** pass (round 15's process finding). The landscape must include the **disclosed consequence** — under S-1 the write reduction is not delivered by default — and the payload half of the invariant alignment pass (`OI:` entries 1–3), which is the ratification point where the user owns the reversal. Do **not** re-present `OI:` entry 1's statement as it stood in round 14, do not re-ask what `US-012` refused or `US-013` delegated, and do not present `FK-002` as open.
2. **`interview`** — record the stamps, then re-run the invariant alignment pass as a whole. Its TLS half is already closed; its payload half is what remains.
3. `roadmap.md` is seeded this pass with the deferred switch and the two `FK-007` entries above the MVP floor.
