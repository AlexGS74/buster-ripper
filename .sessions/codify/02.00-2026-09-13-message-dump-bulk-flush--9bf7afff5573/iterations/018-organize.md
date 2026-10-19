# 018 — Organize (REROUTE: payload/dump-layout cluster — `US-012`)

## Orchestrator Prompt (verbatim)

> Execute the organize-interactive skill for this session. Arguments: $ARGUMENTS
>
> Session ID: 02.00-2026-09-13-message-dump-bulk-flush--9bf7afff5573
> Mode: REROUTE
> Cluster: dump payload/layout cluster
> Reason (user words, verbatim): "Right, but deleting something before we implemented something better is premature." / "Old behavior should stay unaffected until we prove a new one is better and it's worth switching to."
> The dump-layout override is refused and `FK-002`'s resolution is under revision: the old layout must not be removed in the same increment that adds the new format. Scoped to this cluster only; nothing else in the plan reopens.

(`$ARGUMENTS` was empty. The dispatch carried the `organize-interactive`, `codify-agents`, `critical-thinker`, `codify-user-prompt` skill bodies.)

## Session Type / Mode

- **Continuing session**; **REROUTE** (organize → decide → invite → interview), scoped to the **payload/dump-layout cluster only**.
- Session identity re-verified against `session-identity.json`: handle `…--9bf7afff5573`, UID `9bf7afff-5573-402a-b102-827423a1dded` (handle suffix = first 12 hex), display id `02.00-2026-09-13-message-dump-bulk-flush`. **No `parent_session_uid`** → the parent coordination contract does not apply.
- **No `## Interview Alignment` block has ever existed** in `handoff-context.md`, so there is no prior approval to revoke. Confirmed by reading the file in full.
- **TLS cluster is out of scope** (round 15 closed it). No Anchor Map row, task, goal, VP or `OI:`/`PI:` entry belonging to it is touched by this pass.

## Context Conflict Detected

| Source | Says | Resolution |
|---|---|---|
| Priority 1 — `decisions.md#us-012` (2026-09-16, round 14) | the documented per-turn layout **must not be deleted by the change that introduces the collapsed artifact**; old behavior stays unaffected until the new one is proven better and judged worth switching to | **Current authority.** Followed. |
| Priority 1 — `decisions.md#fk-002` `Resolution.Normative choice` (2026-09-13, round 2) | "The documented per-turn layout is **NOT** preserved" | **Superseded by the user's own later words.** The record is intact and carries a `### Revision under review` pointer; its ordering is not applied. `US-012` wins on recency and on the record's own pointer. |
| Priority 5 — the dispatch's `Reason:` field | the two messages, verbatim | **Consistent with `US-012` and adds no gloss.** The scope clause ("this cluster only; nothing else reopens") matches `US-012`'s own non-extension clause. |

## Interview Report Absorption

`iterations/009-interview.md` §Round 14 and §Round 15 are the primary collect input; `decisions.md#us-012` and `#us-013` are the canonical records. Origin of each finding is marked **[interview]** (user-sourced), **[process]** (interview's own analysis), or **[prior]** (earlier analysis).

| Input | Origin | Disposition in this pass |
|---|---|---|
| `US-012`: the documented layout must not be deleted by the increment that adds the new format; implement → prove → then displace | **[interview]** | **Carried as the governing constraint.** Drives F-13, F-14, F-19, G-008 |
| `US-012`: it does **not** say whether the old layout stays the *default* or merely *available*; "prove it is better" is unmeasured | **[interview]** | **OPEN — restated as goal-shape rows and the §Ordering Shapes table.** `decide`'s |
| `US-012`: the principle is **not** extended to the TLS cluster | **[interview]** | Honored — the TLS cluster is untouched by this pass |
| Reroute item 1: three shapes (a)/(b)/(c) with their write-count consequences worked out | **[interview]** | Worked out in §Ordering Shapes; a fourth shape `2N + F` is **excluded** by arithmetic (F-13) |
| Reroute item 2: `2N + F` is worse than today | **[interview]** | **Confirmed and generalized as F-13**, a derived constraint, not a user choice |
| Reroute item 3: no decision point at which a demonstration switches the default | **[interview]** | **Confirmed as F-18**; §Measures table gives the four candidate measures |
| Reroute item 4: successor FK and `FK-002`'s `State` are `decide`'s | **[interview]** | Not authored here; no ID allocated or reserved by this pass |
| Reroute item 5: re-read `US-007`/`US-008` rather than assume them settled for the encoding dial | **[interview]** | Re-read in §Invariant Reconciliation; `US-007` is continuous with `US-012` |
| `US-013`: the agents decide the remaining agent-decidable items; the escalation floor is **not** repealed | **[interview]** | Applied — the default question is treated as user-bounded, not agent-decidable (F-19) |
| "The invariant alignment pass keeps returning one item at a time" — `invite` must present landscape + first question in **one** pass | **[process]** | Carried into §Deferred Questions and §Next Step |
| The 12 findings `F-1`…`F-12`, the research increment, the capacity finding | **[prior]** | Carried by citation; not re-derived |

## User Statement Classification

No new `US-` was reserved, and none should be. The dispatch's `Reason` field quotes the user verbatim, and those exact words are **already** canonically recorded as `US-012` (`User words` field, both messages, in order). Re-reserving would duplicate an existing canonical US. The dispatch carries **no `Fork:` field**, so nothing is an FK answer.

Classification of `US-012`'s clauses under the User Authority Boundary: **normative** throughout — "is premature", "should stay unaffected", "worth switching to" are sequencing and value judgments, not falsifiable propositions. `US-012`'s own `Factual claims: none` is therefore correct, and this pass manufactures no claim row for it. The premise behind the statement ("the plan deletes the old write path in the same increment that adds the new format") was verified by interview in round 14 and re-verified in §Anchor Verification Evidence below: `T002`'s `REMOVED` list names `dump.py:81-82`, `:94-95`, `:104-106` and all three sites exist. The premise holds.

## Goal Shape Analysis

| Goal candidate | Source wording | Observation surface | Met when | Failure witness | Boundary / non-goals | Alternate readings | Authority |
|---|---|---|---|---|---|---|---|
| **G-001** — captured messages are held in memory and written in bulk (restated) | `FK-002` *"our bulks would be single write"* + `US-012` | write-bearing file operations issued against `DUMP_DIR` for a session of N turns; on-disk state of `DUMP_DIR/<sid>/` | **OPEN.** Under **R1** (documented format is the default): for N turns with a **collapsed format selected**, total write operations `= F` versus `2N`, reduction `1 − F/(2N)` ∈ [50%, 100%). Under **R2** (collapsed is the default): the same clause with no configuration required | **A**: a flush writes one artifact per pending turn → the session total is `N + F`; buffering changed only *when* writes happen. **B**: the artifact is rewritten per flush → count falls, bytes grow `O(N²)`. **C (new)**: the documented writes and the collapsed artifact are **both** produced → `2N + F`, worse than today | `/v1/messages` responses unchanged; durability waived (`US-001`); the reduction is a count and witness B is why the count alone is not the goal | R1 vs R2 classify the *same implementation* oppositely → **OPEN** | user (`FK-002` normative) + **OPEN** reading |
| **G-002** — buffered dump reaches disk on SIGINT/SIGTERM | original charter | `DUMP_DIR/<sid>/` after SIGINT/SIGTERM | **Scope-qualified**: holds **when a collapsed format is selected**. Under R1's default path nothing is buffered, so the clause is vacuous rather than observable | Process exits cleanly but the turns a collapsed-format session buffered are missing | SIGKILL out of scope; no custom signal handler | none material | code-grounded |
| **G-004** — periodic flush bounds staleness | original charter | the `DUMP_DIR/<sid>/` artifact's contents/mtime **while the process runs** | **Scope-qualified**: holds **when a collapsed format is selected**; the timer no-ops otherwise | The timer runs and the artifact never gains the turn | no durability guarantee; interval value is a `config` default | none material | code-grounded |
| **G-005** — volume-triggered flush | `FK-001` | `DUMP_DIR/<sid>/` contents with the timer disabled | **Scope-qualified**: holds **when a collapsed format is selected** | Turns accumulate past the threshold with nothing on disk until shutdown | no memory cap promised; basis is rendered-text bytes | none material | code-grounded |
| **G-006** — the dump stays readable (function, not layout) | `README.md:134-135`, `README.md:146-162`, `FK-002` option text | a reader inspecting only `DUMP_DIR/<sid>/` | **Per shipped format.** change-focused: clause holds as written. self-contained: reader-performed comparison. **New**: the documented format is a third shipped format and its workflow is `README.md:154-159` **unchanged** | The artifact is well-formed and a reader holding only the directory cannot find the changed turn | not a requirement that a viewer ship; not a requirement that the machine record ship | none — `F-11` resolved by `FK-007` | code-grounded |
| **G-008 (NEW)** — the documented dump layout is unaffected by this change, and its displacement is gated on a demonstration the user judges worth switching to | `US-012` verbatim: *"Old behavior should stay unaffected until we prove a new one is better and it's worth switching to."* | (a) a default run's `DUMP_DIR/<sid>/` contents and the `README.md:154-159` workflow; (b) the comparison evidence produced at `MS-001`; (c) the plan's task list | **(a)** with **no format configured** (or the documented format selected), N turns produce exactly `turn_000.json`, `turn_{N:03d}.diff`, `index.txt` with today's content, at `2N` write operations — **under R1**. Under R2: the documented format is selectable and produces exactly that **when selected**. **(b)** a comparison of the collapsed path against today's path exists at a named boundary, naming its measure | **W1**: the code keeps the documented layout selectable, all tests pass, the README still documents it — and a default run writes the collapsed artifact, so the old behavior was switched before any proof. **W2**: the documented layout is retained but its files or the `index.txt` line format drift, so "unaffected" is false while every new-format test is green. **W3**: the collapsed format ships and no comparison against today's path exists anywhere — "better" is asserted, never measured | Only the dump layout; **not** the TLS cluster, **not** the `--upstream` default, **not** the flush-trigger values (`US-012` non-extension) | (a) "unaffected" = the no-configuration experience is byte-identical (→ documented format stays default); (b) "unaffected" = the documented format is not deleted and works when selected (→ collapsed may be default). **They classify the same implementation oppositely → OPEN** | user (`US-012`) for the requirement; **OPEN** for the reading |
| **Constraint (NEW)** — at most one write path runs per turn | derived: `G-001` + `FK-002`'s *"one bulk flush is ONE write"* + `US-012` | write-bearing file operations against `DUMP_DIR` under any configuration | For any configuration, a session of N turns performs **either** `2N` **or** `F` write-bearing operations — never `2N + F` | The documented writes and the collapsed artifact are both produced for the same turn: `2N + F`, worse than today, while every test passes | not a statement about *which* path is default | **none — arithmetic.** `2N + F > 2N` for all `F ≥ 1` | code-grounded (arithmetic + two normative sources) |

**OPEN rows to carry into `handoff-context.md`**: `G-001`'s `Met when` (R1 vs R2), `G-008`'s reading (R1 vs R2), and `G-008`(b)'s measure (M1/M2/M3/M4 — see §Measures). `G-008` is the session's new headline goal: it is the user's own requirement, and it currently has **no** plan element, **no** task and **no** VP.

## Ordering Shapes and Their Arithmetic

`US-012` requires an ordering, not a format. Every shape below is a way to satisfy it; the arithmetic is the discriminator. Notation: N turns, `F` flushes, today's cost `2N` (one file write + one `index.txt` append per turn — verified `dump.py:82`, `:95`, `:105`).

| Shape | Default (no config) | Documented selected | Collapsed selected | Writes for N turns | `G-001` met? | Old behavior unaffected? |
|---|---|---|---|---|---|---|
| **today** (baseline) | documented | `2N` | — | `2N` | n/a | n/a |
| **S-1** (R1) | **documented** | `2N` | `F` | `2N` or `F` | **only when a collapsed format is selected** | **yes — under both readings** |
| **S-2** (R2) | collapsed | `2N` | `F` | `F` by default | yes | **no under reading (a)**; yes under reading (b) |
| **S-3** (built, no switch) | documented | `2N` | unreachable | `2N` | never — the reduction is unreachable in production | yes |
| **S-X** (both written) | — | `2N + F` | `2N + F` | `2N + F` | **no — worse than today** | formally yes; violates the derived constraint |

**F-13 — S-X is excluded, and it is the shape the current plan most nearly describes.** `US-012`'s clause 1 read alone ("old behavior stays unaffected") plus the plan's own addition of the collapsed artifact yields `2N + F` — **worse than today**, not better. This is the single most important structural consequence of the round-14 statement, and it is *derived*, not chosen: `2N + F > 2N` for every `F ≥ 1`, and `G-001` forbids a session that writes more than today. The consequence is that the two writers must be **mutually exclusive per turn**, selected by configuration. The plan's `T002` `REMOVED` clauses (the three write sites) therefore cannot stand as written: the sites stay in the code as one branch of the selector. **The removal is not what `US-012` refuses — the *simultaneity* is what the arithmetic refuses.**

**F-14 — `OI:buster_ripper.dump/dump_turn/output_layout` is a `refine`, not an `override`.** The invariant's statement is "session dir holds `turn_000.json`, `turn_{N:03d}.diff`, `index.txt`". Under S-1 or S-2 that statement is **true whenever the documented format is selected** and false only for a collapsed selection — so the property survives, scoped, exactly like `sequential_diff_chain`. An `override` requires the old invariant to be *displaced*; `US-012` forbids displacing it. The register's entry 1 `Decision` cell currently reads `override` and is therefore wrong under the current authority. Corrected in `plan-summary.md`; the replacement *statement* remains `decide`'s to finalize.

**F-15 — `G-002`/`G-004`/`G-005`, `VP-T001-writes-deferred`, `VP-T002-volume-flush`, `VP-T003-interval-flush` and `VP-T003-shutdown-flush` are collapsed-path-only under S-1.** Under S-1's default path there is no buffer and no flush: a turn is written at append time exactly as today. `VP-T001-writes-deferred` asserts "no artifact containing those turns exists on disk" — on the default path the per-turn files *do* exist, so the VP **falsifies on the default configuration**. Every one of these must name the selected format in its scope. This is a correction, not a new goal.

**F-16 — `VP-T002-default-encoding` is held, not finalizable.** It asserts that with no `--dump-format` the artifact is the change-focused *collapsed* encoding. Under S-1 that is false (the default is the documented layout); under S-2 it is true. The handle stays stable; the row is annotated **HELD — fork-dependent** in `plan-summary.md` with both candidate clauses. Authoring it now would hard-code one reading.

**F-17 — "old behavior unaffected" has no witness in the plan.** `G-008`(a) is unfalsifiable today: no VP asserts that the documented path is byte-identical. Required: `VP-T002-documented-format-preserved` — with no format configured, N turns produce `turn_000.json`, `turn_{N:03d}.diff`, `index.txt` with today's content and `2N` write-bearing operations. Without it the plan can complete every task and still be blind to `G-008`'s W1/W2.

**F-18 — "prove it is better" has no measure and no evaluation point.** The plan has `VP-T002-write-count-reduced` (instrumented `F` vs `2N`) but no *decision point* at which the comparison authorizes a switch, and no named measure. §Measures gives four; they disagree on strength.

### Measures — which axis "better" is proved on

| Measure | Today | Collapsed (change-focused) | Deterministic in-repo? | Source |
|---|---|---|---|---|
| **M1** write-bearing operations | `2N` | `F` | **yes** — instrumented, `F` injected | `G-001`; `VP-T002-write-count-reduced` |
| **M2** files created in `DUMP_DIR/<sid>/` | `N + 1` | `1` | **yes** — directory listing | `007` F7 ("one inode + dir entry per turn"); `RB-003` |
| **M3** physical bytes written | ≈ payload × WAF (WAF 2–32×; ext4 4×) | ≈ payload × ~1 | **no** — needs device instrumentation; `/Volumes/data` is **unmounted** (verified this pass) | `007` F1/F2/F4/F7 |
| **M4** logical bytes | ≈ payload | ≈ payload **+ ≈1.5%** (turn 0 keeps body **and** rendering, `F-6`) | yes — file sizes | `F-6`; carried figures, not re-derived |

**M1 and M2 move decisively; M4 does not move (it is slightly worse); M3 is the physical axis and cannot be measured here.** `007`'s own verdict is that "the count is secondary to bytes" — so a demonstration on M1 alone is a *proxy* proof, and the plan must say so. This is the honest form of `US-012`'s clause 2, and it is a plan element, not a VP footnote.

**F-19 — the default's identity is `OPEN`.** Reading (a): "unaffected" = the no-configuration experience is byte-identical → the documented format stays the default. Reading (b): "unaffected" = the documented format is not deleted and works when selected → the collapsed format may be the default. The two classify S-1 oppositely. **Closure note for `decide`:** S-2 performs the *switch* that `US-012`'s second clause gates ("until we … prove a new one is better and it's worth switching to"), so S-2 violates the gating clause under *either* reading of clause 1; S-1 is compatible with both. `decide` owns whether that is `CLOSURE: absorbed` (no fork) or a fork on the reading. **Either way one consequence must reach the user: under S-1 the write reduction is not delivered by default.** That is a user-visible product outcome and it is the price of `US-012`.

**F-20 — under S-1 the session's headline benefit is opt-in.** A user who configures nothing gets no reduction and no flush behavior; the buffering/flush machinery is inactive on the default path. This is not a defect — it is the direct consequence of `US-012` — but it must be stated in user terms before the user ratifies the plan, because the session's Topic ("bulk-flush dump, flush on Ctrl-C/SIGTERM") is otherwise read as delivered.

**F-21 — `T001`'s buffer append must be conditioned on the selected format.** As written ("`dump_turn` appends the turn's encoded record to `state.pending`") the buffer is populated on the documented path too; a later flush then produces `2N + F` (F-13). The append is a collapsed-path branch, not unconditional.

**F-22 — `T002`'s capacity extraction grows, it does not shrink.** `dump_turn` (`dump.py:61-109`) is **34 executable statements** — L:34 is a hard LNCP violation (>30), N:1, C:≈6 (two `if`s plus two generator conditionals), P:1. Under S-1/S-2 it gains a format branch **and must retain** the documented writer, so the extraction must now yield at least: state lookup/creation; the format dispatch; the documented writer (the three write sites + summary); the change-focused record; the self-contained record; the artifact path builder. The earlier ~150–170-line projection for `dump.py` was computed for a *deleting* `T002`; the retained-writer shape is larger. `dump.py` stays Green by file line count (<300) either way.

**F-23 — `T005`'s `REMOVED` clauses are dropped and replaced by an addition.** Under F-13's shape `README.md:146-152` (the layout block) and `:154-159` ("Reading the index") remain **current and correct** — the documented format still behaves that way. `T005` becomes: **add** the collapsed formats, the selector and each format's reading workflow, and **note** that a session dir may hold both shapes after a format change. Nothing in the README is removed.

**F-24 — the selector's value set now spans two axes, and the invariant family is named for one.** `config.DUMP_FORMAT`/`--dump-format` was scoped as a *record encoding* selector (`RB-004`, `PI:T002/artifact_encoding/*`). Under every live shape its value set must also express *whether the documented layout or a collapsed artifact is produced* — a **layout** axis, not an encoding. Two coherent designs exist: (i) one closed set of three values (`documented`, `change-focused`, `self-contained`) with one global and one flag; (ii) a layout switch plus the two-value encoding set, with two globals. The plan's `PI:T002/artifact_encoding/*` names would be misnomers under (i). `decide`'s to settle; recorded so the choice is not made silently by whichever task lands first.

## Changes Since Iterations 013 / 016

- **NEW**: the ordering constraint (`US-012`) and its four shapes; F-13 (mutual exclusivity, derived); F-14 (`output_layout` override → refine); F-15 (goals/VPs are collapsed-path-only under S-1); F-16 (`VP-T002-default-encoding` held); F-17 (the missing preservation witness); F-18 (no measure / no evaluation point); F-19 (the reading is OPEN); F-20 (the opt-in consequence); F-21 (`T001` conditioning); F-22 (capacity extraction grows); F-23 (`T005` addition-only); F-24 (two axes in one selector); G-008.
- **UPDATED**: `G-001`'s `Met when` is now per-format and fork-dependent. `G-002`/`G-004`/`G-005` are scope-qualified to a collapsed format. `B-001`'s disposition changes from "the layout is *replaced*" to "the documented layout is **retained** as a selectable format; the collapsed artifact is added".
- **DEPRECATED**: `OI:` entry 1's `override` direction (F-14). `T002`'s and `T005`'s `REMOVED` clauses as written (round 14 already refused them; this pass replaces them with concrete replacements).
- **UNCHANGED**: the TLS cluster in full; `T001`'s buffer design apart from F-21; `T003`'s trigger machinery apart from F-15; the 12 findings `F-1`…`F-12`; `FK-001`/`FK-007`; `US-007`/`US-008`; every dump `VP-*` handle; `MS-001`…`MS-004`.

## Anchor Map Verification and Refinement

Every row in the payload cluster's contradiction surface was re-read against source this pass (no row is carried on collect's draft alone). Rows verified:

| Row | Binding checked | Result |
|---|---|---|
| dump entry point | `dump.py:61` `def dump_turn(body: dict) -> None:` | resolves — the referent exists and owns the record construction |
| per-session dump state / dump write buffer | `dump.py:16-24` `@dataclasses.dataclass class SessionState` with `turn`, `prev_text`, `session_dir`; `dump.py:24` `_sessions: dict[str, SessionState]` | resolves — one class, one module-level dict; the two rows bind the **same** symbol (pre-existing, not a defect) |
| dump summary line | `dump.py:84` (the `FULL` form) and `dump.py:98-101` (the `+added/-removed` form) | resolves — exactly two construction sites |
| dump dir config | `config.py:16` `DUMP_DIR: Optional[Path] = None` | resolves |
| shutdown flush seam | `server.py:58` `async def on_cleanup(app)` | resolves |
| signal handling | `web_runner.py:283-284` `loop.add_signal_handler(SIGINT/SIGTERM, _raise_graceful_exit)` → `GracefulExit` (`:35-40`) → caught `web.py:511` → `web.py:425` `await runner.cleanup()` | resolves — native path, no custom handler needed |
| dump caller | `routes/messages.py:55-59` `if config.DUMP_DIR is not None: try: dump_turn(body) except Exception:` | resolves — the only call site |
| flush trigger parameters | `config.py:44` `KV_POLL_INTERVAL: int = 10` (the shape peer) | resolves |
| dump encoding selector | `config.py:87-96` `PROFILES` + `main.py:45-48` `--eval-profile` (the shape peer); `main.py:25-26` `--dump-dir` (the only dump CLI option) | resolves |

**No binding was corrected this pass** — every collect/organize binding still resolves to the referent named, so there is no headline correction. Two **Notes** corrections were made (both are scope, not binding):

1. `dump entry point` — its role is now the **format branch**: documented → today's three writes; collapsed → buffer append. Signature unchanged (`dump_turn(body) -> None`), so the `dump caller` row's `reuse` stance is unaffected.
2. `dump encoding selector` — its value set now spans a **layout** axis as well as an encoding axis (F-24); the default's identity is `OPEN` (F-19). Mode stays `NEW`; the axis question is `decide`'s.

### Anchor Verification Evidence

| Anchor / RB handle | Aspect | Proposition / required role | Supporting scope | Checked basis | Result |
|---|---|---|---|---|---|
| dump entry point | binding-exists | `dump_turn` is the function that builds a turn's record and performs its writes | callers: `routes/messages.py:55-59`; imports: `routes/messages.py:12` | read of `dump.py:61-109` | verified |
| dump entry point | scope-meaning | owns the format branch; retains the documented writer as a branch | plan scope §Task List `T002`; the three write sites | `grep -n "write_text\|open(" buster_ripper/dump.py` → exactly `:82`, `:95`, `:105` | verified |
| per-session dump state / dump write buffer | binding-exists | `SessionState` is the session-keyed state; the buffer is a field on it, not a new type | `dump.py:24` `_sessions`; `dump.py:70-74` | read of `dump.py:16-24` | verified |
| dump summary line | binding-exists | two summary-string construction sites exist at `:84` and `:98-101` | the documented writer's index line | read of `dump.py:84`, `:98-101`, `:104-106` | verified |
| dump dir config | binding-exists | `config.DUMP_DIR` is the mode switch `dump_turn` guards | `dump.py:63`, `dump.py:71`; `main.py:88-90` | read of `config.py:16`, `main.py:88-90` | verified |
| shutdown flush seam | binding-exists | `on_cleanup` is the app teardown hook | `server.py:80` `app.on_cleanup.append(on_cleanup)` | read of `server.py:58-70`, `:78-81` | verified |
| signal handling | binding-exists | aiohttp registers SIGINT/SIGTERM natively and runs cleanup hooks on graceful exit | installed aiohttp 3.13.3 source | `web_runner.py:283-284`, `:35-40`; `web.py:511`, `:425` | verified |
| dump caller | binding-exists | `handle_messages` is the only `dump_turn` call site; signature unchanged | package-wide grep for `dump_turn` | `routes/messages.py:12`, `:55-59` | verified |
| bulk flush / flush timer / flush trigger parameters | reuse-absence | no symbol owns the periodic-flush role or its parameter globals | `RB-001`, `RB-002` searches; no new candidate this pass | carried by citation from organize 002/005 — supporting scope unchanged | verified (by citation) |
| collapsed dump artifact | reuse-absence | no symbol owns the per-`(sid, encoding)` append-only artifact role | `RB-003` write- and read-side searches | carried by citation from organize 013; re-checked this pass by the widened search below | verified |
| dump encoding selector | reuse-absence | no symbol owns the dump output-format selector role, **widened to the layout axis** | `RB-004` name search re-run this pass | see §Reuse Evidence | verified |
| TLS cluster rows | binding-exists / scope-meaning | unchanged; out of this replan's scope | `RB-005`; organize 016 + decide 017 | carried by citation; **no row touched** | verified (by citation) |

No `candidates:` token survives anywhere in the map; no `Mode` cell was changed.

## Reuse Evidence

**`RB-004` re-checked this pass with a widened role.** The role is no longer "record encoding selector" alone: under every live shape the selector also chooses **whether the documented per-turn layout or a collapsed artifact is produced** (F-24). Per the Freshness rule, a widened required role invalidates the affected search even when every previously found file is unchanged, so the search was re-run rather than carried.

| Evidence | NEW anchor | Name search | Role/peer search | Native capability | Conclusion |
|---|---|---|---|---|---|
| **RB-004** (re-run 018) | dump output-format selector (`config.py:DUMP_FORMAT`, `DUMP_FORMATS`; `main.py:--dump-format`) | `grep -rniE "dump_format\|dumpformat\|dump_mode\|dumpmode\|output_format\|output_layout\|layout\|per-turn\|per_turn\|collapsed\|artifact"` over `buster_ripper/` + `tests/` (`*.py`, `*.sh`, `*.toml`) → **one** hit, prose only: `main.py:24` `help="Log every normalization applied and per-turn token/latency stats."` Zero selector matches. `grep -rn '"--' buster_ripper/main.py` → 21 options, of which `--dump-dir` (`:25-26`) is the only dump option and **no** `--dump-format`/`--dump-mode` exists. Re-verified: `buster_ripper_legacy.py` is the tracked pre-refactor monolith referenced by nothing under `buster_ripper/`, excluded from scope | No registries or Protocol/ABC contracts exist in this repository (re-confirmed: no `_protocols.py`, no ABC). Shape peers, unchanged: `config.PROFILES` (`config.py:87-96`) + `--eval-profile` (`main.py:45-48`) — a config-declared named value set as `click.Choice` with `default=`; `--backend-type` (`main.py:60`) a second instance; the six `is_flag=True` options (`main.py:19,21,23,38,42,51`) a different shape. `config.DUMP_DIR` (`:16`) is the mode switch this parallels. **No existing global selects an output layout or a record encoding** | `click.Choice` is native and already used at `main.py:46` and `:60`; the value set is a module-level tuple; the branch is a string comparison. Python has no built-in encoding- or layout-registry facility and none is warranted for two-to-three values. Deps stay `aiohttp` + `click` (`pyproject.toml:5-8`); `json`/`difflib` already imported (`dump.py:4-5`) | **No existing symbol owns the widened role.** The NEW anchor stands, widened: it selects the dump's output *shape* (documented layout vs collapsed artifact) and, for a collapsed artifact, the record encoding. The nearest reusable thing remains the `config.PROFILES` + `--eval-profile` **shape**, not code |
| RB-001 / RB-002 / RB-003 | carried | — | — | — | Unchanged this pass; the anchors they authorize (bulk flush, flush timer + parameters, collapsed artifact) all survive every live shape. Carried by citation to organize 002/005/013 |

## Confusability Findings

**No new finding is recorded this pass, and that is the deliberate answer, not an omission.** Per the reliability rule (observed confusion only), each candidate was tested against "did an agent actually misread this artifact this session":

| Candidate | Verdict |
|---|---|
| `README.md:153-162` vs `README.md:146-162` (two different citations of the reading workflow inside this session's own files; `:153` is a blank line, the workflow block is `:154-159`) | **Not recorded.** I noticed the off-by-one; I was not misled by it, and neither citation changes a decision. Noted in §Validation as a citation nit |
| `plan-summary.md` register entry 1 says `override` while §Invariant Reconciliation calls `output_layout` the override target, under an authority that forbids displacement | **Not a codebase/registry confusability finding** — it is a plan defect, recorded as **F-14** and corrected in the plan |
| `dump.py:20` `session_dir: Path = None  # type: ignore[assignment]` — an annotation that disagrees with its default | **Not recorded.** No agent misread it this session, and the session's own rule ("prefer `Optional` over Null-like") makes it a latent style defect, not an observed confusion. Flagged here so `T001`'s author sees it: `SessionState` gains fields in `T001`, and this field is why `state.session_dir` is treated as non-optional in practice |
| `dump.py` module docstring ("saves full body (turn 0) or diff (turn N)") vs the planned collapsed artifact | **Not recorded** — it is accurate today; `T002` already carries its update |

`quality-backlog.md` is therefore **unchanged at 3 findings**.

## Invariant Reconciliation Prep

**No invariant registry exists** (re-verified this pass: `ls -d invariants .invariants` → both absent), so every `EI:` is a session-local draft and every `Registry:` destination reading `none` is accurate rather than skipped.

### Existing invariants in the contradiction scope

- `EI:buster_ripper.dump/dump_turn/output_layout` — session dir holds `turn_000.json`, `turn_{N:03d}.diff`, `index.txt` | Status: **intended** | **disposition changed: `override` → `refine`** (F-14). The property survives for the documented format's selection; only its *unconditionality* changes.
- `EI:buster_ripper.dump/dump_turn/sequential_diff_chain` — turn N diffs turn N−1's rendered text; `prev_text` advances only when a turn is written | **intended** | **refine** — unchanged direction; under S-1 the "when a turn is written" clause must be read per selected format (the documented path still writes at append time).
- `EI:buster_ripper.dump/dump_turn/turn_zero_is_full_body` — turn 0 is written as a full JSON body | **intended** | **refine** — unchanged; the documented path already satisfies it verbatim, so under S-1 it is *preserved* rather than refined for that path.
- `EI:buster_ripper.routes.messages/handle_messages/dump_failure_isolated` | **preserve** — unchanged; `dump_turn`'s signature is unchanged, so the swallow-all contract covers the format branch for free.
- `EI:buster_ripper.session/session_id/derived_after_normalize` | **preserve** — untouched.
- `EI:buster_ripper.server/on_cleanup/graceful_shutdown_hook` | **preserve** — untouched; `on_cleanup` gains a flush call inside the same wrapped contract.

### Likely planned invariant themes

- **The new format is selectable and does not displace the documented one** (new; from `US-012`). This is the theme `US-012` adds and no existing invariant covers.
- **At most one write path runs per turn** (new; F-13's derived constraint). Protects `G-001` against the `2N + F` shape.
- `PI:T002/artifact_encoding/*` (four entries) — all four are **contingent** and remain statements about the collapsed artifact. `encoding_is_per_artifact` and `encoding_does_not_change_write_count` survive every live shape; `default_encoding_is_change_focused` **collides with the default question** (F-19) and must not be confirmed until it is settled; `encoding_set_is_closed` survives but its name must change if F-24 lands on design (i).

### Suspected contradiction pairs

- `EI:.../output_layout` ↔ the planned "the new format is selectable" theme — **not a contradiction under the corrected direction**: the old property is scoped, not displaced. Under the *uncorrected* `override` direction they do contradict, which is exactly why the direction is wrong.
- `PI:.../default_encoding_is_change_focused` ↔ the `OPEN` default reading — if R1 lands, the default is the **documented layout**, and a `PI:` naming change-focused as the default is false on the day it is written. This is the invariant most exposed by the replan.
- `PI:.../encoding_does_not_change_write_count` ↔ `G-001`'s restatement — compatible; the write-count property is what makes `G-001` checkable per format.
- `G-006` ↔ `G-008` — a goal-level tension: the cheapest way to one write per flush is to write the least; the safest way to leave old behavior unaffected is to write both. **Arbitrated by F-13** (mutual exclusivity), and `G-006`'s reading clause survives because the documented format is itself a shipped format with an unchanged workflow.

### Out-of-scope invariants intentionally dropped

- The three TLS-cluster rows and `OI:buster_ripper.server/on_startup/shared_client_verifies_by_default` — finalized in round 15 and independent of the payload cluster.
- `EI:.../dump_failure_isolated`, `EI:.../derived_after_normalize`, `EI:.../graceful_shutdown_hook` are in scope only as **preserve** dispositions; nothing in this replan refines them.

### `US-007` / `US-008` re-read (reroute item 5)

- `US-007` verbatim: *"can we implment additional formats and keep teh current?"* / *"I'm for switchable option"*. The plan read this as being about the record **encoding**, while `FK-002` displaced the **layout**. Under `US-012` the gap closes: **"keep the current" is satisfied only if the current *layout* is retained**, and the natural realization is that the documented layout is one value of the switchable set. `US-007` is continuous with `US-012`, not superseded by it.
- `US-008` verbatim: *"Keep the existing one as one option and add something more human-visible or human-friendly."* `FK-007` resolved the human-facing axis to option 1 (self-contained encoding, no tool). Under S-1 the "one option" clause is satisfied **more** strongly than the plan assumed: the existing layout is now literally one option of the selector, not merely the default encoding.

## Capacity Gate

**TRIGGERED, and it grows (F-22).** Baseline re-measured this pass on the pre-edit worktree: `dump.py` 109 lines (Green); `dump_turn` (`:61-109`) **34 executable statements** — L:34 is a hard violation (>30); N:1; C:≈6 (two `if`s + two generator conditionals); P:1. The increment adds a **format branch** and **retains** the documented writer, so it adds statements, branches and (if the helper takes a parameter) parameters to an already-violated function. The smallest coherent extraction now spans **both** writers: state lookup/creation, the format dispatch, the documented writer (three write sites + summary), the change-focused record, the self-contained record, the artifact path builder. `T002` performs it **before** adding behavior, riding its own increment. File-level: `dump.py` is projected above the earlier ~150–170 estimate but stays Green (<300); `main.py` 130 / `config.py` 104 / `server.py` 112 unchanged in this increment. The `main.py:main` closure (`CLOSURE: absorbed`, 017) is unaffected — `T002`'s and `T003`'s option callbacks still add no parameter.

## QA Planning Context

### Validation Point Candidates (delta from decide 015)

| Validation point | Observable behavior | Observable after | Validate at | QA author pass | Scope reason | Evidence needed |
|---|---|---|---|---|---|---|
| **VP-T002-documented-format-preserved** *(NEW — F-17)* | With **no format configured** (or the documented format selected), N turns produce exactly `turn_000.json`, `turn_{N:03d}.diff` and `index.txt`, with today's content and line format, and `2N` write-bearing operations | T002 | MS-001 | QAP-MS001 | This is `G-008`(a)'s only witness. It is observable as soon as the format branch exists — before that the behavior is unchanged by definition and the VP cannot fail for a reason no other test fails for | `DUMP_DIR/<sid>/` listing + file contents + instrumented write count |
| **VP-T002-write-count-reduced** *(was VP-T002-write-count-reduced, unchanged handle; scope widened)* | For N turns flushed `F` times, instrumented write-bearing operations `= F` **with a collapsed format selected**, versus `2N` for the same N with the documented format selected; **both runs per shipped format**. The pair **is** the comparison `G-008`(b) requires | T002 | MS-001 | QAP-MS001 | The comparison only exists when both paths exist. `F` must be deterministic (inject the parameters, disable the timer). Unit: one write-bearing file operation on a dump artifact (`write_text`, or `open(...,"a"/"w")`+`write`); directory creation is not counted | instrumented write-call count per format + a known, injected `F` |
| **VP-T002-file-count-reduced** *(NEW — measure M2)* | With a collapsed format selected, `DUMP_DIR/<sid>/` holds **1** artifact for N turns; with the documented format selected it holds `N + 1` files. Deterministic from a directory listing | T002 | MS-001 | QAP-MS001 | The metadata-op reduction (`007` F7) has no witness otherwise, and it is the one measure that is both decisive and trivially observable. Paired with the write-count VP so neither mirrors the implementation alone | two directory listings for the same N |
| **VP-T002-default-encoding** *(HELD — fork-dependent, F-16)* | Under **R1**: with no `--dump-format`, the output is the **documented layout**. Under **R2**: with no `--dump-format`, the artifact is the change-focused collapsed encoding | T002 | MS-001 | QAP-MS001 | The clause inverts with the reading. Authoring either clause now hard-codes one reading; the handle stays stable and the row is annotated HELD | `DUMP_DIR/<sid>/` listing + artifact, with no `--dump-format` passed |

### Re-scoped validation points (F-15 — no handle changes)

| VP | Was | Now |
|---|---|---|
| `VP-T001-writes-deferred` | "no artifact containing those turns exists on disk" | "...with a **collapsed format selected** and the timer effectively disabled, no artifact containing those turns exists on disk; with the documented format selected the per-turn files exist at append time" |
| `VP-T002-collapsed-artifact` | "exactly one artifact for the configured encoding, no `turn_000.json`/`turn_NNN.diff`/`index.txt`" | Scoped to a **fresh session dir** *and* a **collapsed format selected**; the "no legacy files" clause is false in a dir where the documented format was used (identity is `(sid, format)`), which the fresh-dir scope already excludes |
| `VP-T002-volume-flush`, `VP-T003-interval-flush`, `VP-T003-shutdown-flush` | asserted on any session | Scoped to a **collapsed format selected** |
| `VP-T002-change-readable`, `VP-T002-turn0-available` | per shipped encoding | Per shipped **format**: the documented format's workflow is `README.md:154-159` **unchanged** |

### Behavior Dispositions (delta)

| Behavior | Disposition | Validation point |
|---|---|---|
| B-001 dump output layout and index format | **mapped (re-dispositioned again)** — the documented layout is **retained** as a selectable format, not replaced; the collapsed artifact is **added** | `VP-T002-documented-format-preserved`, `VP-T002-collapsed-artifact` |
| B-002 diff chain correctness | **mapped (re-dispositioned again)** — per-turn change information in arrival order, **per selected format** | `VP-T002-change-readable` |
| (new) the documented format's behavior is unchanged | **mapped** — `G-008`(a) | `VP-T002-documented-format-preserved` |
| (new) the collapsed format performs fewer write operations than the documented one | **mapped** — `G-008`(b) | `VP-T002-write-count-reduced`, `VP-T002-file-count-reduced` |
| all other rows | **unchanged** | as in `plan-summary.md` |

### Candidate Milestones

| Milestone | Constituent tasks | Complete when | Validation points | QA author pass |
|---|---|---|---|---|
| MS-001 | T001, T002, T005 | **Revised**: a collapsed-format session's turns reach disk only after a flush, in arrival order, with turn 0's body present, `F` writes with `F` controlled, the artifact's name stating its format — **and** a documented-format session of the same N produces today's files with `2N` writes and its README workflow unchanged; **and** the two counts are recorded side by side as the comparison `G-008`(b) requires | + `VP-T002-documented-format-preserved`, `VP-T002-file-count-reduced`; `VP-T002-default-encoding` held | QAP-MS001 |
| MS-002 / MS-003 / MS-004 | unchanged | unchanged | unchanged | unchanged |

### Notes for Interview Agent

- **The process finding from round 15 governs this pass.** `invite` presents the replan's landscape **and** its first question in **one** pass. Do not re-present `OI:` entry 1's replacement statement unchanged; do not re-present anything `US-012` refused or `US-013` delegated.
- **One question is the ceiling.** If `decide` lands on S-1 by closure and records the opt-in consequence as `INTERPRETED`, **no** question is owed and the pass goes straight to ratification. If `decide` judges the reading genuinely open, it authors **one** fork carrying the S-1/S-2 arithmetic, not a menu of sub-questions.
- **The consequence must be visible either way**: under S-1 the write reduction is **not** delivered by default.
- **`G-008` is the session's new headline goal** and currently has no task and no VP. The user must see it in the plan.

## Initial Task Identification (deltas — `decide` finalizes)

| Task | Delta | Why |
|---|---|---|
| **T001** — per-session pending-record buffer | **ADDED (conditioned)**: the append to `state.pending`/`state.pending_bytes` happens **only when a collapsed format is selected**; the documented path writes at append time and does not buffer (F-21) | An unconditional append plus a flush produces `2N + F` (F-13) |
| **T002** — collapsed artifact, format branch, single-write flush | **CHANGED**: the format selector chooses **which** writer runs; the documented writer is **retained as a branch** — its three write sites are **not removed**. The selector's value set, its default, and its axis naming are `decide`'s (F-19, F-24). The capacity extraction covers **both** writers (F-22) | F-13, F-14, F-24 |
| **T003** — wire the flush: volume trigger, timer, shutdown | **CHANGED (scope)**: the volume check and the flush machinery are active **only when a collapsed format is selected**; on the documented path there is nothing pending and the timer no-ops (F-15). Otherwise unchanged | `G-002`/`G-004`/`G-005` are collapsed-path-only under S-1 |
| **T004** — tests and coverage | **ADDED**: the documented-format preservation test (`VP-T002-documented-format-preserved`), the file-count test (`VP-T002-file-count-reduced`), and the write-count test run **per format** — the pair is the comparison evidence. `VP-T002-default-encoding`'s test is **held** pending the reading | F-16, F-17, F-18 |
| **T005** — rewrite the dump section of the README | **CHANGED to addition-only**: nothing is removed. The documented layout block (`:146-152`) and "Reading the index" (`:154-159`) stay; the collapsed formats, the selector, each format's reading workflow, `F-8`'s limitation and the pre-existing-files note are **added** (F-23) | `US-012` forbids deleting the documented layout; the README documents it |
| **T006** — upstream TLS verification opt-out | **UNCHANGED** | The TLS cluster is independent and finalized |
| *(no new task)* | The comparison evidence is `QAP-MS001`'s output, not a task. The **switch** (flipping the default and/or deleting the documented path) is a **roadmap entry gated on the user's judgment of that evidence** (`US-012`: *"worth switching to"*) — `decide` owns the entry | `US-012`'s `Impact.Roadmap` is `pending` and names `decide` as owner |

## Dropped-Clause Sweep

**Triggered** — this pass wholesale replaces `OI:` entry 1's definition (its decision direction, its replacement statement's scope, and the clauses that depend on displacement) and `G-001`'s `Met when`. One hop only. Predecessor clauses enumerated from the artifact being replaced and from `FK-002`'s lineage:

| Clause | Disposition | Why / where it now lives |
|---|---|---|
| `OI:` entry 1 — "**override**" | **dropped — deliberate** | F-14: the property survives, scoped; the correct direction is `refine`. An override requires displacement, which `US-012` forbids |
| `OI:` entry 1 — "the session dir holds one collapsed per-session artifact" | **restored — scoped** | True when a collapsed format is selected; the documented format's selection holds the legacy three files |
| `OI:` entry 1 — "Its filename states its encoding: `DUMP_DIR/<sid>/<encoding>.log`" | **carried** | Unchanged for the collapsed formats; the documented format's names are `turn_000.json`/`turn_{N:03d}.diff`/`index.txt` |
| `OI:` entry 1 — "One encoding per artifact, chosen once per process" | **carried** | Unchanged for the collapsed formats |
| `OI:` entry 1 — "the default is `change-focused`" | **dropped — deliberate** | F-16/F-19: the default's identity is now the open reading, and the *default format* question subsumes it. `VP-T002-default-encoding` is held |
| `OI:` entry 1 — the change-focused record content (summary line; turn 0 body **and** rendering; per-turn unified diff) | **carried** | `plan-summary.md` §Approved Override Register entry 1, unchanged; `T002`'s `ADDED` list |
| `OI:` entry 1 — the self-contained record content | **carried** | Same |
| `OI:` entry 1 — "Turn 0's 'full body' means **both** the raw normalized JSON body and its rendering (`F-6`)" | **carried** | Same |
| `T002`'s three `REMOVED` clauses (`dump.py:81-82`, `:94-95`, `:104-106`) | **dropped — deliberate, and replaced** | F-13: the sites stay as the documented writer's branch. Round 14 already refused them; this pass supplies the replacement |
| `T002`'s "`dump_turn`'s inline record construction (relocated into the extracted helpers, not duplicated)" | **carried** | F-22: the extraction still relocates rather than duplicates, now across both writers |
| `T005`'s two `REMOVED` clauses (the layout block; the "Reading the index" sample output) | **dropped — deliberate, and replaced** | F-23: the README gains a section; nothing is removed |
| `T005`'s `ADDED` clauses (the artifact's name and shape, each shipped encoding and its selection, the reading workflow, `F-8`'s limitation, the pre-existing-files note) | **carried, extended** | F-23: "each shipped encoding" now includes the documented format, and the *layout* block stays rather than being replaced |
| `G-001`'s "reduction `= 1 − F/(2N)`, i.e. 50% ≤ reduction < 100% **for every achievable F**" | **carried, and scoped** | Per selected format; the default path's clause is the open reading (F-19) |
| `G-001`'s failure witnesses A and B | **carried** | Unchanged; witness C (`2N + F`) added |
| `G-002`/`G-004`/`G-005`'s unconditional clauses | **carried, scoped** | F-15: collapsed-format-only under S-1 |
| `G-006`'s "without additional tooling" clause, per encoding (`F-11`) | **carried, extended** | Now per **format**; the documented format's workflow is unchanged |
| `B-001`'s "the layout is *replaced*, the readable-layout **function** is preserved" | **dropped — deliberate** | F-14: the layout is **retained**, so "replaced" is false. Replaced by "retained as a selectable format; the collapsed artifact is added" |
| `VP-T002-default-encoding`'s single clause | **held** | F-16: the clause inverts with the reading; the row is annotated HELD, the handle is not retired |
| `FK-002`'s "one bulk flush is ONE write" | **carried** | `US-012` does not retract it; it is what F-13 is derived from |
| `FK-002`'s three `Factual claims` | **carried verbatim** | `decisions.md#fk-002`, untouched; the revision is normative only |
| `US-007`'s "the current change-focused format is retained rather than replaced" | **restored — now load-bearing** | §Invariant Reconciliation `US-007`/`US-008` re-read; it is continuous with `US-012` |
| `US-008`'s "Keep the existing one as one option" | **restored — now load-bearing** | Same |
| The TLS cluster in full (`T006`, `MS-004`, `G-007`, `OI:` entry 4, the two `PI:T006`, the flag token, the proof route, the harness amendment) | **carried verbatim** | Round 15 closed it; `US-012`'s non-extension clause forbids reaching it |
| `T001`'s buffering design, `T003`'s trigger machinery, `T004`'s coverage clauses (other than the additions above) | **carried** | Unchanged |
| The 12 findings `F-1`…`F-12`, the research increment, the capacity finding | **carried verbatim** | Not re-derived; cited |
| All other clauses of the plan, the FK records and the US records | **carried verbatim** | Untouched by this pass |

## Compaction Decision

- Prior same-phase files reviewed: `002-organize.md`, `005-organize.md`, `006-organize.md`, `013-organize.md`, `016-organize.md`.
- **Superseded: none.**
- **Not superseded: all five.** Slice reasoning: `002` organized the original payload/layout cluster and the Anchor Map draft; `005` the flush-policy cluster; `006` the payload/format cluster and the invariant reconciliation prep; `013` the **encoding** dial (`FK-006`, `G-006`) and the artifact's identity; `016` the **TLS** cluster. This pass organizes the **ordering/layout-retention** question (`US-012`) — a slice none of them covered. `013` in particular remains valid: its encoding-dial conclusions (the artifact's name carries the configured value, the identity is `(sid, encoding)`) are unaffected and are *inputs* to this pass, not superseded by it. The round-14 reroute request explicitly names rounds 2–15 as this pass's collect input, so deleting any prior report would remove an input.
- **Compaction performed: no.**
- `iterations/016-organize.md` and `iterations/017-decide.md` still describe `FK-008`/`FK-009` as open. Not edited: they are the process record of the replan that authored those forks.

## Deferred Questions

- **Under S-1, is the write reduction being opt-in acceptable?** Status: **not authored here** — `decide` owns it. If `decide` closes the reading to S-1, this is a consequence disclosure (`INTERPRETED`, surfaced at ratification), not a fork. If it judges the reading open, this is the single fork, carrying both shapes' arithmetic. Impact: determines `G-001`'s `Met when`, `G-008`'s `Met when`, `VP-T002-default-encoding`'s clause, and `PI:.../default_encoding_is_change_focused`'s truth on the day it is written.
- **What is "better" measured on?** Status: **for `decide`**, not the user — M1 and M2 are decisive and deterministic; M4 shows no improvement and must not be presented as the proof; M3 is the physical axis and is unmeasurable here. Impact: the demonstration's evidence shape at `MS-001`.
- **Where does the switch live?** Status: **for `decide`** — roadmap entry vs a later session. `US-012`'s `Impact.Roadmap` is `pending` and names `decide` as owner.
- No other deferred questions. `US-013`'s delegation covers the remaining agent-decidable items, and the escalation floor bars extending it to anything `US-012` bounds.

## Gates Found Unmet (reported, not cured)

**Count reported to the orchestrator: `gates found unmet 4`.**

1. **`OI:` entry 1** — its `Decision` cell read `override`, which contradicts the current authority (F-14). Corrected to `refine` and left **contingent**; the replacement statement is `decide`'s to finalize. Its `User approved` cell stays **refused**.
2. **`OI:` entries 2–3** — still contingent on the payload replan; this pass narrows what they must say (the documented path is preserved) but does not stamp them.
3. **`G-001`, `G-008`, `VP-T002-default-encoding` and `PI:.../default_encoding_is_change_focused`** are **fork-dependent** and cannot be finalized until the reading is settled.
4. **Final approval is blocked.** The plan is still **not implementable**: entry 1 is refused, entries 2–3 contingent, and the reading open. The implementability mark stays **absent**. `decide` must land the reading before the invariant alignment pass can be re-run as a whole.

*(Not a gate — informational: the successor FK for the payload ordering and `FK-002`'s `State`/`Superseded-by` remain unallocated and `decide`'s. No ID was reserved or written by this pass.)*

## Validation Performed

| Claim | How verified | Result |
|---|---|---|
| Session identity / handle suffix | `session-identity.json` vs handle | matches (`9bf7afff5573`) |
| No parent coordination contract | `session-identity.json` has no `parent_session_uid` | confirmed |
| No prior `## Interview Alignment` to revoke | read of `handoff-context.md` in full | confirmed — no such block exists |
| `US-012`'s premise holds — the plan removes the old write path in the same increment that adds the new format | `plan-summary.md` §Task List `T002` `REMOVED` clauses vs `dump.py` | confirmed |
| The three write sites exist and are the whole set | `grep -n "write_text\|open(" buster_ripper/dump.py` | confirmed — exactly `:82`, `:95`, `:105` |
| `dump.py:63`'s bare assert | read of `dump.py:63` | confirmed — `assert config.DUMP_DIR is not None` |
| `dump.py:83` is the rendered-text byte measure | read of `dump.py:77`, `:83` | confirmed — `len(current_text)` where `current_text = _render_body(body)` |
| `dump_turn` is 34 executable statements (hard L violation) | statement count over `dump.py:61-109` | confirmed — L:34, N:1, C:≈6, P:1 |
| `dump.py` is 109 lines (Green) | `wc -l` | confirmed |
| `README.md:146-162` is the layout block + "Reading the index" | read with line numbers | confirmed — block `:146-152`, workflow `:154-159`, trailing note `:161-162` |
| **Citation nit (not a backlog finding)** | `handoff-context.md` cites the read path as `README.md:153-162`; the plan cites `README.md:146-162`; `:153` is blank | both resolve to the same section; neither misled an agent. Recorded, not filed |
| `RB-004`'s widened name search returns zero selector matches | `grep -rniE "dump_format\|dumpformat\|dump_mode\|dumpmode\|output_format\|output_layout\|layout\|per-turn\|per_turn\|collapsed\|artifact"` over `buster_ripper/` + `tests/` | confirmed — one prose hit (`main.py:24`), zero selector matches |
| No `--dump-format` flag exists; `--dump-dir` is the only dump option | `grep -rn '"--' buster_ripper/main.py` | confirmed — 21 options, one dump option |
| `aiohttp` signal handling is native and reaches `on_cleanup` | installed aiohttp 3.13.3 source | confirmed — `web_runner.py:283-284` → `:35-40` → `web.py:511` → `:425` |
| No invariant registry exists, so every `Registry: none` is accurate | `ls -d invariants .invariants` | confirmed — both absent |
| `/Volumes/data` is unmounted, so the byte figures are carried not re-derived | `ls -d /Volumes/data` | confirmed — absent; the figures are cited from `006`/`007`, not re-measured |
| No new `US-` reserved | the dispatch's `Reason` is verbatim `US-012`'s own `User words` | confirmed — reserving would duplicate an existing canonical US |
| No `FK-` ID authored, reserved, superseded or reused | this pass wrote no `decisions.md` field | confirmed |
| Both context files within caps | `wc -l` after this pass | see §Compaction Decision — re-verified after the plan edits |

## Next Step

1. **`decide`** — reserve and fill the successor FK **only if** the reading survives its closure gate; otherwise close to S-1 and record the opt-in consequence as `INTERPRETED`. Set `FK-002`'s `State`/`Superseded-by`. Settle F-24's axis design. Write the switch's destination (`roadmap.md` entry or later session) per `US-012`'s `Impact.Roadmap`.
2. **`invite`** — present the landscape **and** the first question in **one** pass. Do not re-present `OI:` entry 1's replacement statement unchanged; do not re-ask what `US-012` refused or `US-013` delegated.
3. **`interview`** — record the answer, then re-run the invariant alignment pass as a whole (entries 1–3 and the four `PI:T002` become presentable once the shape is settled).
4. Final approval stays blocked until the reading is settled and entries 1–3 are approved.
